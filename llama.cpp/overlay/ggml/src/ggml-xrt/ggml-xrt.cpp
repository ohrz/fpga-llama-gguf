// SPDX-License-Identifier: Apache-2.0
//
// ggml backend for the gemvqc16 command-list kernel on the Alveo U50 (AMD XRT).
//
// The kernel (fpga/src) executes lists of commands on FP32 vectors in an activation buffer: GEMV for
// Q4_0/Q8_0/Q5_K/Q6_K weights, RMSNorm, add, SwiGLU, convolution and the Gated DeltaNet step. This backend repacks
// the weights of every supported tensor once (gemvqw_layout.h), uploads them into per-engine HBM arenas, turns
// contiguous runs of supported ggml nodes into one command list and launches the kernel once per list.
// Everything the kernel cannot run stays on the CPU backend.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml-xrt.h"
#include "xrt-device-selector.h"
#include "gemvq_ref.h"
#include "gemvqw_layout.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_ip.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <sys/mman.h>

static struct ggml_backend_reg xrt_reg_instance;
static struct ggml_backend_device ggml_xrt_device_inst;

static bool xrt_debug_enabled() {
    const char * value = std::getenv("GGML_XRT_DEBUG");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// GGML_XRT_VERIFY_CALLS=<n>: compare the first n offloaded operations with the CPU reference.
static int xrt_verify_call_count() {
    const char * value = std::getenv("GGML_XRT_VERIFY_CALLS");
    return value != nullptr && value[0] != '\0' ? std::max(0, std::atoi(value)) : 0;
}

// Upper bound for one kernel launch in milliseconds (GGML_XRT_WAIT_TIMEOUT_MS, default 180000).
static unsigned int xrt_wait_timeout_ms() {
    const char * value = std::getenv("GGML_XRT_WAIT_TIMEOUT_MS");
    if (value == nullptr) return 180000;
    char * end = nullptr;
    const unsigned long timeout = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || timeout == 0 || timeout > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("GGML_XRT_WAIT_TIMEOUT_MS must be a positive integer");
    }
    return static_cast<unsigned int>(timeout);
}

static std::string xrt_kernel_name() {
    const char * value = std::getenv("GGML_XRT_KERNEL");
    return value != nullptr && value[0] != '\0' ? value : "gemvqc16";
}

static const char * ggml_backend_xrt_get_name(ggml_backend_t backend) { (void)backend; return "Alveo_U50_XRT"; }

static constexpr size_t XRT_WORD_BYTES = 64;

static size_t align_xrt_word(size_t size) {
    return (size + XRT_WORD_BYTES - 1) & ~(XRT_WORD_BYTES - 1);
}

// Identity of an uploaded weight tensor.
struct xrt_weight_key {
    const ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    enum ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, GGML_MAX_DIMS> ne{};
    std::array<size_t, GGML_MAX_DIMS> nb{};
    size_t bytes = 0;

    bool operator==(const xrt_weight_key & other) const {
        return tensor == other.tensor && data == other.data && type == other.type &&
               ne == other.ne && nb == other.nb && bytes == other.bytes;
    }
};

struct xrt_weight_key_hash {
    size_t operator()(const xrt_weight_key & key) const {
        size_t hash = std::hash<const void *>{}(key.tensor);
        const auto mix = [&hash](size_t value) { hash ^= value + 0x9e3779b9 + (hash << 6) + (hash >> 2); };
        mix(std::hash<const void *>{}(key.data));
        mix(std::hash<int>{}(key.type));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            mix(std::hash<int64_t>{}(key.ne[i]));
            mix(std::hash<size_t>{}(key.nb[i]));
        }
        mix(std::hash<size_t>{}(key.bytes));
        return hash;
    }
};

// gemvqc16 command-list kernel: 16 word-wide engines, engine e reads its own HBM port. Every weight tensor is
// repacked once (gemvqw_layout.h) into the 16 engine arenas at a common word offset; consecutive supported ggml
// nodes become one command list with x and y in the activation buffer.
static constexpr unsigned XRT_QC_ENGINES = 16;
static constexpr size_t XRT_QC_ARENA_BYTES = 250ULL << 20;  // per engine port on a single HBM channel (256 MiB)
// Engine ports that span an HBM channel pair (HBM[2e:2e+1], 512 MiB) get 384 MiB; the rest of a pair holds the
// activation/command buffer or one state buffer.
static constexpr size_t XRT_QC_ARENA_BYTES_PAIR = 384ULL << 20;
static constexpr size_t XRT_QC_ACT_BYTES = 16ULL << 20;     // inputs region + results of a whole split
static constexpr int64_t XRT_QC_MAX_COLS = 16384;
static constexpr int64_t XRT_QC_CMD_ROWS = 65520;           // rows per GEMV command: <= 65536 (kernel limit) and
                                                             // a multiple of 16 * 5, so chunks start at a multiple
                                                             // of the engine count (16 or 20)
// Larger matrices (the output head) are split into several commands; their weights fit next to the layers only with
// channel-pair arenas.
static constexpr int64_t XRT_QC_MAX_ROWS_SMALL = 65536;
static constexpr int64_t XRT_QC_MAX_ROWS_LARGE = 262144;
static constexpr unsigned XRT_QC_CMD_WORDS = 8;
static constexpr size_t XRT_QC_CMD_BYTES = 4096;            // up to 128 commands per launch
// DeltaNet opcodes: the activation buffer grows by a resident region (F32 weights, conv states) behind the per-list
// regions, and the recurrent states get four state ports of XRT_QC_STATE_BYTES each.
static constexpr size_t XRT_QC_ACT_BYTES_GDN = 64ULL << 20;
static constexpr size_t XRT_QC_RES_BASE = XRT_QC_ACT_BYTES / sizeof(float);  // floats
static constexpr unsigned XRT_QC_STATE_PORTS = 4;
static constexpr size_t XRT_QC_STATE_BYTES = 64ULL << 20;
static constexpr size_t XRT_QC_STATE_ROW = 512;

// A buffer object on the card, or plain host memory when the kernel runs as its C++ model (GGML_XRT_EMULATE=1).
struct xrt_qc_mem {
    xrt::bo bo;
    std::vector<uint8_t> host;
    xrt_qc_mem() = default;
    explicit xrt_qc_mem(xrt::bo b) : bo(std::move(b)) {}
    explicit xrt_qc_mem(size_t bytes) : host(bytes, 0) {}
    void write(const void * src, size_t size, size_t off) {
        if (host.empty()) bo.write(src, size, off); else std::memcpy(host.data() + off, src, size);
    }
    void read(void * dst, size_t size, size_t off) {
        if (host.empty()) bo.read(dst, size, off); else std::memcpy(dst, host.data() + off, size);
    }
    void sync(xclBOSyncDirection dir, size_t size, size_t off) {
        if (host.empty()) bo.sync(dir, size, off);
    }
    template <typename T> T map() { return host.empty() ? bo.map<T>() : reinterpret_cast<T>(host.data()); }
};

// Analysis of one gemvqc16 split (xrt_qc_build_plan): fusions, skipped nodes and use counts. It depends only on the
// graph structure, so it is kept per split and reused while the structure stays the same (xrt_qc_get_plan).
struct xrt_qc_plan {
    struct conv_plan {
        ggml_tensor * get_rows = nullptr, * concat = nullptr, * cpy = nullptr, * silu = nullptr;
        bool operator==(const conv_plan & o) const {
            return get_rows == o.get_rows && concat == o.concat && cpy == o.cpy && silu == o.silu;
        }
    };
    struct gdn_plan {
        ggml_tensor * get_rows = nullptr, * cpy = nullptr;
        bool operator==(const gdn_plan & o) const { return get_rows == o.get_rows && cpy == o.cpy; }
    };
    struct gdn_norms {
        bool l2 = false; float l2_eps = 0.0f; ggml_tensor * out = nullptr; const ggml_tensor * w = nullptr;
        const ggml_tensor * z = nullptr; float rms_eps = 0.0f;
        // gates: beta = sigmoid(beta_raw), g = softplus(alpha_raw + dt) * a
        const ggml_tensor * alpha_raw = nullptr, * beta_raw = nullptr, * dt = nullptr, * a = nullptr;
        bool operator==(const gdn_norms & o) const {
            return l2 == o.l2 && l2_eps == o.l2_eps && out == o.out && w == o.w && z == o.z && rms_eps == o.rms_eps &&
                   alpha_raw == o.alpha_raw && beta_raw == o.beta_raw && dt == o.dt && a == o.a;
        }
    };
    std::unordered_map<const ggml_tensor *, conv_plan> convs;  // key: SSM_CONV node
    std::unordered_map<const ggml_tensor *, gdn_plan> gdns;    // key: GATED_DELTA_NET node
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> conv_of_silu;
    std::unordered_map<const ggml_tensor *, int> uses;
    std::unordered_set<const ggml_tensor *> skip;
    std::unordered_map<const ggml_tensor *, std::pair<ggml_tensor *, ggml_tensor *>> swiglus;  // key: GLU node
    std::unordered_set<const ggml_tensor *> cont_folded;  // the UNARY nodes
    std::unordered_map<const ggml_tensor *, int> readers;  // non-view nodes reading a tensor (through views)
    std::unordered_map<const ggml_tensor *, gdn_norms> norms_of;  // key: GATED_DELTA_NET node
    std::unordered_set<const ggml_tensor *> add_src, defer_ok, conv_src, late_set;
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> late_gemv;
    std::unordered_map<const ggml_tensor *, int> last_use;  // filled only with_last_use
    bool operator==(const xrt_qc_plan & o) const {
        return convs == o.convs && gdns == o.gdns && conv_of_silu == o.conv_of_silu && uses == o.uses && skip == o.skip &&
               swiglus == o.swiglus && cont_folded == o.cont_folded && readers == o.readers && norms_of == o.norms_of &&
               add_src == o.add_src && defer_ok == o.defer_ok && conv_src == o.conv_src && late_set == o.late_set &&
               late_gemv == o.late_gemv && last_use == o.last_use;
    }
};

struct xrt_qc_plan_entry {
    std::vector<uint64_t> key;  // xrt_qc_plan_key: everything the analysis reads
    std::unique_ptr<xrt_qc_plan> plan;
};

struct xrt_backend_context {
    xrt::device device;
    std::string kernel_name;
    std::vector<xrt::kernel> kernels;
    int verify_calls_remaining = 0;
    bool runtime_valid = true;
    unsigned qc_engines = XRT_QC_ENGINES;
    size_t qc_arena_bytes = 0;  // weight arena per engine
    std::vector<xrt_qc_mem> qc_weight_bos;
    xrt_qc_mem qc_act_bo;
    xrt_qc_mem qc_cmd_bo;
    bool qc_emulate = false;  // kernel runs as the HLS C++ model on host memory
    xrt::run qc_run;          // all arguments fixed except the command count
    // GGML_XRT_DIRECT_START=1: the CU is started through its control registers (needs ert_disable=1), ~6 us per
    // launch instead of ~190 us through XRT; the argument registers are written once.
    bool qc_direct = false;
    xrt::ip qc_ip;
    uint32_t qc_ncmds_reg = 0;  // register offset of n_cmds
    uint32_t qc_features = 0;   // feature bits of the loaded kernel (0: Q4_0/Q8_0 only)
    size_t qc_used = 0;         // bytes used in every engine arena; the next tensor starts here in all of them
    std::unordered_map<xrt_weight_key, uint64_t, xrt_weight_key_hash> qc_offsets;  // word offset per tensor
    // DeltaNet opcodes (kernel with four state ports): recurrent states live on the card, see XRT_RS buffers.
    bool qc_gdn = false;
    std::vector<xrt_qc_mem> qc_state_bos;                  // one per state port, 512-byte row r on bo[r % 4]
    size_t qc_state_rows_used = 0;                         // global state rows handed out (multiple of 4)
    size_t qc_resident_used = 0;                           // floats used in the resident region of act
    std::unordered_map<const void *, unsigned> qc_consts;  // F32 weight data -> resident act offset
    std::vector<xrt_qc_plan_entry> qc_plans;               // split analyses by graph structure
};

// ert_disable of the device from sysfs, -1 if it cannot be read. The direct register start needs 1; loading a
// different xclbin while it is 1 deadlocks the xocl driver.
static int xrt_ert_disable(const xrt::device & device) {
    std::ifstream file("/sys/bus/pci/devices/" + device.get_info<xrt::info::device::bdf>() + "/ert_disable");
    int value = -1;
    return (file >> value) ? value : -1;
}

static void xrt_wait(xrt::run & run) {
    const unsigned int timeout = xrt_wait_timeout_ms();
    if (run.wait2(std::chrono::milliseconds(timeout)) == std::cv_status::timeout) {
        try { run.abort(); } catch (...) {}
        throw std::runtime_error("XRT kernel wait timed out after " + std::to_string(timeout) + " ms");
    }
}

// Diagnostic wall-clock accounting, enabled with GGML_XRT_TIMING=1 and printed at exit.
struct xrt_timing_totals {
    std::atomic<uint64_t> graph_ns{0}, graph_count{0}, gemv_count{0}, gemv_bytes{0};
    std::atomic<uint64_t> gemv_x_ns{0}, gemv_launch_ns{0}, gemv_wait_ns{0}, gemv_y_ns{0};
    std::atomic<uint64_t> plan_ns{0}, flush_ns{0};  // gemvqc16 split analysis; whole flushes (graph - flush = build)
    std::atomic<uint64_t> host_ns{0}, state_ns{0}, operands_ns{0};  // host-only splits, state ops, operand lookup
    bool enabled = std::getenv("GGML_XRT_TIMING") != nullptr;
    ~xrt_timing_totals() {
        if (!enabled) return;
        fprintf(stderr,
                "XRT-TIMING: graphs=%llu graph_ms=%.1f gemv=%llu gemv_weight_MB=%.1f x_ms=%.1f launch_ms=%.1f "
                "wait_ms=%.1f y_ms=%.1f plan_ms=%.1f flush_ms=%.1f host_ms=%.1f state_ms=%.1f operands_ms=%.1f\n",
                (unsigned long long) graph_count.load(), graph_ns.load() / 1e6, (unsigned long long) gemv_count.load(),
                gemv_bytes.load() / 1e6, gemv_x_ns.load() / 1e6, gemv_launch_ns.load() / 1e6,
                gemv_wait_ns.load() / 1e6, gemv_y_ns.load() / 1e6, plan_ns.load() / 1e6, flush_ns.load() / 1e6,
                host_ns.load() / 1e6, state_ns.load() / 1e6, operands_ns.load() / 1e6);
    }
};
static xrt_timing_totals g_xrt_timing;

struct xrt_scoped_ns {
    std::atomic<uint64_t> & sink;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    explicit xrt_scoped_ns(std::atomic<uint64_t> & target) : sink(target) {}
    ~xrt_scoped_ns() {
        if (g_xrt_timing.enabled) {
            sink += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        }
    }
};

// Only one backend instance owns the card at a time.
struct xrt_device_context {
    std::mutex mutex;
    xrt_backend_context * active_backend = nullptr;
};

static xrt_device_context xrt_device_context_inst;

static xrt_device_context * xrt_device_ctx(ggml_backend_dev_t dev) {
    return dev == &ggml_xrt_device_inst
        ? static_cast<xrt_device_context *>(dev->context)
        : nullptr;
}

// Largest GEMV the active kernel accepts: the output head only when its weights fit into the arenas.
static int64_t xrt_active_qc_max_rows() {
    std::lock_guard<std::mutex> lock(xrt_device_context_inst.mutex);
    const xrt_backend_context * ctx = xrt_device_context_inst.active_backend;
    const bool large = ctx != nullptr && ctx->qc_arena_bytes >= XRT_QC_ARENA_BYTES_PAIR;
    return large ? XRT_QC_MAX_ROWS_LARGE : XRT_QC_MAX_ROWS_SMALL;
}

static bool xrt_active_qc_mode() {
    std::lock_guard<std::mutex> lock(xrt_device_context_inst.mutex);
    return xrt_device_context_inst.active_backend != nullptr;
}

// Feature word of the kernel (GEMVQC_FEATURES of gemvqc16.h): marker in the high half, format bits below.
static constexpr uint32_t XRT_QC_FEATURE_MARK = 0x6B510000U;
static constexpr uint32_t XRT_QC_FEATURE_Q5_K = 1U;
static constexpr uint32_t XRT_QC_FEATURE_Q6_K = 2U;
static constexpr uint32_t XRT_QC_FEATURE_Q4_1 = 4U;
static constexpr uint32_t XRT_QC_FEATURE_SWIGLU = 8U;       // GEMV groups, SwiGLU pairs (format bit 8)
static constexpr uint32_t XRT_QC_FEATURE_STATE_LOAD = 16U;  // STATE_LOAD opcode, GDN on chip (H_v <= 32)
static constexpr uint32_t XRT_QC_FEATURE_GDN_GEMV = 32U;    // a GDN step and the GEMV behind it in one pass
static constexpr uint32_t XRT_QC_FEATURE_CONV_EPILOG = 64U; // CONV as epilog of the GEMV that produces its input
static constexpr uint32_t XRT_QC_FEATURE_GDN_NORMS = 128U;  // GDN with L2 norm of q/k and gated RMS norm of y
static constexpr uint32_t XRT_QC_FEATURE_ADD_EPILOG = 256U; // residual ADD as epilog of the GEMV
static constexpr uint32_t XRT_QC_FEATURE_NORM_DEFER = 512U; // deferred RMSNORM: GEMVs quantize the row from the chip
static constexpr uint32_t XRT_QC_FEATURE_GDN_GATES = 1024U; // GDN computes its gates from the raw projections
static constexpr uint32_t XRT_QC_FEATURE_STRIDED = 2048U;   // strided rows in RMSNORM and UNARY, UNARY copy
static constexpr uint32_t XRT_QC_FEATURE_ROPE = 4096U;      // ROPE opcode (rope cache from the host)
static constexpr uint32_t XRT_QC_FEATURE_Q6K_PACKED = 8192U; // Q6_K in the packed word layout (gemvqw_layout.h)

static uint32_t xrt_active_qc_features() {
    std::lock_guard<std::mutex> lock(xrt_device_context_inst.mutex);
    const xrt_backend_context * ctx = xrt_device_context_inst.active_backend;
    return ctx != nullptr ? ctx->qc_features : 0U;
}

// Kernel format code (the ggml type id) of a weight type, 0 if a kernel with these features cannot run it.
static unsigned xrt_qc_format(enum ggml_type type, uint32_t features) {
    if (type == GGML_TYPE_Q4_0) return 2U;
    if (type == GGML_TYPE_Q8_0) return 8U;
    if (type == GGML_TYPE_Q4_1 && (features & XRT_QC_FEATURE_Q4_1)) return 3U;
    if (type == GGML_TYPE_Q5_K && (features & XRT_QC_FEATURE_Q5_K)) return 13U;
    if (type == GGML_TYPE_Q6_K && (features & XRT_QC_FEATURE_Q6_K)) return 14U;
    return 0U;
}

static void xrt_activate_backend(xrt_backend_context * ctx) {
    std::lock_guard<std::mutex> lock(xrt_device_context_inst.mutex);
    if (xrt_device_context_inst.active_backend != nullptr) {
        throw std::runtime_error("only one XRT backend instance may be active at a time");
    }
    xrt_device_context_inst.active_backend = ctx;
}

// Buffer types of the recurrent-state buffers are device-private (never host-accessible).
static size_t xrt_rs_buft_alignment(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return XRT_WORD_BYTES;
}

static size_t xrt_rs_buft_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    (void) buft;
    return align_xrt_word(ggml_nbytes(tensor));
}

static bool xrt_rs_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return false;
}

// Recurrent-state buffers (llama.cpp's r_l / s_l caches) for a gemvqc16 kernel with DeltaNet opcodes. The states
// live only on the card: s_l tensors are striped over the four state ports in 512-byte rows (global row r on port
// r % 4 at byte (r / 4) * 512), r_l tensors (conv states) sit in the resident region of act. ggml sees an address
// range reserved without access rights, so a CPU op that touches it faults instead of using stale data; reads and
// writes go through get/set_tensor or the backend's own ops.
struct xrt_rs_tensor {
    size_t off = 0, bytes = 0;  // within the buffer
    bool striped = false;
    size_t card = 0;            // first global state row (striped) or float offset in act
};

struct xrt_rs_buffer_context {
    xrt_backend_context * owner = nullptr;
    uint8_t * base = nullptr;
    size_t size = 0;
    size_t rows_begin = 0, rows_end = 0, res_begin = 0, res_end = 0;  // card space handed to this buffer
    std::vector<xrt_rs_tensor> tensors;
};

static xrt_rs_buffer_context * xrt_rs_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<xrt_rs_buffer_context *>(buffer->context);
}

static const xrt_rs_tensor * xrt_rs_find(const xrt_rs_buffer_context * rs, const void * data) {
    const auto * p = static_cast<const uint8_t *>(data);
    if (p < rs->base || p >= rs->base + rs->size) return nullptr;
    const size_t off = size_t(p - rs->base);
    for (const auto & t : rs->tensors) {
        if (off >= t.off && off < t.off + t.bytes) return &t;
    }
    return nullptr;
}

// Copies bytes [off, off + size) of an RS tensor between host memory and the card.
static void xrt_rs_io(xrt_backend_context * owner, const xrt_rs_tensor & t, size_t off, uint8_t * host, size_t size,
                      bool to_card) {
    if (size == 0) return;
    if (off + size > t.bytes) throw std::runtime_error("XRT_RS access out of range");
    if (!t.striped) {
        const size_t at = t.card * sizeof(float) + off;
        if (to_card) {
            owner->qc_act_bo.write(host, size, at);
            owner->qc_act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, size, at);
        } else {
            owner->qc_act_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, at);
            owner->qc_act_bo.read(host, size, at);
        }
        return;
    }
    const size_t P = XRT_QC_STATE_PORTS, R = XRT_QC_STATE_ROW;
    const size_t r0 = t.card + off / R, r1 = t.card + (off + size - 1) / R;
    // The rows of one port inside [r0, r1] are contiguous on that port: one sync per port.
    const auto sync = [&](xclBOSyncDirection dir) {
        for (size_t p = 0; p < P; ++p) {
            const size_t first = r0 + (p + P - r0 % P) % P;
            if (first > r1) continue;
            const size_t last = r1 - (r1 % P + P - p) % P;
            owner->qc_state_bos[p].sync(dir, (last / P - first / P + 1) * R, first / P * R);
        }
    };
    if (!to_card) sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (size_t done = 0; done < size;) {
        const size_t pos = off + done, row = t.card + pos / R, within = pos % R;
        const size_t chunk = std::min(R - within, size - done);
        auto & bo = owner->qc_state_bos[row % P];
        if (to_card) bo.write(host + done, chunk, row / P * R + within);
        else bo.read(host + done, chunk, row / P * R + within);
        done += chunk;
    }
    if (to_card) sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

static void xrt_rs_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = xrt_rs_ctx(buffer);
    {
        // Give the card space back when this buffer was the last one handed out (llama-bench creates and frees
        // one context per test).
        std::lock_guard<std::mutex> lock(xrt_device_context_inst.mutex);
        auto * owner = xrt_device_context_inst.active_backend;
        if (owner == ctx->owner) {
            if (owner->qc_state_rows_used == ctx->rows_end) owner->qc_state_rows_used = ctx->rows_begin;
            if (owner->qc_resident_used == ctx->res_end) owner->qc_resident_used = ctx->res_begin;
        }
    }
    munmap(ctx->base, ctx->size);
    delete ctx;
}

static void * xrt_rs_buffer_get_base(ggml_backend_buffer_t buffer) { return xrt_rs_ctx(buffer)->base; }

static enum ggml_status xrt_rs_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src != nullptr) return GGML_STATUS_SUCCESS;
    auto * rs = xrt_rs_ctx(buffer);
    auto * owner = rs->owner;
    xrt_rs_tensor t;
    t.off = size_t(static_cast<uint8_t *>(tensor->data) - rs->base);
    t.bytes = ggml_nbytes(tensor);
    t.striped = std::strncmp(tensor->name, "cache_s_l", 9) == 0;
    if (t.striped) {
        const size_t rows = (t.bytes + XRT_QC_STATE_ROW - 1) / XRT_QC_STATE_ROW;
        const size_t padded = (rows + XRT_QC_STATE_PORTS - 1) / XRT_QC_STATE_PORTS * XRT_QC_STATE_PORTS;
        if ((owner->qc_state_rows_used + padded) * XRT_QC_STATE_ROW > XRT_QC_STATE_BYTES * XRT_QC_STATE_PORTS) {
            fprintf(stderr, "XRT: state ports full at %s\n", tensor->name);
            return GGML_STATUS_ALLOC_FAILED;
        }
        t.card = owner->qc_state_rows_used;
        owner->qc_state_rows_used += padded;
        rs->rows_end = owner->qc_state_rows_used;
    } else {
        const size_t floats = (t.bytes / sizeof(float) + 15U) / 16U * 16U;
        if (XRT_QC_RES_BASE + owner->qc_resident_used + floats > XRT_QC_ACT_BYTES_GDN / sizeof(float)) {
            fprintf(stderr, "XRT: resident act region full at %s\n", tensor->name);
            return GGML_STATUS_ALLOC_FAILED;
        }
        t.card = XRT_QC_RES_BASE + owner->qc_resident_used;
        owner->qc_resident_used += floats;
        rs->res_end = owner->qc_resident_used;
    }
    rs->tensors.push_back(t);
    return GGML_STATUS_SUCCESS;
}

static const xrt_rs_tensor & xrt_rs_tensor_of(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, size_t offset,
                                              size_t * within) {
    const auto * t = xrt_rs_find(xrt_rs_ctx(buffer), static_cast<const uint8_t *>(tensor->data) + offset);
    if (t == nullptr) throw std::runtime_error(std::string("XRT_RS tensor not registered: ") + tensor->name);
    *within = size_t(static_cast<const uint8_t *>(tensor->data) - xrt_rs_ctx(buffer)->base) + offset - t->off;
    return *t;
}

static void xrt_rs_buffer_memset(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    size_t within = 0;
    const auto & t = xrt_rs_tensor_of(buffer, tensor, offset, &within);
    std::vector<uint8_t> fill(size, value);
    xrt_rs_io(xrt_rs_ctx(buffer)->owner, t, within, fill.data(), size, true);
}

static void xrt_rs_buffer_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    size_t within = 0;
    const auto & t = xrt_rs_tensor_of(buffer, tensor, offset, &within);
    xrt_rs_io(xrt_rs_ctx(buffer)->owner, t, within, static_cast<uint8_t *>(const_cast<void *>(data)), size, true);
}

static void xrt_rs_buffer_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    size_t within = 0;
    const auto & t = xrt_rs_tensor_of(buffer, tensor, offset, &within);
    xrt_rs_io(xrt_rs_ctx(buffer)->owner, t, within, static_cast<uint8_t *>(data), size, false);
}

static void xrt_rs_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * rs = xrt_rs_ctx(buffer);
    for (const auto & t : rs->tensors) {
        std::vector<uint8_t> fill(t.bytes, value);
        xrt_rs_io(rs->owner, t, 0, fill.data(), t.bytes, true);
    }
}

static const ggml_backend_buffer_i xrt_rs_buffer_iface = {
    .free_buffer = xrt_rs_buffer_free,
    .get_base = xrt_rs_buffer_get_base,
    .init_tensor = xrt_rs_buffer_init_tensor,
    .memset_tensor = xrt_rs_buffer_memset,
    .set_tensor = xrt_rs_buffer_set,
    .get_tensor = xrt_rs_buffer_get,
    .set_tensor_2d = nullptr,
    .get_tensor_2d = nullptr,
    .cpy_tensor = nullptr,
    .clear = xrt_rs_buffer_clear,
    .reset = nullptr,
};

static const char * xrt_rs_buft_name(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return "XRT_RS";
}

static ggml_backend_buffer_t xrt_rs_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    std::lock_guard<std::mutex> lock(xrt_device_context_inst.mutex);
    auto * owner = xrt_device_context_inst.active_backend;
    if (owner == nullptr || !owner->qc_gdn) return nullptr;
    const size_t reserved = std::max<size_t>(size, 1);
    void * base = mmap(nullptr, reserved, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) return nullptr;
    auto * ctx = new xrt_rs_buffer_context();
    ctx->owner = owner;
    ctx->base = static_cast<uint8_t *>(base);
    ctx->size = reserved;
    ctx->rows_begin = ctx->rows_end = owner->qc_state_rows_used;
    ctx->res_begin = ctx->res_end = owner->qc_resident_used;
    auto buffer = ggml_backend_buffer_init(buft, xrt_rs_buffer_iface, ctx, size);
    if (buffer == nullptr) {
        munmap(base, reserved);
        delete ctx;
    }
    return buffer;
}

static const ggml_backend_buffer_type_i xrt_rs_buft_iface = {
    .get_name = xrt_rs_buft_name,
    .alloc_buffer = xrt_rs_buft_alloc,
    .get_alignment = xrt_rs_buft_alignment,
    .get_max_size = nullptr,
    .get_alloc_size = xrt_rs_buft_alloc_size,
    .is_host = xrt_rs_buft_is_host,
};

static ggml_backend_buffer_type xrt_rs_buft = { xrt_rs_buft_iface, &ggml_xrt_device_inst, nullptr };

static bool xrt_is_rs_buffer(const ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->buft == &xrt_rs_buft;
}

// RS buffer of a tensor or of the tensor it views (views get their buffer only when the graph is allocated).
static ggml_backend_buffer_t xrt_rs_buffer_of(const ggml_tensor * t) {
    if (t == nullptr) return nullptr;
    const ggml_backend_buffer_t b = t->view_src != nullptr ? t->view_src->buffer : t->buffer;
    return xrt_is_rs_buffer(b) ? b : nullptr;
}

static void ggml_backend_xrt_free(ggml_backend_t backend) {
    auto * ctx = static_cast<xrt_backend_context *>(backend->context);
    if (xrt_debug_enabled()) {
        fprintf(stderr, "XRT-DEBUG: backend free context=%p\n", static_cast<void *>(ctx));
    }
    ctx->qc_offsets.clear();
    ctx->qc_weight_bos.clear();
    {
        std::lock_guard<std::mutex> lock(xrt_device_context_inst.mutex);
        if (xrt_device_context_inst.active_backend == ctx) {
            xrt_device_context_inst.active_backend = nullptr;
        }
    }
    delete ctx;
    free(backend);
}

static xrt_weight_key xrt_make_weight_key(const ggml_tensor * weights, size_t bytes) {
    xrt_weight_key key;
    key.tensor = weights;
    key.data = weights->data;
    key.type = weights->type;
    key.bytes = bytes;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        key.ne[i] = weights->ne[i];
        key.nb[i] = weights->nb[i];
    }
    return key;
}

// Single-vector MUL_MAT accepted by gemvqc16 (contiguous GGUF rows): Q4_0/Q8_0 (and Q4_1 on kernels with that
// feature) with cols a multiple of 128, Q5_K/Q6_K (kernels with those features) with cols a multiple of 256.
static bool is_single_vector_gemvqc(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const int64_t max_rows = xrt_active_qc_max_rows();
    const unsigned format = xrt_qc_format(src0->type, xrt_active_qc_features());
    if (format == 0) {
        return false;
    }
    const int64_t cols = src0->ne[0];
    const int64_t rows = src0->ne[1];
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           cols > 0 && cols <= XRT_QC_MAX_COLS && (cols % (format >= 13U ? 256 : 128)) == 0 && rows > 0 && rows <= max_rows &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[0] == cols && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           dst->ne[0] == rows && dst->ne[1] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 &&
           src0->nb[0] == ggml_type_size(src0->type) && src0->nb[1] == ggml_row_size(src0->type, cols) &&
           src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float);
}

// Repacks and uploads a weight tensor once; returns its word offset, common to all engine arenas.
static uint64_t xrt_qc_weight_offset(xrt_backend_context * ctx, const ggml_tensor * weights) {
    const xrt_weight_key key = xrt_make_weight_key(weights, ggml_nbytes(weights));
    if (const auto it = ctx->qc_offsets.find(key); it != ctx->qc_offsets.end()) {
        return it->second;
    }
    const unsigned rows = unsigned(weights->ne[1]), cols = unsigned(weights->ne[0]);
    const unsigned format = xrt_qc_format(weights->type, ctx->qc_features);
    const bool q6k_packed = (ctx->qc_features & XRT_QC_FEATURE_Q6K_PACKED) != 0;
    const auto arenas = gemvqw_layout::repack(static_cast<const uint8_t *>(weights->data), rows, cols, format,
                                              ctx->qc_engines, q6k_packed);
    size_t need = 0;
    for (const auto & arena : arenas) need = std::max(need, align_xrt_word(arena.size()));
    if (ctx->qc_used + need > ctx->qc_arena_bytes) {
        throw std::runtime_error(std::string("gemvqc16 weight arenas full at tensor ") + weights->name);
    }
    for (unsigned e = 0; e < ctx->qc_engines; ++e) {
        if (arenas[e].empty()) continue;
        ctx->qc_weight_bos[e].write(arenas[e].data(), arenas[e].size(), ctx->qc_used);
        ctx->qc_weight_bos[e].sync(XCL_BO_SYNC_BO_TO_DEVICE, align_xrt_word(arenas[e].size()), ctx->qc_used);
    }
    const uint64_t offset_words = ctx->qc_used / XRT_WORD_BYTES;
    ctx->qc_used += need;
    ctx->qc_offsets.emplace(key, offset_words);
    if (xrt_debug_enabled()) {
        fprintf(stderr, "XRT-DEBUG: gemvqc16 upload %s fmt=%u rows=%u cols=%u offset_words=%llu used=%.1f MiB/engine\n",
                weights->name, format, rows, cols, (unsigned long long) offset_words,
                double(ctx->qc_used) / (1024.0 * 1024.0));
    }
    return offset_words;
}

static size_t xrt_qc_round16(size_t floats) { return (floats + 15U) / 16U * 16U; }

// Activation buffer split: external inputs (from host memory) live below XRT_QC_OUT_BASE, results above it.
static constexpr size_t XRT_QC_OUT_BASE = 256U * 1024U;                     // floats (1 MiB of inputs)
static constexpr size_t XRT_QC_ACT_FLOATS = XRT_QC_ACT_BYTES / sizeof(float);
static constexpr unsigned XRT_QC_OP_GEMV = 1, XRT_QC_OP_RMSNORM = 2, XRT_QC_OP_ADD = 3, XRT_QC_OP_SILU_MUL = 4;

static bool xrt_qc_is_vector(const ggml_tensor * t, int64_t n) {
    return t != nullptr && t->type == GGML_TYPE_F32 && t->ne[0] == n && t->ne[1] == 1 && t->ne[2] == 1 &&
           t->ne[3] == 1 && t->nb[0] == sizeof(float) && ggml_is_contiguous(t);
}

static bool xrt_qc_vector_len_ok(int64_t n) { return n > 0 && n <= XRT_QC_MAX_COLS && (n % 16) == 0; }

// Single-token ops gemvqc16 can execute: Q4_0/Q8_0 GEMV, RMS_NORM, the weight MUL right after an RMS_NORM,
// vector ADD and split SwiGLU.
static bool xrt_qc_supports(const ggml_tensor * op) {
    const int64_t n = op->ne[0];
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return is_single_vector_gemvqc(op->src[0], op->src[1], op);
        case GGML_OP_RMS_NORM:
            return xrt_qc_vector_len_ok(n) && xrt_qc_is_vector(op, n) && xrt_qc_is_vector(op->src[0], n);
        case GGML_OP_MUL:
            return xrt_qc_vector_len_ok(n) && xrt_qc_is_vector(op, n) && op->src[0]->op == GGML_OP_RMS_NORM &&
                   xrt_qc_is_vector(op->src[0], n) && xrt_qc_is_vector(op->src[1], n);
        case GGML_OP_ADD:
            // Short adds (e.g. 32-element DeltaNet gates) sit between CPU ops; offloading them would only cut the
            // CPU region into more splits. Residual adds are hidden-size vectors.
            return n >= 256 && xrt_qc_vector_len_ok(n) && xrt_qc_is_vector(op, n) &&
                   xrt_qc_is_vector(op->src[0], n) && xrt_qc_is_vector(op->src[1], n);
        case GGML_OP_GLU:
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && ggml_get_op_params_i32(op, 1) == 0 &&
                   xrt_qc_vector_len_ok(n) && xrt_qc_is_vector(op, n) && xrt_qc_is_vector(op->src[0], n) &&
                   xrt_qc_is_vector(op->src[1], n);
        default:
            return false;
    }
}

static constexpr unsigned XRT_QC_OP_MUL = 5, XRT_QC_OP_UNARY = 6, XRT_QC_OP_GDN = 7, XRT_QC_OP_CONV = 8;
static constexpr unsigned XRT_QC_OP_STATE_LOAD = 9, XRT_QC_GEMV_SWIGLU = 0x100, XRT_QC_GDN_WITH_GEMV = 1;
static constexpr unsigned XRT_QC_GEMV_CONV = 0x200, XRT_QC_GEMV_ADD = 0x400, XRT_QC_GDN_L2_QK = 2, XRT_QC_GDN_GATED = 4;
static constexpr unsigned XRT_QC_GDN_GATES = 8;
static constexpr unsigned XRT_QC_OP_ROPE = 10;
static constexpr unsigned XRT_QC_NORM_L2 = 1, XRT_QC_NORM_NO_WEIGHT = 2;
static constexpr unsigned XRT_QC_UNARY_SILU = 1, XRT_QC_UNARY_SIGMOID = 2, XRT_QC_UNARY_SOFTPLUS = 3, XRT_QC_UNARY_COPY = 4;
static constexpr int64_t XRT_QC_MAX_EW = 4096 * 16;  // element-wise length limit of the on-chip buffers
static constexpr int64_t XRT_QC_GDN_HEAD = 128;

static bool xrt_active_qc_gdn() {
    std::lock_guard<std::mutex> lock(xrt_device_context_inst.mutex);
    return xrt_device_context_inst.active_backend != nullptr && xrt_device_context_inst.active_backend->qc_gdn;
}

// Contiguous F32 tensor of a single token: [n] vectors or per-head [head size, heads]. Prompt batches carry a
// token dimension and stay on the CPU.
static bool xrt_qc_token_tensor(const ggml_tensor * t) {
    const int64_t n = t != nullptr ? ggml_nelements(t) : 0;
    return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[2] == 1 && t->ne[3] == 1 &&
           (t->ne[1] == 1 || t->ne[0] == 1 || t->ne[0] == 128 || t->ne[0] == 256) && n > 0 && n % 16 == 0 &&
           n <= XRT_QC_MAX_EW;
}

static bool xrt_qc_unary_kind(const ggml_tensor * op, unsigned * kind) {
    switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_SILU: *kind = XRT_QC_UNARY_SILU; return true;
        case GGML_UNARY_OP_SIGMOID: *kind = XRT_QC_UNARY_SIGMOID; return true;
        case GGML_UNARY_OP_SOFTPLUS: *kind = XRT_QC_UNARY_SOFTPLUS; return true;
        default: return false;
    }
}

// ggml's rope cache for one position (ggml-cpu ops.cpp: rope_yarn, ggml_rope_cache_init, ggml_mrope_cache_init),
// n_dims floats: cos and sin of pair i at 2i, 2i + 1. The kernel's ROPE rotates with it exactly as ggml does.
static float xrt_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
    return 1 - std::min(1.0f, std::max(0.0f, y));
}

static void xrt_rope_yarn(float theta_extrap, float freq_scale, const float corr_dims[2], int64_t i0, float ext_factor,
                          float mscale, float * cos_theta, float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = xrt_rope_yarn_ramp(corr_dims[0], corr_dims[1], int(i0)) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static std::vector<float> xrt_rope_cache(const ggml_tensor * op) {
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];
    const int n_dims = ggml_get_op_params_i32(op, 1), mode = ggml_get_op_params_i32(op, 2);
    const int n_ctx_orig = ggml_get_op_params_i32(op, 4);
    memcpy(&freq_base, (const int32_t *) op->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (const int32_t *) op->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (const int32_t *) op->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (const int32_t *) op->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (const int32_t *) op->op_params + 10, sizeof(float));
    memcpy(&sections, (const int32_t *) op->op_params + 11, sizeof(int) * 4);
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    const int32_t * pos = static_cast<const int32_t *>(op->src[1]->data);
    const int64_t ne0 = op->ne[0], ne2 = op->ne[2];
    std::vector<float> cache(static_cast<size_t>(ne0));
    if (!(mode & GGML_ROPE_TYPE_MROPE)) {
        float theta = float(pos[0]);
        for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            xrt_rope_yarn(theta, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cache[i0], &cache[i0 + 1]);
            theta *= theta_scale;
        }
    } else {
        const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
        float theta_t = float(pos[0]), theta_h = float(pos[ne2]), theta_w = float(pos[ne2 * 2]), theta_e = float(pos[ne2 * 3]);
        const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        const int sec_w = sections[1] + sections[0];
        for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const int sector = int((i0 / 2) % sect_dims);
            float theta = theta_t;
            if (is_imrope) {
                if (sector % 3 == 1 && sector < 3 * sections[1]) theta = theta_h;
                else if (sector % 3 == 2 && sector < 3 * sections[2]) theta = theta_w;
                else if (sector % 3 == 0 && sector < 3 * sections[0]) theta = theta_t;
                else theta = theta_e;
            } else {
                if (sector >= sections[0] && sector < sec_w) theta = theta_h;
                else if (sector >= sec_w && sector < sec_w + sections[2]) theta = theta_w;
                else if (sector >= sec_w + sections[2]) theta = theta_e;
            }
            xrt_rope_yarn(theta, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cache[i0], &cache[i0 + 1]);
            theta_t *= theta_scale;
            theta_w *= theta_scale;
            theta_h *= theta_scale;
            theta_e *= theta_scale;
        }
    }
    cache.resize(size_t(n_dims));
    return cache;
}

// Rows of a single token read with a stride (e.g. the q or gate half of each head of the attention Q projection):
// F32, rows of a multiple of 16 floats every nb[1] bytes (a whole number of 64-byte words).
static bool xrt_qc_strided_rows(const ggml_tensor * t) {
    return (xrt_active_qc_features() & XRT_QC_FEATURE_STRIDED) && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) &&
           t->ne[0] % 16 == 0 && t->nb[1] % 64 == 0 && t->nb[1] >= size_t(t->ne[0]) * sizeof(float) &&
           t->nb[1] / 64 <= 0xffff && t->ne[2] == 1 && t->ne[3] == 1 && ggml_nelements(t) <= XRT_QC_MAX_EW &&
           ggml_nelements(t) <= int64_t(XRT_QC_MAX_COLS) * 4;
}

// Floats from the first to the last element of strided rows (the act range they occupy).
static size_t xrt_qc_rows_span(const ggml_tensor * t) {
    return size_t(t->ne[1] - 1) * (t->nb[1] / sizeof(float)) + size_t(t->ne[0]);
}

// True if t is (a view of) GET_ROWS from an XRT_RS state: the conv and GDN ops run here only then.
static bool xrt_qc_rs_rows(const ggml_tensor * t) {
    while (t != nullptr && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) t = t->src[0];
    return t != nullptr && t->op == GGML_OP_GET_ROWS && xrt_rs_buffer_of(t->src[0]) != nullptr;
}

// Ops of a gemvqc16 kernel with DeltaNet opcodes: the whole single-token DeltaNet chain (gate vectors, conv,
// L2 norms, the GDN step, the gated RMS norm) besides GEMV, norms, adds and SwiGLU. Reads, writes and clears of
// XRT_RS states always run here, for prompts too (the CPU cannot touch them).
static bool xrt_qc_supports_gdn(const ggml_tensor * op) {
    if ((op->op == GGML_OP_GET_ROWS && xrt_rs_buffer_of(op->src[0]) != nullptr && op->src[0]->type == GGML_TYPE_F32 &&
         op->src[1]->type == GGML_TYPE_I32) ||
        (op->op == GGML_OP_CPY && xrt_rs_buffer_of(op->src[1]) != nullptr && op->src[0]->type == GGML_TYPE_F32 &&
         op->src[1]->type == GGML_TYPE_F32) ||
        (op->op == GGML_OP_SCALE && xrt_rs_buffer_of(op->src[0]) != nullptr && op->type == GGML_TYPE_F32)) {
        return true;
    }
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    switch (op->op) {
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return xrt_qc_token_tensor(op) && (xrt_qc_token_tensor(a) || xrt_qc_strided_rows(a)) && op->ne[0] % 16 == 0;
        case GGML_OP_ROPE: {  // one token, NEOX pairing (NEOX, MROPE, IMROPE), no frequency factors
            const int n_dims = ggml_get_op_params_i32(op, 1), mode = ggml_get_op_params_i32(op, 2);
            return (xrt_active_qc_features() & XRT_QC_FEATURE_ROPE) && op->src[2] == nullptr && xrt_qc_token_tensor(op) &&
                   xrt_qc_token_tensor(a) && b->type == GGML_TYPE_I32 && op->ne[2] == 1 &&
                   (mode == GGML_ROPE_TYPE_NEOX || mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE) &&
                   n_dims % 32 == 0 && n_dims <= 128 && n_dims <= op->ne[0] && op->ne[0] % 16 == 0;
        }
        case GGML_OP_CONT:  // copy of strided rows (a UNARY copy, or folded into the unary that reads it)
            return xrt_qc_token_tensor(op) && xrt_qc_strided_rows(a) && ggml_nelements(op) == ggml_nelements(a);
        case GGML_OP_MUL:
            // same-size product, or the weight of a row-wise RMS_NORM broadcast over its rows
            if (!xrt_qc_token_tensor(op) || !xrt_qc_token_tensor(a) || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(b)) {
                return false;
            }
            return ggml_nelements(b) == ggml_nelements(op) ||
                   (a->op == GGML_OP_RMS_NORM && (xrt_qc_token_tensor(a->src[0]) || xrt_qc_strided_rows(a->src[0])) &&
                    ggml_nelements(b) == op->ne[0] &&
                    op->ne[0] % 16 == 0);
        case GGML_OP_ADD:
            return xrt_qc_token_tensor(op) && xrt_qc_token_tensor(a) && b->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(b) && ggml_nelements(b) == ggml_nelements(op);
        case GGML_OP_UNARY: {
            unsigned kind = 0;
            return xrt_qc_unary_kind(op, &kind) && xrt_qc_token_tensor(op) &&
                   (xrt_qc_token_tensor(a) || (a->op == GGML_OP_CONT && xrt_qc_strided_rows(a->src[0])));
        }
        case GGML_OP_CONCAT:  // conv window of one token: [d_conv - 1, ch] state and the [1, ch] input
            return ggml_get_op_params_i32(op, 0) == 0 && op->type == GGML_TYPE_F32 && op->ne[0] == 4 &&
                   op->ne[2] == 1 && op->ne[3] == 1 && op->ne[1] % 16 == 0 && a->ne[0] == 3 && b->ne[0] == 1 &&
                   xrt_qc_rs_rows(a);
        case GGML_OP_SSM_CONV:
            return a->ne[0] == 4 && a->ne[2] == 1 && b->ne[0] == 4 && b->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(b) && op->ne[1] == 1 && op->ne[2] == 1 && op->ne[0] % 16 == 0 &&
                   op->ne[0] <= 16384 && a->op == GGML_OP_CONCAT && xrt_qc_rs_rows(a->src[0]);
        case GGML_OP_GATED_DELTA_NET: {
            const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * g = op->src[3], * be = op->src[4];
            const int64_t hv = v->ne[1], hk = q->ne[1];
            const auto rows_ok = [](const ggml_tensor * t) {
                return t->type == GGML_TYPE_F32 && t->ne[0] == XRT_QC_GDN_HEAD && t->ne[2] == 1 && t->ne[3] == 1 &&
                       t->nb[0] == sizeof(float) && t->nb[1] == XRT_QC_GDN_HEAD * sizeof(float);
            };
            const int64_t max_hv = (xrt_active_qc_features() & XRT_QC_FEATURE_STATE_LOAD) ? 32 : 64;
            return ggml_get_op_params_i32(op, 0) == 1 && rows_ok(q) && rows_ok(k) && rows_ok(v) && k->ne[1] == hk &&
                   hk > 0 && hv <= max_hv && hv % hk == 0 && g->ne[0] == 1 && be->ne[0] == 1 &&
                   ggml_nelements(g) == hv && ggml_nelements(be) == hv && ggml_is_contiguous(g) &&
                   ggml_is_contiguous(be) && xrt_qc_rs_rows(op->src[5]);
        }
        default:
            return xrt_qc_supports(op);
    }
}

// One command list under construction. Tensors are identified by their data pointer.
struct xrt_qc_list {
    struct output { ggml_tensor * node; unsigned off; size_t floats; };
    // Offsets into the activation buffer: host memory may be reused by a later tensor of the same split.
    struct check { unsigned op; ggml_tensor * node; unsigned a_off, b_off, y_off, a_len; float eps; };
    struct entry { const uint8_t * data; size_t bytes; unsigned off; };
    std::vector<unsigned> cmds;
    std::vector<entry> entries;  // inputs and outputs in the order they were added
    std::deque<std::vector<float>> owned;  // host-made inputs (rope caches): their addresses stay unique
    std::vector<output> outputs;
    std::vector<check> checks;
    std::vector<std::function<void(xrt_backend_context *)>> verifies;  // DeltaNet opcodes
    size_t in_used = 0, out_used = XRT_QC_OUT_BASE;
    size_t state_load_at = 0;  // command word where the next STATE_LOAD goes: behind the list's last GDN step
    std::unordered_map<const ggml_tensor *, size_t> late_at;  // GEMV command (word) to move behind its GDN step
    std::unordered_map<const ggml_tensor *, size_t> conv_src_at;  // GEMV command (word) that may get a CONV epilog
    std::unordered_map<const ggml_tensor *, size_t> add_src_at;   // GEMV command (word) that may get an ADD epilog
    // Deferred RMSNORM commands (word positions): only GEMVs of this list read their output. The newest one is the
    // one the kernel keeps on chip; clearing the flag (undefer) makes a command write its y again.
    std::unordered_map<const ggml_tensor *, size_t> deferred_at;
    const ggml_tensor * last_deferred = nullptr;
    void undefer(const ggml_tensor * t) {
        const auto it = deferred_at.find(t);
        if (it == deferred_at.end()) return;
        cmds[it->second + 7] &= ~4U;
        deferred_at.erase(it);
        if (last_deferred == t) last_deferred = nullptr;
    }
    // Command words from `from` on moved by `delta` words: keep the recorded positions in step.
    void shift_positions(size_t from, std::ptrdiff_t delta) {
        for (auto * m : {&late_at, &conv_src_at, &add_src_at, &deferred_at}) {
            for (auto & e : *m) if (e.second >= from) e.second = size_t(std::ptrdiff_t(e.second) + delta);
        }
    }
    // Readback filter: with last_use set, a flush copies an output back only if a later split reads it, it is a graph
    // output, or a node of this split at or after pos reads it (last_use: data owner -> last reading node).
    const std::unordered_map<const ggml_tensor *, int> * last_use = nullptr;
    int pos = 0;

    // The newest mapping wins: ggml may reuse an address for a later tensor of the same split.
    const unsigned * find(const void * data) const {
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) if (it->data == data) return &it->off;
        return nullptr;
    }
    // Act offset of [data, data + floats) inside the newest input or output it touches; overlap is set when that
    // entry only partly covers the range (the list must be flushed and the data uploaded again).
    bool resolve(const void * data, size_t floats, unsigned * off, bool * overlap) const {
        const auto * lo = static_cast<const uint8_t *>(data);
        const auto * hi = lo + floats * sizeof(float);
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            if (lo < it->data + it->bytes && it->data < hi) {
                if (lo >= it->data && hi <= it->data + it->bytes && (lo - it->data) % (16 * sizeof(float)) == 0) {
                    *off = it->off + unsigned((lo - it->data) / sizeof(float));
                    return true;
                }
                *overlap = true;
                return false;
            }
        }
        return false;
    }
    // True if [data, data + floats) touches a pending result without being exactly that result.
    bool partially_overlaps_output(const void * data, size_t floats) const {
        const auto * lo = static_cast<const uint8_t *>(data);
        const auto * hi = lo + floats * sizeof(float);
        for (const auto & o : outputs) {
            const auto * olo = static_cast<const uint8_t *>(o.node->data);
            const auto * ohi = olo + o.floats * sizeof(float);
            if (lo < ohi && olo < hi && !(lo == olo && hi == ohi)) return true;
        }
        return false;
    }
    bool empty() const { return cmds.empty(); }
};

static const std::vector<float> & xrt_qc_ones() {
    static const std::vector<float> ones(XRT_QC_MAX_COLS, 1.0f);
    return ones;
}

static size_t xrt_qc_act_floats(const xrt_backend_context * ctx) {
    return ctx->qc_gdn ? XRT_QC_RES_BASE : XRT_QC_ACT_FLOATS;  // per-list regions end where the resident one starts
}

// Floats a node would add to the input and output regions and whether it fits next to the current list.
static bool xrt_qc_fits(const xrt_qc_list & list, const std::vector<std::pair<const void *, size_t>> & inputs,
                        size_t out_floats) {
    const size_t new_cmds = (out_floats + XRT_QC_CMD_ROWS - 1) / XRT_QC_CMD_ROWS;  // GEMV row chunks (else 1)
    size_t in = 0;
    for (const auto & i : inputs) if (list.find(i.first) == nullptr) in += xrt_qc_round16(i.second);
    return list.in_used + in <= XRT_QC_OUT_BASE && list.out_used + xrt_qc_round16(out_floats) <= XRT_QC_ACT_FLOATS &&
           (list.cmds.size() / XRT_QC_CMD_WORDS + new_cmds) * XRT_QC_CMD_WORDS * sizeof(unsigned) <= XRT_QC_CMD_BYTES;
}

static unsigned xrt_qc_input(xrt_backend_context * ctx, xrt_qc_list & list, const void * data, size_t floats) {
    if (const unsigned * off = list.find(data)) return *off;
    const unsigned off = unsigned(list.in_used);
    ctx->qc_act_bo.write(data, floats * sizeof(float), size_t(off) * sizeof(float));
    list.entries.push_back({static_cast<const uint8_t *>(data), floats * sizeof(float), off});
    list.in_used += xrt_qc_round16(floats);
    return off;
}

static unsigned xrt_qc_output(xrt_qc_list & list, ggml_tensor * node, size_t floats) {
    const unsigned off = unsigned(list.out_used);
    list.entries.push_back({static_cast<const uint8_t *>(node->data), floats * sizeof(float), off});
    list.outputs.push_back({node, off, floats});
    list.out_used += xrt_qc_round16(floats);
    return off;
}

static void xrt_qc_verify(xrt_backend_context * ctx, const xrt_qc_list::check & c) {
    ggml_tensor * node = c.node;
    const unsigned n = unsigned(node->ne[0]);
    const float * act = ctx->qc_act_bo.map<const float *>();
    const float * a = act + c.a_off;
    const float * b = act + c.b_off;
    const float * y = act + c.y_off;
    std::vector<float> ref;
    std::vector<double> magnitude;
    const bool gemv = c.op == XRT_QC_OP_GEMV;
    if (gemv) {
        const ggml_tensor * w = node->src[0];
        ref = gemvq_ref::reference(static_cast<const uint8_t *>(w->data), a, n, c.a_len,
                                   xrt_qc_format(w->type, ctx->qc_features), magnitude);
    } else if (c.op == XRT_QC_OP_RMSNORM) {
        ref = gemvq_ref::rmsnorm(a, b, n, c.eps);
    } else if (c.op == XRT_QC_OP_ADD) {
        ref = gemvq_ref::add(a, b, n);
    } else {
        ref = gemvq_ref::silu_mul(a, b, n);
    }
    unsigned bad = 0;
    double worst = 0;
    for (unsigned i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(double(y[i]) - ref[i]));
        const bool ok = gemv ? gemvq_ref::within_tolerance(y[i], ref[i], magnitude[i])
                             : gemvq_ref::within_elementwise(y[i], ref[i]);
        if (!ok) ++bad;
    }
    fprintf(stderr, "XRT-VERIFY: %s gemvqc16 op=%u %s n=%u bad=%u max_abs=%g\n", bad ? "FAIL" : "PASS", c.op,
            node->name, n, bad, worst);
    if (bad) throw std::runtime_error("gemvqc16 result differs from the CPU reference");
}

// Element-wise check of a DeltaNet opcode result (magnitude empty: element-wise tolerance; else sums of |terms|).
static void xrt_qc_report(const char * what, const char * name, const float * got, const std::vector<float> & ref,
                          const std::vector<double> & magnitude) {
    unsigned bad = 0;
    double worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double err = std::fabs(double(got[i]) - ref[i]);
        worst = std::max(worst, err);
        const bool ok = magnitude.empty() ? gemvq_ref::within_elementwise(got[i], ref[i])
                                          : err <= 1e-6 + 1e-5 * std::fabs(double(ref[i])) + 4e-6 * magnitude[i];
        if (!ok) ++bad;
    }
    fprintf(stderr, "XRT-VERIFY: %s gemvqc16 %s %s n=%zu bad=%u max_abs=%g\n", bad ? "FAIL" : "PASS", what, name,
            ref.size(), bad, worst);
    if (bad) throw std::runtime_error("gemvqc16 result differs from the CPU reference");
}

#ifdef GGML_XRT_GEMVQ_EMULATE
// The HLS C++ model of the kernel (fpga/src/gemvqc16.cpp); ap_uint<512> has the 64-byte HBM layout.
extern "C" int gemvqc16(const void * w0, const void * w1, const void * w2, const void * w3, const void * w4,
                        const void * w5, const void * w6, const void * w7, const void * w8, const void * w9,
                        const void * w10, const void * w11, const void * w12, const void * w13, const void * w14,
                        const void * w15,
#if GGML_XRT_EMULATE_ENGINES == 20
                        const void * w16, const void * w17, const void * w18, const void * w19,
#endif
                        void * act, const unsigned * cmds, unsigned n_cmds, const void * st0_rd,
                        void * st0_wr, const void * st1_rd, void * st1_wr, const void * st2_rd, void * st2_wr,
                        const void * st3_rd, void * st3_wr, const void * act_rd);
#endif

static void xrt_qc_emulate_run(xrt_backend_context * ctx, unsigned n_cmds) {
#ifdef GGML_XRT_GEMVQ_EMULATE
    std::array<void *, GGML_XRT_EMULATE_ENGINES> w{};
    for (unsigned e = 0; e < w.size(); ++e) w[e] = ctx->qc_weight_bos[e].map<void *>();
    std::array<void *, 4> st{};
    for (unsigned p = 0; p < 4; ++p) st[p] = ctx->qc_state_bos[p].map<void *>();
    const int status = gemvqc16(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12],
                                w[13], w[14], w[15],
#if GGML_XRT_EMULATE_ENGINES == 20
                                w[16], w[17], w[18], w[19],
#endif
                                ctx->qc_act_bo.map<void *>(), ctx->qc_cmd_bo.map<const unsigned *>(),
                                n_cmds, st[0], st[0], st[1], st[1], st[2], st[2], st[3], st[3], ctx->qc_act_bo.map<void *>());
    if (status != 0) throw std::runtime_error("gemvqc16 model rejected command " + std::to_string(status));
#else
    (void) ctx;
    (void) n_cmds;
    throw std::runtime_error("this build has no gemvqc16 model (GGML_XRT_GEMVQ_EMULATE)");
#endif
}

// Launches the list, copies every result back to its tensor and runs the optional reference checks.
// Runs the first n_cmds commands of the command buffer (emulator, direct register start or xrt::run).
static void xrt_qc_launch(xrt_backend_context * ctx, unsigned n_cmds) {
    // GGML_XRT_EMULATE_DRY=1 (diagnostic): the emulator skips the kernel model, so host-side timing runs fast;
    // the results are garbage.
    static const bool dry = std::getenv("GGML_XRT_EMULATE_DRY") != nullptr;
    if (ctx->qc_emulate) {
        if (!dry || n_cmds == 0) xrt_qc_emulate_run(ctx, n_cmds);
    } else if (ctx->qc_direct) {
        // ap_ctrl_chain: start, poll until done or idle, read ap_return (0x10, the 1-based index of a rejected
        // command), then ap_continue.
        {
            xrt_scoped_ns launch_timer{g_xrt_timing.gemv_launch_ns};
            ctx->qc_ip.write_register(ctx->qc_ncmds_reg, n_cmds);
            ctx->qc_ip.write_register(0x00, 0x1);
        }
        uint32_t status = 0;
        {
            xrt_scoped_ns wait_timer{g_xrt_timing.gemv_wait_ns};
            const unsigned timeout_ms = xrt_wait_timeout_ms();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
            while (((status = ctx->qc_ip.read_register(0x00)) & 0x6) == 0) {
                if (std::chrono::steady_clock::now() > deadline) {
                    throw std::runtime_error("gemvqc direct start: no ap_done after " + std::to_string(timeout_ms) + " ms");
                }
            }
        }
        const uint32_t rejected = ctx->qc_ip.read_register(0x10);
        ctx->qc_ip.write_register(0x00, 0x10);
        if (rejected != 0) {
            throw std::runtime_error("gemvqc rejected command " + std::to_string(rejected) + " of " + std::to_string(n_cmds));
        }
    } else {
        {
            xrt_scoped_ns launch_timer{g_xrt_timing.gemv_launch_ns};
            ctx->qc_run.set_arg(int(ctx->qc_engines) + 2, n_cmds);
            ctx->qc_run.start();
        }
        xrt_scoped_ns wait_timer{g_xrt_timing.gemv_wait_ns};
        xrt_wait(ctx->qc_run);
    }
}

// Feature probe: a launch without commands writes GEMVQC_FEATURES into the first act float; older kernels
// leave the zero there.
static uint32_t xrt_qc_probe_features(xrt_backend_context * ctx) {
    const uint32_t zero[16] = {};
    ctx->qc_act_bo.write(zero, sizeof(zero), 0);
    ctx->qc_act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, sizeof(zero), 0);
    xrt_qc_launch(ctx, 0);
    ctx->qc_act_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, sizeof(zero), 0);
    uint32_t word = 0;
    ctx->qc_act_bo.read(&word, sizeof(word), 0);
    const uint32_t features = (word & 0xFFFF0000U) == XRT_QC_FEATURE_MARK ? (word & 0xFFFFU) : 0U;
    // GGML_XRT_FEATURE_MASK=<bits> (diagnostic): hides fusion bits of the kernel, so their plain command sequences
    // run instead. The format and layout bits (Q5_K, Q6_K, Q4_1, packed Q6_K) always stay.
    const char * mask = std::getenv("GGML_XRT_FEATURE_MASK");
    const uint32_t keep = XRT_QC_FEATURE_Q5_K | XRT_QC_FEATURE_Q6_K | XRT_QC_FEATURE_Q4_1 | XRT_QC_FEATURE_Q6K_PACKED;
    return mask != nullptr ? features & (uint32_t(std::strtoul(mask, nullptr, 0)) | keep) : features;
}

static void xrt_qc_flush(xrt_backend_context * ctx, xrt_qc_list & list) {
    if (list.empty()) return;
    xrt_scoped_ns flush_timer{g_xrt_timing.flush_ns};
    // A deferred norm whose output a later node still reads (or the host) must write it after all.
    std::vector<const ggml_tensor *> needed;
    for (const auto & d : list.deferred_at) {
        bool later = list.last_use == nullptr ||
                     (d.first->flags & (GGML_TENSOR_FLAG_SPLIT_OUTPUT | GGML_TENSOR_FLAG_OUTPUT)) != 0;
        if (!later) {
            const auto it = list.last_use->find(d.first);
            later = it != list.last_use->end() && it->second >= list.pos;
        }
        if (later) needed.push_back(d.first);
    }
    for (const auto * t : needed) list.undefer(t);
    {
        xrt_scoped_ns x_timer{g_xrt_timing.gemv_x_ns};
        ctx->qc_act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, align_xrt_word(list.in_used * sizeof(float)), 0);
        ctx->qc_cmd_bo.write(list.cmds.data(), list.cmds.size() * sizeof(unsigned), 0);
        ctx->qc_cmd_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, align_xrt_word(list.cmds.size() * sizeof(unsigned)), 0);
    }
    xrt_qc_launch(ctx, unsigned(list.cmds.size() / XRT_QC_CMD_WORDS));
    {
        xrt_scoped_ns y_timer{g_xrt_timing.gemv_y_ns};
        const auto needed = [&](const ggml_tensor * t) {
if (list.last_use == nullptr || (t->flags & (GGML_TENSOR_FLAG_SPLIT_OUTPUT | GGML_TENSOR_FLAG_OUTPUT))) {
                return true;
            }
            const auto it = list.last_use->find(t);
            return it != list.last_use->end() && it->second >= list.pos;
        };
        size_t lo = SIZE_MAX, hi = 0, copied = 0;
        for (const auto & o : list.outputs) {
            if (!needed(o.node)) continue;
            lo = std::min(lo, size_t(o.off));
            hi = std::max(hi, size_t(o.off) + o.floats);
        }
        if (hi > lo) {
            ctx->qc_act_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, align_xrt_word((hi - lo) * sizeof(float)), lo * sizeof(float));
            for (const auto & o : list.outputs) {
                if (!needed(o.node)) continue;
                const size_t bytes = o.floats * sizeof(float);
                ctx->qc_act_bo.read(o.node->data, bytes, size_t(o.off) * sizeof(float));
                copied += o.floats;
            }
        }
        if (xrt_debug_enabled()) {
            fprintf(stderr, "XRT-DEBUG: gemvqc16 list commands=%zu inputs=%zu floats outputs=%zu floats, %zu copied back\n",
                    list.cmds.size() / XRT_QC_CMD_WORDS, list.in_used, list.out_used - XRT_QC_OUT_BASE, copied);
        }
    }
    for (const auto & c : list.checks) {
        if (ctx->verify_calls_remaining <= 0) break;
        --ctx->verify_calls_remaining;
        xrt_qc_verify(ctx, c);
    }
    for (const auto & v : list.verifies) {
        if (ctx->verify_calls_remaining <= 0) break;
        --ctx->verify_calls_remaining;
        v(ctx);
    }
    const auto * last_use = list.last_use;
    const int pos = list.pos;
    list = xrt_qc_list();
    list.last_use = last_use;
    list.pos = pos;
}

// A split without any MUL_MAT is cheaper on the host than one kernel launch (~0.2 ms): RMSNORM, ADD and SwiGLU on
// single-token vectors take microseconds. Same arithmetic as the CPU reference.
static bool xrt_qc_run_on_host(struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_op op = cgraph->nodes[i]->op;
        if (op == GGML_OP_MUL_MAT) return false;
        // DeltaNet ops (states on the card) and row-wise norms need the kernel
        if (op != GGML_OP_RMS_NORM && op != GGML_OP_MUL && op != GGML_OP_ADD && op != GGML_OP_GLU &&
            op != GGML_OP_VIEW && op != GGML_OP_PERMUTE && op != GGML_OP_TRANSPOSE && op != GGML_OP_RESHAPE &&
            op != GGML_OP_NONE) {
            return false;
        }
        if (cgraph->nodes[i]->ne[1] != 1 && op != GGML_OP_VIEW && op != GGML_OP_PERMUTE &&
            op != GGML_OP_TRANSPOSE && op != GGML_OP_RESHAPE && op != GGML_OP_NONE) {
            return false;
        }
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        const unsigned n = unsigned(node->ne[0]);
        std::vector<float> y;
        switch (node->op) {
            case GGML_OP_RMS_NORM:
                y = gemvq_ref::rmsnorm(static_cast<const float *>(node->src[0]->data), xrt_qc_ones().data(), n,
                                       ggml_get_op_params_f32(node, 0));
                break;
            case GGML_OP_MUL:  // weight MUL of an RMS_NORM: plain element-wise product
                y.resize(n);
                for (unsigned k = 0; k < n; ++k) {
                    y[k] = static_cast<const float *>(node->src[0]->data)[k] * static_cast<const float *>(node->src[1]->data)[k];
                }
                break;
            case GGML_OP_ADD:
                y = gemvq_ref::add(static_cast<const float *>(node->src[0]->data),
                                   static_cast<const float *>(node->src[1]->data), n);
                break;
            case GGML_OP_GLU:
                y = gemvq_ref::silu_mul(static_cast<const float *>(node->src[0]->data),
                                        static_cast<const float *>(node->src[1]->data), n);
                break;
            default:
                continue;  // views
        }
        std::memcpy(node->data, y.data(), size_t(n) * sizeof(float));
    }
    return true;
}

static bool xrt_qc_is_view(const ggml_tensor * node) {
    return node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE ||
           node->op == GGML_OP_RESHAPE || node->op == GGML_OP_NONE;
}

// Appends GEMV commands (row chunks of at most XRT_QC_CMD_ROWS) for MUL_MAT node with x at a_off, y at y_off.
// flags go into the format word (XRT_QC_GEMV_SWIGLU, only for a single chunk).
static void xrt_qc_emit_gemv(xrt_backend_context * ctx, xrt_qc_list & list, ggml_tensor * node, unsigned a_off,
                             unsigned y_off, unsigned flags = 0) {
    const ggml_tensor * w = node->src[0];
    const unsigned format = xrt_qc_format(w->type, ctx->qc_features);
    const unsigned cols = unsigned(w->ne[0]), n = unsigned(node->ne[0]);
    const uint64_t w_off = xrt_qc_weight_offset(ctx, w);
    // Row chunks start at multiples of the engine count, so engine e's rows of a chunk follow each other in its
    // arena at (r0 / engines) rows past the tensor start.
    for (unsigned r0 = 0; r0 < n; r0 += unsigned(XRT_QC_CMD_ROWS)) {
        const unsigned rows = std::min<unsigned>(unsigned(XRT_QC_CMD_ROWS), n - r0);
        const uint64_t off = w_off + uint64_t(r0 / ctx->qc_engines) *
                                         gemvqw_layout::words_per_row(cols, format, (ctx->qc_features & XRT_QC_FEATURE_Q6K_PACKED) != 0);
        const unsigned cmd[XRT_QC_CMD_WORDS] = {XRT_QC_OP_GEMV, format | flags, rows, cols, unsigned(off),
                                                unsigned(off >> 32), a_off, y_off + r0};
        list.cmds.insert(list.cmds.end(), cmd, cmd + XRT_QC_CMD_WORDS);
    }
    ++g_xrt_timing.gemv_count;
    g_xrt_timing.gemv_bytes += ggml_nbytes(w);
}

// Translates a graph split into gemvqc16 command lists. RMS_NORM followed by its weight MUL becomes one
// RMSNORM command; inputs from host memory are uploaded once per list, every result is copied back.
static enum ggml_status xrt_qc_graph_compute(xrt_backend_context * ctx, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!xrt_qc_is_view(node) && !xrt_qc_supports(node)) {
            fprintf(stderr, "XRT: gemvqc16 cannot run node op=%d (%s)\n", node->op, node->name);
            return GGML_STATUS_FAILED;
        }
    }
    if (xrt_qc_run_on_host(cgraph)) {
        return GGML_STATUS_SUCCESS;
    }
    xrt_qc_list list;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (xrt_qc_is_view(node)) {
            continue;
        }
        // A weight MUL fused into the preceding RMS_NORM command needs no command of its own.
        if (node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes) {
            ggml_tensor * next = cgraph->nodes[i + 1];
            if (next->op == GGML_OP_MUL && next->src[0] == node && xrt_qc_supports(next)) {
                continue;  // emitted with the MUL
            }
        }
        const unsigned n = unsigned(node->ne[0]);
        std::vector<std::pair<const void *, size_t>> inputs;
        unsigned op = 0;
        const void * a = nullptr;
        const void * b = nullptr;
        size_t a_len = n, b_len = n;
        float eps = 0.0f;
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                op = XRT_QC_OP_GEMV; a = node->src[1]->data; a_len = size_t(node->src[0]->ne[0]);
                break;
            case GGML_OP_RMS_NORM:
                op = XRT_QC_OP_RMSNORM; a = node->src[0]->data; b = xrt_qc_ones().data();
                eps = ggml_get_op_params_f32(node, 0);
                break;
            case GGML_OP_MUL:  // weight MUL of an RMS_NORM (xrt_qc_supports guarantees the pattern)
                op = XRT_QC_OP_RMSNORM; a = node->src[0]->src[0]->data; b = node->src[1]->data;
                eps = ggml_get_op_params_f32(node->src[0], 0);
                break;
            case GGML_OP_ADD:
                op = XRT_QC_OP_ADD; a = node->src[0]->data; b = node->src[1]->data;
                break;
            case GGML_OP_GLU:
                op = XRT_QC_OP_SILU_MUL; a = node->src[0]->data; b = node->src[1]->data;
                break;
            default:
                return GGML_STATUS_FAILED;
        }
        inputs.emplace_back(a, a_len);
        if (b != nullptr) inputs.emplace_back(b, b_len);
        bool flush = !xrt_qc_fits(list, inputs, n);
        for (const auto & in : inputs) flush = flush || list.partially_overlaps_output(in.first, in.second);
        if (flush) {
            xrt_qc_flush(ctx, list);
        }
        const unsigned a_off = xrt_qc_input(ctx, list, a, a_len);
        const unsigned b_off = b != nullptr ? xrt_qc_input(ctx, list, b, b_len) : 0U;
        const unsigned y_off = xrt_qc_output(list, node, n);
        if (op == XRT_QC_OP_GEMV) {
            xrt_qc_emit_gemv(ctx, list, node, a_off, y_off);
        } else {
            unsigned eps_bits = 0;
            std::memcpy(&eps_bits, &eps, sizeof eps_bits);
            const unsigned cmd[XRT_QC_CMD_WORDS] = {op, n, a_off, b_off, y_off, eps_bits, 0, 0};
            list.cmds.insert(list.cmds.end(), cmd, cmd + XRT_QC_CMD_WORDS);
        }
        list.checks.push_back({op, node, a_off, b_off, y_off, unsigned(a_len), eps});
    }
    xrt_qc_flush(ctx, list);
    return GGML_STATUS_SUCCESS;
}

// ---- gemvqc16 with DeltaNet opcodes -----------------------------------------------------------------------------

static bool xrt_qc_is_weight(const ggml_tensor * t) {
    return t != nullptr && t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
           t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

// F32 weight (norm weights, conv kernel, DeltaNet gate constants), uploaded once into the resident act region.
static unsigned xrt_qc_const(xrt_backend_context * ctx, const ggml_tensor * t) {
    if (const auto it = ctx->qc_consts.find(t->data); it != ctx->qc_consts.end()) return it->second;
    const size_t floats = size_t(ggml_nelements(t)), padded = xrt_qc_round16(floats);
    if (XRT_QC_RES_BASE + ctx->qc_resident_used + padded > XRT_QC_ACT_BYTES_GDN / sizeof(float)) {
        throw std::runtime_error(std::string("gemvqc16 resident act region full at ") + t->name);
    }
    const unsigned off = unsigned(XRT_QC_RES_BASE + ctx->qc_resident_used);
    ctx->qc_act_bo.write(t->data, floats * sizeof(float), size_t(off) * sizeof(float));
    ctx->qc_act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, floats * sizeof(float), size_t(off) * sizeof(float));
    ctx->qc_resident_used += padded;
    ctx->qc_consts.emplace(t->data, off);
    return off;
}

struct xrt_qc_operand { const ggml_tensor * t; const void * data; size_t floats; };

// Act offsets of the operands of one command: a pending input or output that contains them, a resident weight,
// else a fresh upload. Flushes the list first when the command does not fit or an operand only partly covers a
// pending result.
static std::vector<unsigned> xrt_qc_operands(xrt_backend_context * ctx, xrt_qc_list & list,
                                             const std::vector<xrt_qc_operand> & ops, size_t out_floats,
                                             size_t new_cmds) {
    bool flush = false;
    size_t in = 0;
    {
        xrt_scoped_ns operands_timer{g_xrt_timing.operands_ns};
        for (const auto & o : ops) {
            unsigned off = 0;
            bool overlap = false;
            if (list.resolve(o.data, o.floats, &off, &overlap)) continue;
            flush = flush || overlap;
            if (!xrt_qc_is_weight(o.t)) in += xrt_qc_round16(o.floats);
        }
    }
    flush = flush || list.in_used + in > XRT_QC_OUT_BASE ||
            list.out_used + xrt_qc_round16(out_floats) > xrt_qc_act_floats(ctx) ||
            (list.cmds.size() / XRT_QC_CMD_WORDS + new_cmds) * XRT_QC_CMD_WORDS * sizeof(unsigned) > XRT_QC_CMD_BYTES;
    if (flush) xrt_qc_flush(ctx, list);
    xrt_scoped_ns operands_timer{g_xrt_timing.operands_ns};
    std::vector<unsigned> offs;
    for (const auto & o : ops) {
        unsigned off = 0;
        bool overlap = false;
        if (list.resolve(o.data, o.floats, &off, &overlap)) offs.push_back(off);
        else if (xrt_qc_is_weight(o.t)) offs.push_back(xrt_qc_const(ctx, o.t));
        else offs.push_back(xrt_qc_input(ctx, list, o.data, o.floats));
    }
    return offs;
}

static void xrt_qc_push(xrt_qc_list & list, std::initializer_list<unsigned> words) {
    std::vector<unsigned> cmd(words);
    cmd.resize(XRT_QC_CMD_WORDS, 0);
    list.cmds.insert(list.cmds.end(), cmd.begin(), cmd.end());
}

static unsigned xrt_qc_float_bits(float v) {
    unsigned bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return bits;
}

static const ggml_tensor * xrt_qc_view_root(const ggml_tensor * t) {
    while (t != nullptr && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
                            t->op == GGML_OP_TRANSPOSE)) {
        t = t->src[0];
    }
    return t;
}

// RS tensor and byte offset of a host pointer into an XRT_RS buffer (t: the tensor the pointer belongs to).
static const xrt_rs_tensor & xrt_rs_at(const ggml_tensor * t, const void * data, size_t * within) {
    const ggml_backend_buffer_t buffer = xrt_rs_buffer_of(t);
    const auto * rs = buffer != nullptr ? xrt_rs_ctx(buffer) : nullptr;
    const auto * e = rs != nullptr ? xrt_rs_find(rs, data) : nullptr;
    if (e == nullptr) throw std::runtime_error(std::string("not an XRT_RS tensor: ") + t->name);
    *within = size_t(static_cast<const uint8_t *>(data) - rs->base) - e->off;
    return *e;
}

// Pointer to row ids[0] of GET_ROWS node gr (the state slot a decode step reads).
static const uint8_t * xrt_qc_state_row(const ggml_tensor * gr) {
    const int32_t id = static_cast<const int32_t *>(gr->src[1]->data)[0];
    return static_cast<const uint8_t *>(gr->src[0]->data) + size_t(id) * gr->src[0]->nb[1];
}

// Host-side state ops (prompts, sequence starts): GET_ROWS from, CPY into and SCALE of XRT_RS tensors.
static void xrt_qc_state_op(xrt_backend_context * ctx, ggml_tensor * node) {
    if (node->op == GGML_OP_GET_ROWS) {
        const ggml_tensor * src = node->src[0];
        const int32_t * ids = static_cast<const int32_t *>(node->src[1]->data);
        for (int64_t r = 0; r < node->src[1]->ne[0]; ++r) {
            const uint8_t * from = static_cast<const uint8_t *>(src->data) + size_t(ids[r]) * src->nb[1];
            size_t within = 0;
            const auto & e = xrt_rs_at(src, from, &within);
            xrt_rs_io(ctx, e, within, static_cast<uint8_t *>(node->data) + size_t(r) * node->nb[1],
                      size_t(src->ne[0]) * sizeof(float), false);
        }
    } else if (node->op == GGML_OP_CPY) {
        const ggml_tensor * src = node->src[0];
        const ggml_tensor * dst = node->src[1];
        std::vector<float> v;
        v.reserve(size_t(ggml_nelements(src)));
        for (int64_t i3 = 0; i3 < src->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < src->ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < src->ne[1]; ++i1)
                    for (int64_t i0 = 0; i0 < src->ne[0]; ++i0) {
                        v.push_back(*reinterpret_cast<const float *>(static_cast<const uint8_t *>(src->data) +
                                    i0 * src->nb[0] + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3]));
                    }
        if (dst->nb[0] != sizeof(float)) throw std::runtime_error("XRT_RS copy needs contiguous rows");
        size_t k = 0;
        for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < dst->ne[1]; ++i1) {
                    const uint8_t * row = static_cast<const uint8_t *>(dst->data) + i1 * dst->nb[1] + i2 * dst->nb[2] +
                                          i3 * dst->nb[3];
                    size_t within = 0;
                    const auto & e = xrt_rs_at(dst, row, &within);
                    xrt_rs_io(ctx, e, within, reinterpret_cast<uint8_t *>(v.data() + k), size_t(dst->ne[0]) * sizeof(float), true);
                    k += size_t(dst->ne[0]);
                }
    } else {  // SCALE in place: y = x * s + b
        const float s = ggml_get_op_params_f32(node, 0), b = ggml_get_op_params_f32(node, 1);
        if (!ggml_is_contiguous(node)) throw std::runtime_error("XRT_RS scale needs a contiguous view");
        std::vector<float> v(size_t(ggml_nelements(node)));
        size_t within = 0;
        const auto & e = xrt_rs_at(node, node->data, &within);
        xrt_rs_io(ctx, e, within, reinterpret_cast<uint8_t *>(v.data()), v.size() * sizeof(float), false);
        for (auto & x : v) x = x * s + b;
        xrt_rs_io(ctx, e, within, reinterpret_cast<uint8_t *>(v.data()), v.size() * sizeof(float), true);
    }
}

// Reads hv * 128 state rows starting at global row `row` from the card (verification only).
static std::vector<float> xrt_qc_read_state(xrt_backend_context * ctx, size_t row, size_t rows) {
    xrt_rs_tensor t;
    t.striped = true;
    t.card = row;
    t.bytes = rows * XRT_QC_STATE_ROW;
    std::vector<float> v(rows * XRT_QC_STATE_ROW / sizeof(float));
    xrt_rs_io(ctx, t, 0, reinterpret_cast<uint8_t *>(v.data()), t.bytes, false);
    return v;
}

static std::vector<float> xrt_qc_read_act(xrt_backend_context * ctx, size_t off, size_t floats) {
    std::vector<float> v(floats);
    ctx->qc_act_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, floats * sizeof(float), off * sizeof(float));
    ctx->qc_act_bo.read(v.data(), floats * sizeof(float), off * sizeof(float));
    return v;
}

// Graph split -> command lists for a kernel with DeltaNet opcodes. Besides one command per op, two patterns span
// several nodes: the conv window (GET_ROWS of r_l, CONCAT with the new input, CPY of the shifted window back into
// r_l, SSM_CONV, SILU) becomes one CONV command, and the state traffic of a GDN step (GET_ROWS of s_l, the
// GATED_DELTA_NET node, CPY of its new state into s_l) one GDN command; states never leave the card. A row-wise
// RMS_NORM with its weight MUL becomes one RMSNORM, SILU(z) * x one SILU_MUL. State ops outside these patterns
// (prompts, sequence starts) run on the host against the card copy. With the kernel features: gate and up GEMVs
// whose only reader is a SwiGLU become one SwiGLU pair (no SILU_MUL, gate and up never stored), and every GDN step
// gets a STATE_LOAD behind the list's previous GDN step, so its state is read while the GEMVs in between run. A
// GEMV since the previous step whose first reader comes after the step (z of DeltaNet) moves behind the step and
// runs in the step's pass.
static const ggml_tensor * xrt_qc_data_root(const ggml_tensor * t) {
    while (t->view_src != nullptr) t = t->view_src;
    return t;
}

static bool xrt_qc_unfused_output(const ggml_tensor * t) {
    return (t->flags & (GGML_TENSOR_FLAG_SPLIT_OUTPUT | GGML_TENSOR_FLAG_OUTPUT)) == 0;
}

// The analysis of a split for xrt_qc_graph_compute_gdn; false if a conv or GDN step has an unexpected shape.
// with_last_use: also fill last_use (data owner -> last reading node) for the readback filter of the lists.
static bool xrt_qc_build_plan(const xrt_backend_context * ctx, struct ggml_cgraph * cgraph, bool with_last_use,
                              xrt_qc_plan & P) {
    const int n_nodes = cgraph->n_nodes;
    using conv_plan = xrt_qc_plan::conv_plan;
    using gdn_plan = xrt_qc_plan::gdn_plan;
    using gdn_norms = xrt_qc_plan::gdn_norms;
    auto & convs = P.convs;
    auto & gdns = P.gdns;
    auto & conv_of_silu = P.conv_of_silu;
    auto & uses = P.uses;
    auto & skip = P.skip;
    const auto skipped = [&](const ggml_tensor * t) { return skip.count(t) != 0; };
    for (int i = 0; i < n_nodes; i++) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (cgraph->nodes[i]->src[s] != nullptr) ++uses[cgraph->nodes[i]->src[s]];
        }
    }
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_SSM_CONV) {
            conv_plan p;
            p.concat = node->src[0];
            p.get_rows = p.concat->op == GGML_OP_CONCAT ? const_cast<ggml_tensor *>(xrt_qc_view_root(p.concat->src[0])) : nullptr;
            for (int j = 0; j < n_nodes; j++) {
                ggml_tensor * m = cgraph->nodes[j];
                if (m->op == GGML_OP_CPY && m->src[0]->view_src == p.concat && xrt_rs_buffer_of(m->src[1]) != nullptr) p.cpy = m;
                if (m->op == GGML_OP_UNARY && ggml_get_unary_op(m) == GGML_UNARY_OP_SILU && m->src[0] == node) p.silu = m;
            }
            if (p.get_rows == nullptr || p.get_rows->op != GGML_OP_GET_ROWS || xrt_rs_buffer_of(p.get_rows->src[0]) == nullptr ||
                p.cpy == nullptr || p.cpy->src[0]->view_offs != sizeof(float) || ggml_nelements(p.get_rows->src[1]) != 1) {
                fprintf(stderr, "XRT: unexpected conv graph around %s\n", node->name);
                return false;
            }
            convs[node] = p;
            skip.insert({p.get_rows, p.concat, p.cpy});
            if (p.silu != nullptr) {
                skip.insert(node);
                conv_of_silu[p.silu] = node;
            }
        } else if (node->op == GGML_OP_GATED_DELTA_NET) {
            gdn_plan p;
            p.get_rows = const_cast<ggml_tensor *>(xrt_qc_view_root(node->src[5]));
            const size_t y_bytes = size_t(node->src[2]->ne[1]) * XRT_QC_GDN_HEAD * sizeof(float);
            for (int j = 0; j < n_nodes; j++) {
                ggml_tensor * m = cgraph->nodes[j];
                if (m->op == GGML_OP_CPY && m->src[0]->view_src == node && m->src[0]->view_offs == y_bytes &&
                    xrt_rs_buffer_of(m->src[1]) != nullptr) {
                    p.cpy = m;
                }
            }
            if (p.get_rows == nullptr || p.get_rows->op != GGML_OP_GET_ROWS || xrt_rs_buffer_of(p.get_rows->src[0]) == nullptr ||
                p.cpy == nullptr || ggml_nelements(p.get_rows->src[1]) != 1) {
                fprintf(stderr, "XRT: unexpected GDN graph around %s\n", node->name);
                return false;
            }
            gdns[node] = p;
            skip.insert({p.get_rows, p.cpy});
        }
    }
    // SwiGLU pairs: GLU(gate, up) of two GEMVs of this split on the same x that nothing else reads.
    auto & swiglus = P.swiglus;
    if (ctx->qc_features & XRT_QC_FEATURE_SWIGLU) {
        const std::unordered_set<const ggml_tensor *> nodes(cgraph->nodes, cgraph->nodes + n_nodes);
        const auto fusable = [&](const ggml_tensor * m) {
            return m != nullptr && m->op == GGML_OP_MUL_MAT && nodes.count(m) != 0 && uses[m] == 1 &&
                   (m->flags & (GGML_TENSOR_FLAG_SPLIT_OUTPUT | GGML_TENSOR_FLAG_OUTPUT)) == 0 &&
                   m->ne[0] <= XRT_QC_MAX_COLS;
        };
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node->op != GGML_OP_GLU || ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU ||
                ggml_get_op_params_i32(node, 1) != 0) {
                continue;
            }
            ggml_tensor * g = node->src[0], * u = node->src[1];
            // The kernel pairs only GEMVs that quantize x the same way (Q8_0, Q8_1 or Q8_K).
            const auto act_type = [&](const ggml_tensor * m) {
                const unsigned f = xrt_qc_format(m->src[0]->type, ctx->qc_features);
                return f == 13U || f == 14U ? 2 : (f == 3U ? 1 : 0);
            };
            if (fusable(g) && fusable(u) && g->src[1]->data == u->src[1]->data && g->src[0]->ne[0] == u->src[0]->ne[0] &&
                g->ne[0] == u->ne[0] && g->ne[0] == node->ne[0] && act_type(g) == act_type(u)) {
                swiglus[node] = {g, u};
                skip.insert({g, u});
            }
        }
    }
    // CONT of strided rows read only by a supported UNARY: the unary reads the rows directly.
    auto & cont_folded = P.cont_folded;
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * m = cgraph->nodes[i];
        unsigned kind = 0;
        if (m->op != GGML_OP_UNARY || !xrt_qc_unary_kind(m, &kind) || m->src[0]->op != GGML_OP_CONT) continue;
        ggml_tensor * ct = m->src[0];
        if (uses[ct] == 1 && xrt_qc_strided_rows(ct->src[0]) && (ct->flags & (GGML_TENSOR_FLAG_SPLIT_OUTPUT | GGML_TENSOR_FLAG_OUTPUT)) == 0) {
            cont_folded.insert(m);
            skip.insert(ct);
        }
    }
    auto & readers = P.readers;
    for (int i = 0; i < n_nodes; i++) {
        if (xrt_qc_is_view(cgraph->nodes[i])) continue;
        std::unordered_set<const ggml_tensor *> roots;
        for (const ggml_tensor * src : cgraph->nodes[i]->src) {
            if (src == nullptr) continue;
            while (src->view_src != nullptr) src = src->view_src;
            roots.insert(src);
        }
        for (const ggml_tensor * r : roots) ++readers[r];
    }
    const auto root_of = xrt_qc_data_root;
    const auto unfused_output = xrt_qc_unfused_output;
    // GDN norms: L2_NORM of q and k read only by the step, and the gated norm behind it
    // (MUL(MUL(RMS_NORM(y), w), SILU(z)), every node read once) become part of the GDN command.
    auto & norms_of = P.norms_of;
    if (ctx->qc_features & (XRT_QC_FEATURE_GDN_NORMS | XRT_QC_FEATURE_GDN_GATES)) {
        const bool norms_on = (ctx->qc_features & XRT_QC_FEATURE_GDN_NORMS) != 0;
        const bool gates_on = (ctx->qc_features & XRT_QC_FEATURE_GDN_GATES) != 0;
        std::unordered_map<const ggml_tensor *, ggml_tensor *> only_reader;  // tensor -> its single non-view reader
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * m = cgraph->nodes[i];
            if (xrt_qc_is_view(m)) continue;
            for (const ggml_tensor * src : m->src) if (src != nullptr) only_reader[root_of(src)] = m;
        }
        const auto single = [&](const ggml_tensor * t) -> ggml_tensor * {
            return readers[t] == 1 && unfused_output(t) ? only_reader[t] : nullptr;
        };
        for (const auto & gp : gdns) {
            const ggml_tensor * node = gp.first;
            gdn_norms nm;
            const ggml_tensor * q = node->src[0], * k = node->src[1];
            if (norms_on && q->op == GGML_OP_L2_NORM && k->op == GGML_OP_L2_NORM && single(q) == node && single(k) == node &&
                ggml_is_contiguous(q->src[0]) && ggml_is_contiguous(k->src[0]) &&
                ggml_get_op_params_f32(q, 0) == ggml_get_op_params_f32(k, 0)) {
                nm.l2 = true;
                nm.l2_eps = ggml_get_op_params_f32(q, 0);
            }
            // y (the first hv * 128 floats of the node) -> RMS_NORM rows of 128 -> MUL weight -> MUL SILU(z); the node
            // is read by that RMS_NORM and the CPY of the new state only.
            ggml_tensor * r = nullptr;
            for (int i = 0; i < n_nodes; i++) {
                ggml_tensor * m = cgraph->nodes[i];
                if (m->op == GGML_OP_RMS_NORM && root_of(m->src[0]) == node && m->src[0]->view_offs == 0 &&
                    m->ne[0] == XRT_QC_GDN_HEAD && ggml_nelements(m) == ggml_nelements(node->src[2])) {
                    r = m;
                }
            }
            ggml_tensor * m1 = r != nullptr ? single(r) : nullptr;
            ggml_tensor * m2 = m1 != nullptr && m1->op == GGML_OP_MUL && m1->src[0] == r ? single(m1) : nullptr;
            const ggml_tensor * w = m1 != nullptr ? m1->src[1] : nullptr;
            const ggml_tensor * sl = m2 != nullptr && m2->op == GGML_OP_MUL && m2->src[0] == m1 ? m2->src[1] : nullptr;
            if (norms_on && r != nullptr && readers[node] == 2 && w != nullptr && xrt_qc_is_weight(w) && ggml_nelements(w) == r->ne[0] &&
                sl != nullptr && sl->op == GGML_OP_UNARY && ggml_get_unary_op(sl) == GGML_UNARY_OP_SILU &&
                single(sl) == m2 && ggml_nelements(sl) == ggml_nelements(m2) && ggml_nelements(m2) == ggml_nelements(r) &&
                ggml_is_contiguous(sl->src[0]) && ggml_is_contiguous(m2)) {
                nm.out = m2;
                nm.w = w;
                nm.z = sl->src[0];
                nm.rms_eps = ggml_get_op_params_f32(r, 0);
                skip.insert({r, m1, const_cast<ggml_tensor *>(sl), m2});
            }
            if (nm.l2) skip.insert({const_cast<ggml_tensor *>(q), const_cast<ggml_tensor *>(k)});
            // Gates: SIGMOID(beta_raw) and MUL(SOFTPLUS(ADD(alpha_raw, dt)), a), every node read once, dt and a weights.
            const int64_t hv = node->src[2]->ne[1];
            ggml_tensor * sg = const_cast<ggml_tensor *>(root_of(node->src[4]));
            ggml_tensor * ml = const_cast<ggml_tensor *>(root_of(node->src[3]));
            const auto is_w = [&](const ggml_tensor * x) { return x != nullptr && xrt_qc_is_weight(x) && ggml_nelements(x) == hv; };
            const auto raw = [&](const ggml_tensor * x) { return x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_nelements(x) == hv; };
            if (gates_on && sg->op == GGML_OP_UNARY && ggml_get_unary_op(sg) == GGML_UNARY_OP_SIGMOID && single(sg) == node &&
                ml->op == GGML_OP_MUL && single(ml) == node && raw(sg->src[0])) {
                const int si = ml->src[0]->op == GGML_OP_UNARY ? 0 : 1;
                ggml_tensor * sp = ml->src[si];
                const ggml_tensor * a = ml->src[1 - si];
                ggml_tensor * ad = sp->op == GGML_OP_UNARY && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS && single(sp) == ml
                                       ? sp->src[0] : nullptr;
                if (ad != nullptr && ad->op == GGML_OP_ADD && single(ad) == sp && is_w(a)) {
                    const int di = is_w(ad->src[1]) ? 1 : 0;
                    if (is_w(ad->src[di]) && raw(ad->src[1 - di])) {
                        nm.beta_raw = sg->src[0];
                        nm.alpha_raw = ad->src[1 - di];
                        nm.dt = ad->src[di];
                        nm.a = a;
                        skip.insert({sg, ml, sp, ad});
                    }
                }
            }
            if (nm.l2 || nm.out != nullptr || nm.alpha_raw != nullptr) norms_of[node] = nm;
        }
    }
    // GEMVs whose only reader is an ADD of the same length whose other operand exists before the GEMV (residual adds
    // of o_proj, ssm_out, ffn_down): the ADD becomes the GEMV's epilog.
    auto & add_src = P.add_src;
    if (ctx->qc_features & XRT_QC_FEATURE_ADD_EPILOG) {
        std::unordered_map<const ggml_tensor *, int> index;
        for (int i = 0; i < n_nodes; i++) index[cgraph->nodes[i]] = i;
        for (int i = 0; i < n_nodes; i++) {
            const ggml_tensor * add = cgraph->nodes[i];
            if (add->op != GGML_OP_ADD || skipped(add) || !xrt_qc_supports_gdn(add)) continue;
            for (int s = 0; s < 2; ++s) {
                const ggml_tensor * m = root_of(add->src[s]);
                const ggml_tensor * r = root_of(add->src[1 - s]);
                const auto im = index.find(m), ir = index.find(r);
                if (m->op == GGML_OP_MUL_MAT && im != index.end() && !skipped(m) && readers[m] == 1 && unfused_output(m) &&
                    add->src[s]->data == m->data && ggml_nelements(m) == ggml_nelements(add) && m->ne[0] <= 16384 &&
                    m->ne[0] >= 256 &&  // short adds (DeltaNet gate bias): leaving the projection group costs more
                    m->ne[0] % 16 == 0 && (ir == index.end() || ir->second < im->second)) {
                    add_src.insert(m);
                    break;
                }
            }
        }
    }
    // Norm outputs read only by GEMVs of this split may stay on chip (deferred RMSNORM).
    auto & defer_ok = P.defer_ok;
    if (ctx->qc_features & XRT_QC_FEATURE_NORM_DEFER) {
        std::unordered_map<const ggml_tensor *, bool> only_gemv;
        for (int i = 0; i < n_nodes; i++) {
            const ggml_tensor * m = cgraph->nodes[i];
            if (xrt_qc_is_view(m)) continue;
            for (const ggml_tensor * src : m->src) {
                if (src == nullptr) continue;
                const ggml_tensor * r = root_of(src);
                const bool ok = m->op == GGML_OP_MUL_MAT && m->src[1] == src;
                const auto it = only_gemv.find(r);
                only_gemv[r] = (it == only_gemv.end() || it->second) && ok;
            }
        }
        for (const auto & e : only_gemv) {
            const ggml_tensor * r = e.first;
            if (e.second && (r->op == GGML_OP_RMS_NORM || r->op == GGML_OP_L2_NORM || r->op == GGML_OP_MUL) &&
                unfused_output(r) && ggml_nelements(r) == r->ne[0] && r->ne[0] % 16 == 0 && r->ne[0] <= 16384) {
                defer_ok.insert(r);
            }
        }
    }
    // GEMVs whose only reader is the conv window (qkv of DeltaNet): their command gets the conv as epilog.
    auto & conv_src = P.conv_src;
    if (ctx->qc_features & XRT_QC_FEATURE_CONV_EPILOG) {
        const std::unordered_set<const ggml_tensor *> nodes(cgraph->nodes, cgraph->nodes + n_nodes);
        for (const auto & cv : convs) {
            const ggml_tensor * x = cv.second.concat->src[1];
            while (x->view_src != nullptr) x = x->view_src;
            if (x->op == GGML_OP_MUL_MAT && nodes.count(x) != 0 && !skipped(x) && readers[x] == 1 &&
                x->ne[0] == cv.first->ne[0] && x->ne[0] <= XRT_QC_CMD_ROWS && cv.second.silu != nullptr &&
                (x->flags & (GGML_TENSOR_FLAG_SPLIT_OUTPUT | GGML_TENSOR_FLAG_OUTPUT)) == 0) {
                conv_src.insert(x);
            }
        }
    }
    // GDN node -> the GEMV for its pass: emitted since the previous GDN node, one command, first read after the step
    // (readers through views count at the view's first non-view reader). The largest such GEMV wins.
    auto & late_gemv = P.late_gemv;
    auto & late_set = P.late_set;
    if (ctx->qc_features & XRT_QC_FEATURE_GDN_GEMV) {
        std::unordered_map<const ggml_tensor *, int> first_use;
        for (int i = 0; i < n_nodes; i++) {
            if (xrt_qc_is_view(cgraph->nodes[i])) continue;
            for (const ggml_tensor * src : cgraph->nodes[i]->src) {
                if (src == nullptr) continue;
                while (src->view_src != nullptr) src = src->view_src;
                first_use.emplace(src, i);
            }
        }
        int prev = -1;
        for (int i = 0; i < n_nodes; i++) {
            if (cgraph->nodes[i]->op != GGML_OP_GATED_DELTA_NET) continue;
            const ggml_tensor * best = nullptr;
            for (int j = prev + 1; j < i; j++) {
                const ggml_tensor * m = cgraph->nodes[j];
                const auto it = first_use.find(m);
                if (m->op == GGML_OP_MUL_MAT && !skipped(m) && m->ne[0] <= XRT_QC_CMD_ROWS &&
                    (it == first_use.end() || it->second > i) && (best == nullptr || m->ne[0] > best->ne[0])) {
                    best = m;
                }
            }
            if (best != nullptr) {
                late_gemv[cgraph->nodes[i]] = best;
                late_set.insert(best);
            }
            prev = i;
        }
    }
    if (with_last_use) {
        for (int i = 0; i < n_nodes; i++) {
            for (const ggml_tensor * src : cgraph->nodes[i]->src) {
                if (src == nullptr) continue;
                while (src->view_src != nullptr) src = src->view_src;
                P.last_use[src] = i;
            }
        }
    }
    return true;
}

// The fields of t the analysis may read (not the data itself).
static void xrt_qc_key_tensor(std::vector<uint64_t> & key, const ggml_tensor * t) {
    key.push_back(uint64_t(reinterpret_cast<uintptr_t>(t)));
    if (t == nullptr) return;
    key.push_back(uint64_t(reinterpret_cast<uintptr_t>(t->data)));
    key.push_back(uint64_t(reinterpret_cast<uintptr_t>(t->buffer)));
    key.push_back(uint64_t(reinterpret_cast<uintptr_t>(t->view_src)));
    key.push_back(uint64_t(t->view_offs));
    key.push_back(uint64_t(t->op) | (uint64_t(t->type) << 16) | (uint64_t(uint32_t(t->flags)) << 32));
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        key.push_back(uint64_t(t->ne[d]));
        key.push_back(uint64_t(t->nb[d]));
    }
    for (int k = 0; k < GGML_MAX_OP_PARAMS / int(sizeof(int32_t)); ++k) key.push_back(uint64_t(uint32_t(t->op_params[k])));
}

// Exact identity of a split for the plan cache: every node with its fields and source pointers, the fields of
// sources (and view roots) that are not nodes of the split, the kernel features and whether last_use is wanted.
static void xrt_qc_plan_key(const xrt_backend_context * ctx, const struct ggml_cgraph * cgraph, bool with_last_use,
                            std::vector<uint64_t> & key) {
    key.clear();
    key.push_back(uint64_t(cgraph->n_nodes));
    key.push_back(uint64_t(ctx->qc_features) | (uint64_t(with_last_use) << 32));
    std::unordered_set<const ggml_tensor *> nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    const auto add_outside = [&](const ggml_tensor * t) {
        if (t != nullptr && nodes.count(t) == 0) xrt_qc_key_tensor(key, t);
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        xrt_qc_key_tensor(key, node);
        int n_src = GGML_MAX_SRC;
        while (n_src > 0 && node->src[n_src - 1] == nullptr) --n_src;
        key.push_back(uint64_t(n_src));
        for (int k = 0; k < n_src; ++k) {
            const ggml_tensor * src = node->src[k];
            key.push_back(uint64_t(reinterpret_cast<uintptr_t>(src)));
            add_outside(src);
            if (src != nullptr) add_outside(src->view_src);
        }
        add_outside(node->view_src);
    }
}

// The plan of this split, from the cache when the structure is unchanged (GGML_XRT_PLAN_CACHE=0 turns the cache
// off; GGML_XRT_PLAN_CHECK=1 rebuilds every cached plan and fails if it differs). nullptr if the analysis fails.
static const xrt_qc_plan * xrt_qc_get_plan(xrt_backend_context * ctx, struct ggml_cgraph * cgraph, bool with_last_use) {
    static const bool cache_off = [] {
        const char * v = std::getenv("GGML_XRT_PLAN_CACHE");
        return v != nullptr && std::strcmp(v, "0") == 0;
    }();
    static const bool check = std::getenv("GGML_XRT_PLAN_CHECK") != nullptr;
    static thread_local std::vector<uint64_t> key;
    xrt_qc_plan_key(ctx, cgraph, with_last_use, key);
    if (!cache_off) {
        for (const auto & e : ctx->qc_plans) {
            if (e.key.size() != key.size() || std::memcmp(e.key.data(), key.data(), key.size() * sizeof(uint64_t)) != 0) {
                continue;
            }
            if (check) {
                xrt_qc_plan fresh;
                if (!xrt_qc_build_plan(ctx, cgraph, with_last_use, fresh) || !(fresh == *e.plan)) {
                    throw std::runtime_error("cached gemvqc16 plan differs from a fresh analysis");
                }
            }
            return e.plan.get();
        }
    }
    auto plan = std::make_unique<xrt_qc_plan>();
    if (!xrt_qc_build_plan(ctx, cgraph, with_last_use, *plan)) return nullptr;
    if (ctx->qc_plans.size() >= 64) ctx->qc_plans.clear();  // a decode step has ~14 splits
    ctx->qc_plans.push_back({key, std::move(plan)});
    return ctx->qc_plans.back().plan.get();
}

static enum ggml_status xrt_qc_graph_compute_gdn(xrt_backend_context * ctx, struct ggml_cgraph * cgraph) {
    const int n_nodes = cgraph->n_nodes;
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!xrt_qc_is_view(node) && !xrt_qc_supports_gdn(node)) {
            fprintf(stderr, "XRT: gemvqc16 cannot run node op=%d (%s)\n", node->op, node->name);
            return GGML_STATUS_FAILED;
        }
    }
    {
        xrt_scoped_ns host_timer{g_xrt_timing.host_ns};
        if (xrt_qc_run_on_host(cgraph)) return GGML_STATUS_SUCCESS;
    }
    std::optional<xrt_scoped_ns> plan_timer;
    plan_timer.emplace(g_xrt_timing.plan_ns);
    const bool verify = ctx->verify_calls_remaining > 0;
    // Results that nothing reads after their list stay on the card. The reference checks read host inputs, so
    // verification (and GGML_XRT_READBACK_ALL=1) copies everything back.
    const char * readback_all = std::getenv("GGML_XRT_READBACK_ALL");
    const bool with_last_use = !verify && (readback_all == nullptr || std::strcmp(readback_all, "0") == 0);
    const xrt_qc_plan * plan = xrt_qc_get_plan(ctx, cgraph, with_last_use);
    if (plan == nullptr) return GGML_STATUS_FAILED;
    using conv_plan = xrt_qc_plan::conv_plan;
    using gdn_plan = xrt_qc_plan::gdn_plan;
    using gdn_norms = xrt_qc_plan::gdn_norms;
    const auto & convs = plan->convs;
    const auto & gdns = plan->gdns;
    const auto & conv_of_silu = plan->conv_of_silu;
    const auto & swiglus = plan->swiglus;
    const auto & cont_folded = plan->cont_folded;
    const auto & norms_of = plan->norms_of;
    const auto & add_src = plan->add_src;
    const auto & defer_ok = plan->defer_ok;
    const auto & conv_src = plan->conv_src;
    const auto & late_gemv = plan->late_gemv;
    const auto & late_set = plan->late_set;
    const auto skipped = [&](const ggml_tensor * t) { return plan->skip.count(t) != 0; };
    const auto use_count = [&](const ggml_tensor * t) {
        const auto it = plan->uses.find(t);
        return it == plan->uses.end() ? 0 : it->second;
    };
    const auto root_of = xrt_qc_data_root;
    const bool state_load = (ctx->qc_features & XRT_QC_FEATURE_STATE_LOAD) != 0;
    std::vector<const ggml_tensor *> fused_silu;  // SILU nodes folded into the next MUL
    xrt_qc_list list;
    if (with_last_use) list.last_use = &plan->last_use;
    plan_timer.reset();
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        list.pos = i;
        if (xrt_qc_is_view(node) || skipped(node) || ggml_nelements(node) == 0) {
            continue;
        }
        const size_t n = size_t(ggml_nelements(node));
        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];
        ggml_tensor * next = i + 1 < n_nodes ? cgraph->nodes[i + 1] : nullptr;
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                const size_t cols = size_t(a->ne[0]);
                // x from an older deferred norm: the kernel only keeps the newest one on chip.
                if (root_of(b) != list.last_deferred) list.undefer(root_of(b));
                const auto offs = xrt_qc_operands(ctx, list, {{b, b->data, cols}}, n,
                                                  (n + XRT_QC_CMD_ROWS - 1) / XRT_QC_CMD_ROWS);
                const unsigned y_off = xrt_qc_output(list, node, n);
                if (late_set.count(node) != 0) list.late_at[node] = list.cmds.size();
                if (conv_src.count(node) != 0) list.conv_src_at[node] = list.cmds.size();
                if (add_src.count(node) != 0) list.add_src_at[node] = list.cmds.size();
                xrt_qc_emit_gemv(ctx, list, node, offs[0], y_off);
                list.checks.push_back({XRT_QC_OP_GEMV, node, offs[0], 0, y_off, unsigned(cols), 0.0f});
                break;
            }
            case GGML_OP_RMS_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_MUL: {
                const ggml_tensor * norm = node->op == GGML_OP_MUL ? a : node;
                const bool weight_mul = node->op == GGML_OP_MUL && a->op == GGML_OP_RMS_NORM && i > 0 &&
                                        cgraph->nodes[i - 1] == a && use_count(a) == 1;
                if (node->op != GGML_OP_MUL && next != nullptr && next->op == GGML_OP_MUL && next->src[0] == node &&
                    node->op == GGML_OP_RMS_NORM && use_count(node) == 1) {
                    break;  // emitted with the weight MUL
                }
                if (node->op != GGML_OP_MUL || weight_mul) {
                    const unsigned row = unsigned(norm->ne[0]), rows = unsigned(n / row);
                    const unsigned flags = (norm->op == GGML_OP_L2_NORM ? XRT_QC_NORM_L2 : 0U) |
                                           (weight_mul ? 0U : XRT_QC_NORM_NO_WEIGHT);
                    const float eps = ggml_get_op_params_f32(norm, 0);
                    const ggml_tensor * nx = norm->src[0];
                    const bool strided = !xrt_qc_token_tensor(nx);  // xrt_qc_supports_gdn: then strided rows
                    const unsigned stride_words = strided ? unsigned(nx->nb[1] / 64) : 0U;
                    std::vector<xrt_qc_operand> ops = {{nx, nx->data, strided ? xrt_qc_rows_span(nx) : n}};
                    if (weight_mul) ops.push_back({b, b->data, row});
                    const bool defer = rows == 1 && !strided && defer_ok.count(node) != 0;
                    const auto offs = xrt_qc_operands(ctx, list, ops, n, defer && verify ? 2 : 1);
                    const unsigned y_off = xrt_qc_output(list, node, n);
                    const unsigned w_off = weight_mul ? offs[1] : 0U;
                    if (defer) {
                        // The row stays on chip for the GEMVs on it. With reference checks a plain RMSNORM behind it
                        // also writes y (the same values), so every check finds its input in act.
                        if (!verify) list.deferred_at[node] = list.cmds.size();  // with checks y is written anyway
                        list.last_deferred = node;
                        xrt_qc_push(list, {XRT_QC_OP_RMSNORM, row, offs[0], w_off, y_off, xrt_qc_float_bits(eps), rows, flags | 4U});
                    }
                    if (!defer || verify) {
                        xrt_qc_push(list, {XRT_QC_OP_RMSNORM, row, offs[0], w_off, y_off, xrt_qc_float_bits(eps), rows,
                                           flags | (stride_words << 8)});
                    }
                    if (verify) {
                        const unsigned x_off = offs[0];
                        list.verifies.push_back([=](xrt_backend_context * c) {
                            const float * act = c->qc_act_bo.map<const float *>();
                            std::vector<float> xs;  // the rows, packed
                            for (unsigned r = 0; r < rows; ++r) {
                                const float * src = act + x_off + size_t(r) * (stride_words ? stride_words * 16U : row);
                                xs.insert(xs.end(), src, src + row);
                            }
                            const auto ref = gemvq_ref::rownorm(xs.data(), act + w_off, row, rows, eps, flags);
                            xrt_qc_report("rmsnorm", node->name, act + y_off, ref, {});
                        });
                    }
                    break;
                }
                if (ggml_nelements(b) != ggml_nelements(node)) {
                    fprintf(stderr, "XRT: broadcast MUL %s is not the weight of the preceding RMS_NORM\n", node->name);
                    return GGML_STATUS_FAILED;
                }
                // plain product, or x * SILU(z) with the SILU folded in
                const bool silu = std::find(fused_silu.begin(), fused_silu.end(), b) != fused_silu.end();
                const ggml_tensor * g = silu ? b->src[0] : b;
                const auto offs = xrt_qc_operands(ctx, list, {{g, g->data, n}, {a, a->data, n}}, n, 1);
                const unsigned y_off = xrt_qc_output(list, node, n);
                if (silu) xrt_qc_push(list, {XRT_QC_OP_SILU_MUL, unsigned(n), offs[0], offs[1], y_off});
                else xrt_qc_push(list, {XRT_QC_OP_MUL, unsigned(n), offs[1], offs[0], y_off});
                if (verify) {
                    const unsigned g_off = offs[0], x_off = offs[1];
                    list.verifies.push_back([=](xrt_backend_context * c) {
                        const float * act = c->qc_act_bo.map<const float *>();
                        const auto ref = silu ? gemvq_ref::silu_mul(act + g_off, act + x_off, unsigned(n))
                                              : gemvq_ref::mul(act + x_off, act + g_off, unsigned(n));
                        xrt_qc_report(silu ? "silu_mul" : "mul", node->name, act + y_off, ref, {});
                    });
                }
                break;
            }
            case GGML_OP_ADD:
            case GGML_OP_GLU: {
                if (const auto it = swiglus.find(node); it != swiglus.end()) {
                    ggml_tensor * g = it->second.first, * u = it->second.second;
                    const ggml_tensor * x = g->src[1];
                    if (root_of(x) != list.last_deferred) list.undefer(root_of(x));
                    const size_t cols = size_t(g->src[0]->ne[0]);
                    const auto offs = xrt_qc_operands(ctx, list, {{x, x->data, cols}}, n, 2);
                    const unsigned y_off = xrt_qc_output(list, node, n);
                    xrt_qc_emit_gemv(ctx, list, g, offs[0], y_off, XRT_QC_GEMV_SWIGLU);
                    xrt_qc_emit_gemv(ctx, list, u, offs[0], y_off);
                    if (verify) {
                        const unsigned x_off = offs[0];
                        list.verifies.push_back([=](xrt_backend_context * c) {
                            const float * act = c->qc_act_bo.map<const float *>();
                            std::vector<double> gm, um;
                            const auto gr = gemvq_ref::reference(static_cast<const uint8_t *>(g->src[0]->data), act + x_off,
                                                                 unsigned(n), unsigned(cols),
                                                                 xrt_qc_format(g->src[0]->type, c->qc_features), gm);
                            const auto ur = gemvq_ref::reference(static_cast<const uint8_t *>(u->src[0]->data), act + x_off,
                                                                 unsigned(n), unsigned(cols),
                                                                 xrt_qc_format(u->src[0]->type, c->qc_features), um);
                            unsigned bad = 0;
                            double worst = 0;
                            for (size_t r = 0; r < n; ++r) {
                                const double ref = double(gr[r]) / (1.0 + std::exp(-double(gr[r]))) * ur[r];
                                worst = std::max(worst, std::fabs(double(act[y_off + r]) - ref));
                                if (!gemvq_ref::within_swiglu(act[y_off + r], gr[r], gm[r], ur[r], um[r])) ++bad;
                            }
                            fprintf(stderr, "XRT-VERIFY: %s gemvqc16 swiglu pair %s n=%zu bad=%u max_abs=%g\n",
                                    bad ? "FAIL" : "PASS", node->name, n, bad, worst);
                            if (bad) throw std::runtime_error("gemvqc16 result differs from the CPU reference");
                        });
                    }
                    break;
                }
                const auto offs = xrt_qc_operands(ctx, list, {{a, a->data, n}, {b, b->data, n}}, n, 1);
                const unsigned y_off = xrt_qc_output(list, node, n);
                if (node->op == GGML_OP_ADD) {
                    // ADD epilog: the ADD goes right behind the GEMV of one operand (still in this list, the operand is
                    // exactly its y); that GEMV's own y is then never written.
                    for (int s = 0; s < 2; ++s) {
                        const ggml_tensor * m = root_of(node->src[s]);  // the GEMV, possibly behind a reshape
                        const auto aa = list.add_src_at.find(m);
                        if (aa == list.add_src_at.end() || list.cmds[aa->second + 7] != offs[s]) continue;
                        const size_t at = aa->second;
                        list.add_src_at.erase(aa);
                        list.cmds[at + 1] |= XRT_QC_GEMV_ADD;
                        const unsigned add_cmd[XRT_QC_CMD_WORDS] = {XRT_QC_OP_ADD, unsigned(n), offs[0], offs[1], y_off, 0, 0, 0};
                        list.cmds.insert(list.cmds.begin() + std::ptrdiff_t(at + XRT_QC_CMD_WORDS), add_cmd,
                                         add_cmd + XRT_QC_CMD_WORDS);
                        list.shift_positions(at + XRT_QC_CMD_WORDS, XRT_QC_CMD_WORDS);
                        list.checks.erase(std::remove_if(list.checks.begin(), list.checks.end(),
                                                         [&](const xrt_qc_list::check & k) { return k.node == m; }),
                                          list.checks.end());
                        if (verify) {
                            // The GEMV output is not on the card: reference GEMV + residual, GEMV tolerance.
                            const unsigned mx_off = list.cmds[at + 6], mcols = list.cmds[at + 3], r_off = offs[1 - s];
                            const ggml_tensor * mw = m->src[0];
                            list.verifies.push_back([=](xrt_backend_context * c) {
                                const float * act = c->qc_act_bo.map<const float *>();
                                std::vector<double> mag;
                                const auto ref = gemvq_ref::reference(static_cast<const uint8_t *>(mw->data), act + mx_off,
                                                                     unsigned(n), mcols, xrt_qc_format(mw->type, c->qc_features), mag);
                                unsigned bad = 0;
                                double worst = 0;
                                for (size_t i = 0; i < n; ++i) {
                                    const double want = double(ref[i]) + double(act[r_off + i]);
                                    const double err = std::fabs(double(act[y_off + i]) - want);
                                    worst = std::max(worst, err);
                                    if (err > 1e-6 + 1e-5 * std::fabs(double(ref[i])) + 1e-6 * mag[i] + 2e-5 * std::fabs(want)) ++bad;
                                }
                                fprintf(stderr, "XRT-VERIFY: %s gemvqc16 add epilog %s n=%zu bad=%u max_abs=%g\n",
                                        bad ? "FAIL" : "PASS", node->name, n, bad, worst);
                                if (bad) throw std::runtime_error("gemvqc16 result differs from the CPU reference");
                            });
                        }
                        goto add_done;
                    }
                }
                {
                const unsigned op = node->op == GGML_OP_ADD ? XRT_QC_OP_ADD : XRT_QC_OP_SILU_MUL;
                xrt_qc_push(list, {op, unsigned(n), offs[0], offs[1], y_off});
                list.checks.push_back({op, node, offs[0], offs[1], y_off, unsigned(n), 0.0f});
                }
            add_done:
                break;
            }
            case GGML_OP_UNARY: {
                if (const auto it = conv_of_silu.find(node); it != conv_of_silu.end()) {
                    // the conv of this SILU: CONV command with SiLU, states in the resident act region
                    const ggml_tensor * conv = it->second;
                    const conv_plan & p = convs.at(conv);
                    const ggml_tensor * x = p.concat->src[1];  // transposed [1, ch] view of the new input
                    const ggml_tensor * w = conv->src[1];
                    const unsigned ch = unsigned(conv->ne[0]);
                    size_t src_within = 0, dst_within = 0;
                    const auto & src_t = xrt_rs_at(p.get_rows->src[0], xrt_qc_state_row(p.get_rows), &src_within);
                    const auto & dst_t = xrt_rs_at(p.cpy->src[1], p.cpy->src[1]->data, &dst_within);
                    const unsigned s_src = unsigned(src_t.card + src_within / sizeof(float));
                    const unsigned s_dst = unsigned(dst_t.card + dst_within / sizeof(float));
                    const auto offs = xrt_qc_operands(ctx, list, {{x, x->data, ch}, {w, w->data, size_t(ch) * 4}}, ch, 1);
                    const unsigned y_off = xrt_qc_output(list, node, ch);
                    const unsigned conv_cmd[XRT_QC_CMD_WORDS] = {XRT_QC_OP_CONV, ch, offs[0], offs[1], y_off, s_src, s_dst, 1U};
                    // CONV epilog: the CONV goes right behind the GEMV that produces x (still in this list, x is exactly
                    // its y); that GEMV's own y is then never written.
                    const ggml_tensor * src = x;
                    while (src->view_src != nullptr) src = src->view_src;
                    const auto ca = list.conv_src_at.find(src);
                    if (ca != list.conv_src_at.end() && list.cmds[ca->second + 7] == offs[0]) {
                        const size_t at = ca->second;
                        list.conv_src_at.erase(ca);
                        list.cmds[at + 1] |= XRT_QC_GEMV_CONV;
                        list.cmds.insert(list.cmds.begin() + std::ptrdiff_t(at + XRT_QC_CMD_WORDS), conv_cmd,
                                         conv_cmd + XRT_QC_CMD_WORDS);
                        list.shift_positions(at + XRT_QC_CMD_WORDS, XRT_QC_CMD_WORDS);
                        list.checks.erase(std::remove_if(list.checks.begin(), list.checks.end(),
                                                         [&](const xrt_qc_list::check & k) { return k.node == src; }),
                                          list.checks.end());
                        if (verify) {
                            // x is not on the card: its reference comes from the weights, and the GEMV tolerance is
                            // carried through the last tap (|w3| * tol) into y (|silu'| <= 1.1) and the new state.
                            const auto old_state = xrt_qc_read_act(ctx, s_src, size_t(ch) * 3);
                            const unsigned qx_off = list.cmds[at + 6], qcols = list.cmds[at + 3], w_off = offs[1];
                            const ggml_tensor * qw = src->src[0];
                            list.verifies.push_back([=](xrt_backend_context * c) {
                                const float * act = c->qc_act_bo.map<const float *>();
                                std::vector<double> mag;
                                const auto q = gemvq_ref::reference(static_cast<const uint8_t *>(qw->data), act + qx_off, ch,
                                                                   qcols, xrt_qc_format(qw->type, c->qc_features), mag);
                                std::vector<float> new_state;
                                const auto ref = gemvq_ref::conv(old_state.data(), q.data(), act + w_off, ch, true, new_state);
                                const auto got = xrt_qc_read_act(c, s_dst, size_t(ch) * 3);
                                unsigned bad = 0;
                                double worst = 0;
                                for (unsigned i = 0; i < ch; ++i) {
                                    const double tol_q = 1e-6 + 1e-5 * std::fabs(double(q[i])) + 1e-6 * mag[i];
                                    const double ey = std::fabs(double(act[y_off + i]) - ref[i]);
                                    const double es = std::fabs(double(got[size_t(i) * 3 + 2]) - q[i]);
                                    worst = std::max(worst, ey);
                                    if (ey > 1.1 * std::fabs(double(act[w_off + size_t(i) * 4 + 3])) * tol_q + 1e-6 +
                                                 2e-5 * std::fabs(double(ref[i])) ||
                                        es > tol_q || got[size_t(i) * 3] != new_state[size_t(i) * 3] ||
                                        got[size_t(i) * 3 + 1] != new_state[size_t(i) * 3 + 1]) {
                                        ++bad;
                                    }
                                }
                                fprintf(stderr, "XRT-VERIFY: %s gemvqc16 conv epilog %s n=%u bad=%u max_abs=%g\n",
                                        bad ? "FAIL" : "PASS", node->name, ch, bad, worst);
                                if (bad) throw std::runtime_error("gemvqc16 result differs from the CPU reference");
                            });
                        }
                        break;
                    }
                    list.cmds.insert(list.cmds.end(), conv_cmd, conv_cmd + XRT_QC_CMD_WORDS);
                    if (verify) {
                        const auto old_state = xrt_qc_read_act(ctx, s_src, size_t(ch) * 3);
                        const unsigned x_off = offs[0], w_off = offs[1];
                        list.verifies.push_back([=](xrt_backend_context * c) {
                            const float * act = c->qc_act_bo.map<const float *>();
                            std::vector<float> new_state;
                            const auto ref = gemvq_ref::conv(old_state.data(), act + x_off, act + w_off, ch, true, new_state);
                            xrt_qc_report("conv", node->name, act + y_off, ref, {});
                            const auto got = xrt_qc_read_act(c, s_dst, size_t(ch) * 3);
                            xrt_qc_report("conv state", node->name, got.data(), new_state, {});
                        });
                    }
                    break;
                }
                if (ggml_get_unary_op(node) == GGML_UNARY_OP_SILU && next != nullptr && next->op == GGML_OP_MUL &&
                    next->src[1] == node && next->src[0] != node && use_count(node) == 1 &&
                    ggml_nelements(next) == ggml_nelements(node)) {
                    fused_silu.push_back(node);
                    break;  // emitted with the MUL
                }
                unsigned kind = 0;
                xrt_qc_unary_kind(node, &kind);
                // With a folded CONT the unary reads the strided rows behind it.
                const ggml_tensor * x = cont_folded.count(node) != 0 ? a->src[0] : a;
                const bool strided = x != a;
                const unsigned row = strided ? unsigned(x->ne[0]) : 0U, stride = strided ? unsigned(x->nb[1] / sizeof(float)) : 0U;
                const auto offs = xrt_qc_operands(ctx, list, {{x, x->data, strided ? xrt_qc_rows_span(x) : n}}, n, 1);
                const unsigned y_off = xrt_qc_output(list, node, n);
                xrt_qc_push(list, {XRT_QC_OP_UNARY, unsigned(n), offs[0], kind, y_off, row, stride});
                if (verify) {
                    const unsigned x_off = offs[0];
                    list.verifies.push_back([=](xrt_backend_context * c) {
                        const float * act = c->qc_act_bo.map<const float *>();
                        std::vector<float> xs;
                        for (size_t i = 0; i < n; i += (row ? row : n)) {
                            const float * src = act + x_off + (row ? (i / row) * stride : 0);
                            xs.insert(xs.end(), src, src + (row ? row : n));
                        }
                        xrt_qc_report("unary", node->name, act + y_off, gemvq_ref::unary(xs.data(), unsigned(n), kind), {});
                    });
                }
                break;
            }
            case GGML_OP_ROPE: {
                const unsigned nd = unsigned(ggml_get_op_params_i32(node, 1)), ne0 = unsigned(node->ne[0]);
                const unsigned rows = unsigned(n / ne0);
                // The cache is uploaded from cs, which then moves into the list (same buffer): its address stays
                // unique while the list lives, also if the operands flushed the list before.
                std::vector<float> cs = xrt_rope_cache(node);
                const std::vector<float> cache = cs;
                const auto offs = xrt_qc_operands(ctx, list, {{a, a->data, n}, {nullptr, cs.data(), cs.size()}}, n, 1);
                list.owned.push_back(std::move(cs));
                const unsigned y_off = xrt_qc_output(list, node, n);
                xrt_qc_push(list, {XRT_QC_OP_ROPE, nd, offs[0], offs[1], y_off, ne0, rows});
                if (verify) {
                    const unsigned x_off = offs[0];
                    list.verifies.push_back([=](xrt_backend_context * c) {
                        const float * act = c->qc_act_bo.map<const float *>();
                        std::vector<float> ref(act + x_off, act + x_off + n);
                        for (unsigned r = 0; r < rows; ++r) {
                            for (unsigned i = 0; i < nd / 2; ++i) {  // ggml rotate_pairs, NEOX pairing
                                const float x0 = act[x_off + r * ne0 + i], x1 = act[x_off + r * ne0 + i + nd / 2];
                                ref[r * ne0 + i] = x0 * cache[2 * i] - x1 * cache[2 * i + 1];
                                ref[r * ne0 + i + nd / 2] = x0 * cache[2 * i + 1] + x1 * cache[2 * i];
                            }
                        }
                        xrt_qc_report("rope", node->name, act + y_off, ref, {});
                    });
                }
                break;
            }
            case GGML_OP_CONT: {  // not folded into a unary: a strided copy
                const ggml_tensor * x = a;
                const auto offs = xrt_qc_operands(ctx, list, {{x, x->data, xrt_qc_rows_span(x)}}, n, 1);
                const unsigned y_off = xrt_qc_output(list, node, n);
                xrt_qc_push(list, {XRT_QC_OP_UNARY, unsigned(n), offs[0], XRT_QC_UNARY_COPY, y_off, unsigned(x->ne[0]),
                                   unsigned(x->nb[1] / sizeof(float))});
                break;
            }
            case GGML_OP_SSM_CONV:
                fprintf(stderr, "XRT: conv %s without SILU is not supported\n", node->name);
                return GGML_STATUS_FAILED;
            case GGML_OP_GATED_DELTA_NET: {
                const gdn_plan & p = gdns.at(node);
                const ggml_tensor * q = node->src[0], * k = node->src[1], * v = node->src[2];
                const ggml_tensor * g = node->src[3], * be = node->src[4];
                const unsigned hv = unsigned(v->ne[1]), hk = unsigned(q->ne[1]);
                const size_t S = XRT_QC_GDN_HEAD;
                size_t src_within = 0, dst_within = 0;
                const auto & src_t = xrt_rs_at(p.get_rows->src[0], xrt_qc_state_row(p.get_rows), &src_within);
                const auto & dst_t = xrt_rs_at(p.cpy->src[1], p.cpy->src[1]->data, &dst_within);
                if (!src_t.striped || !dst_t.striped || src_within % (XRT_QC_STATE_ROW * XRT_QC_STATE_PORTS) != 0 ||
                    dst_within % (XRT_QC_STATE_ROW * XRT_QC_STATE_PORTS) != 0) {
                    fprintf(stderr, "XRT: GDN state of %s is not row-aligned on the state ports\n", node->name);
                    return GGML_STATUS_FAILED;
                }
                const unsigned src_row = unsigned(src_t.card + src_within / XRT_QC_STATE_ROW);
                const unsigned dst_row = unsigned(dst_t.card + dst_within / XRT_QC_STATE_ROW);
                const float scale = 1.0f / sqrtf(float(S));
                const auto ni = norms_of.find(node);
                const gdn_norms nm = ni != norms_of.end() ? ni->second : gdn_norms();
                const ggml_tensor * qi = nm.l2 ? q->src[0] : q, * ki = nm.l2 ? k->src[0] : k;  // raw q/k with fused L2
                const bool gates = nm.alpha_raw != nullptr;
                const ggml_tensor * gi = gates ? nm.alpha_raw : g, * bi = gates ? nm.beta_raw : be;
                std::vector<xrt_qc_operand> ops = {{qi, qi->data, hk * S}, {ki, ki->data, hk * S}, {v, v->data, hv * S},
                                                   {gi, gi->data, hv}, {bi, bi->data, hv}};
                if (nm.out != nullptr) {
                    ops.push_back({nm.w, nm.w->data, S});
                    ops.push_back({nm.z, nm.z->data, hv * S});
                }
                if (gates) {
                    ops.push_back({nm.dt, nm.dt->data, hv});
                    ops.push_back({nm.a, nm.a->data, hv});
                }
                const auto offs = xrt_qc_operands(ctx, list, ops, hv * S, (state_load ? 3 : 2) + (gates ? 1 : 0));
                // With the gated norm the command writes only its result (the node behind the norm).
                const unsigned y_off = xrt_qc_output(list, nm.out != nullptr ? nm.out : node, hv * S);
                const unsigned norm_flags = (nm.l2 ? XRT_QC_GDN_L2_QK : 0U) | (nm.out != nullptr ? XRT_QC_GDN_GATED : 0U) |
                                            (gates ? XRT_QC_GDN_GATES : 0U);
                const unsigned dt_off = gates ? offs[ops.size() - 2] : 0U, a_off = gates ? offs[ops.size() - 1] : 0U;
                const unsigned w_off = nm.out != nullptr ? offs[5] : 0U, z_off = nm.out != nullptr ? offs[6] : 0U;
                // The step's GEMV, if it is still in this list: its command moves behind the step (nothing in between
                // reads its y, and act regions of one list are never reused).
                std::vector<unsigned> late_cmd;
                if (const auto lg = late_gemv.find(node); lg != late_gemv.end()) {
                    if (const auto la = list.late_at.find(lg->second); la != list.late_at.end()) {
                        const auto at = list.cmds.begin() + std::ptrdiff_t(la->second);
                        late_cmd.assign(at, at + XRT_QC_CMD_WORDS);
                        const size_t pos = la->second;
                        list.cmds.erase(at, at + XRT_QC_CMD_WORDS);
                        list.late_at.erase(la);
                        list.shift_positions(pos + XRT_QC_CMD_WORDS, -std::ptrdiff_t(XRT_QC_CMD_WORDS));
                    }
                }
                if (state_load) {
                    // Read the state while the commands since the previous GDN step (projections, conv) run.
                    const unsigned load[XRT_QC_CMD_WORDS] = {XRT_QC_OP_STATE_LOAD, hv, src_row, 0, 0, 0, 0, 0};
                    list.cmds.insert(list.cmds.begin() + std::ptrdiff_t(list.state_load_at), load, load + XRT_QC_CMD_WORDS);
                    list.shift_positions(list.state_load_at, XRT_QC_CMD_WORDS);
                }
                xrt_qc_push(list, {XRT_QC_OP_GDN, hv | (hk << 16), offs[0], offs[1], offs[2], offs[3], offs[4], y_off});
                xrt_qc_push(list, {src_row, dst_row, xrt_qc_float_bits(scale),
                                   (late_cmd.empty() ? 0U : XRT_QC_GDN_WITH_GEMV) | norm_flags, xrt_qc_float_bits(nm.l2_eps),
                                   xrt_qc_float_bits(nm.rms_eps), w_off, z_off});
                if (gates) xrt_qc_push(list, {dt_off, a_off});
                list.cmds.insert(list.cmds.end(), late_cmd.begin(), late_cmd.end());
                list.state_load_at = list.cmds.size();
                if (verify) {
                    auto state = xrt_qc_read_state(ctx, src_row, hv * S);
                    list.verifies.push_back([=](xrt_backend_context * c) mutable {
                        const float * act = c->qc_act_bo.map<const float *>();
                        std::vector<double> y_mag, s_mag;
                        std::vector<float> qn(act + offs[0], act + offs[0] + hk * S), kn(act + offs[1], act + offs[1] + hk * S);
                        if (nm.l2) {
                            qn = gemvq_ref::rownorm(qn.data(), nullptr, unsigned(S), hk, nm.l2_eps, XRT_QC_NORM_L2 | XRT_QC_NORM_NO_WEIGHT);
                            kn = gemvq_ref::rownorm(kn.data(), nullptr, unsigned(S), hk, nm.l2_eps, XRT_QC_NORM_L2 | XRT_QC_NORM_NO_WEIGHT);
                        }
                        std::vector<float> gv(act + offs[3], act + offs[3] + hv), bv(act + offs[4], act + offs[4] + hv);
                        if (gates) {  // the gate commands' arithmetic on the raw projections
                            bv = gemvq_ref::unary(bv.data(), hv, XRT_QC_UNARY_SIGMOID);
                            const auto t1 = gemvq_ref::add(gv.data(), act + dt_off, hv);
                            const auto t2 = gemvq_ref::unary(t1.data(), hv, XRT_QC_UNARY_SOFTPLUS);
                            gv = gemvq_ref::mul(t2.data(), act + a_off, hv);
                        }
                        const auto ref = gemvq_ref::gdn(state, qn.data(), kn.data(), act + offs[2], gv.data(), bv.data(),
                                                        hv, hk, scale, y_mag, s_mag);
                        if (nm.out == nullptr) {
                            xrt_qc_report("gdn", node->name, act + y_off, ref, y_mag);
                        } else {
                            // out = (y * s_h) * w * silu(z): the y tolerance scaled by s_h * |w| * |silu(z)|.
                            const float * w = act + w_off, * z = act + z_off;
                            const auto nref = gemvq_ref::rownorm(ref.data(), w, unsigned(S), hv, nm.rms_eps, 0);
                            const auto oref = gemvq_ref::silu_mul(z, nref.data(), hv * unsigned(S));
                            unsigned bad = 0;
                            double worst = 0;
                            for (unsigned h = 0; h < hv; ++h) {
                                double sum = 0;
                                for (size_t j = 0; j < S; ++j) sum += double(ref[h * S + j]) * ref[h * S + j];
                                const double sh = 1.0 / std::sqrt(sum / double(S) + nm.rms_eps);
                                for (size_t j = 0; j < S; ++j) {
                                    const size_t i = h * S + j;
                                    const double sz = std::fabs(double(z[i]) / (1.0 + std::exp(-double(z[i]))));
                                    const double tol = (1e-6 + 1e-5 * std::fabs(double(ref[i])) + 4e-6 * y_mag[i]) * sh *
                                                           std::fabs(double(w[j])) * sz * 1.1 + 1e-6 + 2e-5 * std::fabs(double(oref[i]));
                                    const double err = std::fabs(double(act[y_off + i]) - oref[i]);
                                    worst = std::max(worst, err);
                                    if (err > tol) ++bad;
                                }
                            }
                            fprintf(stderr, "XRT-VERIFY: %s gemvqc16 gdn gated %s n=%u bad=%u max_abs=%g\n",
                                    bad ? "FAIL" : "PASS", node->name, hv * unsigned(S), bad, worst);
                            if (bad) throw std::runtime_error("gemvqc16 result differs from the CPU reference");
                        }
                        const auto got = xrt_qc_read_state(c, dst_row, hv * S);
                        xrt_qc_report("gdn state", node->name, got.data(), state, s_mag);
                    });
                }
                break;
            }
            case GGML_OP_GET_ROWS:
            case GGML_OP_CPY:
            case GGML_OP_SCALE:
                xrt_qc_flush(ctx, list);
                {
                    xrt_scoped_ns state_timer{g_xrt_timing.state_ns};
                    xrt_qc_state_op(ctx, node);
                }
                break;
            default:
                fprintf(stderr, "XRT: gemvqc16 cannot run node op=%d (%s)\n", node->op, node->name);
                return GGML_STATUS_FAILED;
        }
    }
    list.pos = n_nodes;
    xrt_qc_flush(ctx, list);
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_xrt_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto ctx = (xrt_backend_context*)backend->context;
    if (!ctx->runtime_valid) {
        fprintf(stderr, "XRT: backend invalidated by an earlier failure\n");
        return GGML_STATUS_FAILED;
    }
    xrt_scoped_ns graph_timer{g_xrt_timing.graph_ns};
    ++g_xrt_timing.graph_count;

    if (xrt_debug_enabled()) {
        fprintf(stderr, "XRT-DEBUG: graph_compute nodes=%d first=%s last=%s\n", cgraph->n_nodes,
                cgraph->n_nodes ? cgraph->nodes[0]->name : "-", cgraph->n_nodes ? cgraph->nodes[cgraph->n_nodes - 1]->name : "-");
    }

    try {
        return ctx->qc_gdn ? xrt_qc_graph_compute_gdn(ctx, cgraph) : xrt_qc_graph_compute(ctx, cgraph);
    } catch (const std::exception & error) {
        fprintf(stderr, "XRT: graph execution failed: %s\n", error.what());
    } catch (...) {
        fprintf(stderr, "XRT: graph execution failed with an unknown exception\n");
    }
    // A failed or timed-out launch leaves the card in an unknown state: refuse further work.
    ctx->runtime_valid = false;
    return GGML_STATUS_FAILED;
}

static const struct ggml_backend_i xrt_backend_iface = {
    .get_name          = ggml_backend_xrt_get_name,
    .free              = ggml_backend_xrt_free,
    .set_tensor_async  = NULL,
    .get_tensor_async  = NULL,
    .cpy_tensor_async  = NULL,
    .graph_compute     = ggml_backend_xrt_graph_compute,
    .event_record      = NULL,
    .event_wait        = NULL,
};

static const char * ggml_backend_xrt_dev_get_name(ggml_backend_dev_t dev) { (void)dev; return "Alveo_U50_XRT"; }
static const char * ggml_backend_xrt_dev_get_description(ggml_backend_dev_t dev) { (void)dev; return "AMD Alveo U50 XRT Device"; }
static enum ggml_backend_dev_type ggml_backend_xrt_dev_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    // llama.cpp's --n-gpu-layers device selection targets GPU-class backends.
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}
static void ggml_backend_xrt_dev_get_memory(ggml_backend_dev_t dev, size_t * f, size_t * t) { (void)dev; if(f) *f = 8ULL*1024*1024*1024; if(t) *t = 8ULL*1024*1024*1024; }
static void ggml_backend_xrt_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * p) { (void)dev; p->name = "Alveo_U50_XRT"; p->type = GGML_BACKEND_DEVICE_TYPE_GPU; }

static ggml_backend_t ggml_backend_xrt_dev_init_backend(ggml_backend_dev_t dev, const char * p) {
    (void)p;
    const char * emulate_env = std::getenv("GGML_XRT_EMULATE");
    const bool emulate = emulate_env != nullptr && std::strcmp(emulate_env, "1") == 0;
    xrt::device xrt_dev;

    if (!emulate) {
        const char * device_env = std::getenv("GGML_XRT_DEVICE");
        const std::string device_id = device_env == nullptr ? "0" : device_env;
        try {
            const xrt_device_selector selection = xrt_device_selector_from_env("GGML_XRT_DEVICE");
            xrt_dev = selection.is_index ? xrt::device(selection.index) : xrt::device(selection.bdf);
            fprintf(stderr, "XRT Backend: device %s opened.\n", device_id.c_str());
        } catch (const std::exception& e) {
            fprintf(stderr, "XRT Backend: cannot initialize device %s: %s\n", device_id.c_str(), e.what());
            return nullptr;
        }
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(struct ggml_backend));
    if (!backend) return nullptr;
    memset(backend, 0, sizeof(struct ggml_backend));

    xrt_backend_context * ctx = nullptr;

    try {
        ctx = new xrt_backend_context();
        if (!emulate) {
            ctx->device = xrt_dev;
        }
        ctx->verify_calls_remaining = xrt_verify_call_count();
        ctx->kernel_name = xrt_kernel_name();
        if (ctx->kernel_name != "gemvqc16") {
            throw std::runtime_error("GGML_XRT_KERNEL must be gemvqc16 (the only supported kernel)");
        }
        if (emulate) {
            // gemvqc16 as its HLS C++ model on host memory: tests the whole backend without a card (slow, one pass
            // over all weights takes seconds).
            ctx->qc_gdn = ctx->qc_emulate = true;
#ifdef GGML_XRT_EMULATE_ENGINES
            ctx->qc_engines = GGML_XRT_EMULATE_ENGINES;
#endif
            ctx->qc_arena_bytes = XRT_QC_ARENA_BYTES_PAIR;
            for (unsigned e = 0; e < ctx->qc_engines; ++e) ctx->qc_weight_bos.emplace_back(ctx->qc_arena_bytes);
            ctx->qc_act_bo = xrt_qc_mem(XRT_QC_ACT_BYTES_GDN);
            ctx->qc_cmd_bo = xrt_qc_mem(XRT_QC_CMD_BYTES);
            for (unsigned p = 0; p < XRT_QC_STATE_PORTS; ++p) ctx->qc_state_bos.emplace_back(XRT_QC_STATE_BYTES);
            ctx->qc_features = xrt_qc_probe_features(ctx);
            xrt_activate_backend(ctx);
            fprintf(stderr, "XRT Backend: gemvqc16 emulated by its HLS C++ model (DeltaNet opcodes, no card).\n");
            backend->iface = xrt_backend_iface;
            backend->device = dev;
            backend->context = ctx;
            return backend;
        }
        const char * xclbin_env = std::getenv("GGML_XRT_XCLBIN");
        std::string xclbin_path = xclbin_env ? xclbin_env : ctx->kernel_name + ".xclbin";

        fprintf(stderr, "XRT Backend: loading XCLBIN from %s ...\n", xclbin_path.c_str());
        // With ert_disable=1 only the first load after boot or the xclbin that is already loaded are safe.
        const xrt::uuid loaded = ctx->device.get_xclbin_uuid();
        if (xrt_ert_disable(ctx->device) == 1 && loaded && loaded != xrt::xclbin(xclbin_path).get_uuid()) {
            throw std::runtime_error("ert_disable=1 and another xclbin is loaded: loading " + xclbin_path +
                                     " now would deadlock xocl, set ert_disable=0 first");
        }
        auto uuid = ctx->device.load_xclbin(xclbin_path);
        ctx->kernels.emplace_back(ctx->device, uuid, ctx->kernel_name);
        // Memory bank of a kernel argument. XRT answers an argument the kernel does not have with an invalid bank
        // instead of an error, and xocl oopses on a buffer in such a bank: check before every allocation.
        const auto bank = [&](int arg) {
            const int group = ctx->kernels[0].group_id(arg);
            if (group < 0 || group > 1024) {
                throw std::runtime_error("kernel argument " + std::to_string(arg) + " has no memory bank");
            }
            return group;
        };
        // Engine ports on single channels get 250 MiB arenas, ports on channel pairs 384 MiB. The arenas are
        // allocated first, so each starts at the bottom of its pair and the other buffers fit behind it.
        const xrt::xclbin xclbin(xclbin_path);
        // One weight port w<e> per engine (16 or 20).
        ctx->qc_engines = 0;
        for (const auto & arg : xclbin.get_kernel(ctx->kernel_name).get_args()) {
            const std::string name = arg.get_name();
            if (name.size() > 1 && name[0] == 'w' && name.find_first_not_of("0123456789", 1) == std::string::npos) {
                ++ctx->qc_engines;
            }
        }
        if (ctx->qc_engines != 16 && ctx->qc_engines != 20) {
            throw std::runtime_error("gemvqc16: unexpected number of weight ports (" + std::to_string(ctx->qc_engines) + ")");
        }
        size_t engine_bank_bytes = SIZE_MAX;
        for (unsigned e = 0; e < ctx->qc_engines; ++e) {
            size_t bytes = 0;
            for (const auto & mem : xclbin.get_mems()) {
                if (mem.get_index() == bank(int(e))) bytes = size_t(mem.get_size_kb()) << 10;
            }
            engine_bank_bytes = std::min(engine_bank_bytes, bytes);
        }
        ctx->qc_arena_bytes = engine_bank_bytes >= (512ULL << 20) ? XRT_QC_ARENA_BYTES_PAIR : XRT_QC_ARENA_BYTES;
        if (engine_bank_bytes < ctx->qc_arena_bytes + (6ULL << 20)) {
            throw std::runtime_error("gemvqc16 engine memory bank too small for the weight arena");
        }
        for (unsigned e = 0; e < ctx->qc_engines; ++e) {
            ctx->qc_weight_bos.emplace_back(xrt::bo(ctx->device, ctx->qc_arena_bytes, bank(int(e))));
        }
        const int act_arg = int(ctx->qc_engines);
        // A kernel with DeltaNet opcodes has four pairs of state ports after n_cmds, found by name in the xclbin
        // (GGML_XRT_GDN=0 ignores them).
        const xrt::xclbin::kernel meta = xclbin.get_kernel(ctx->kernel_name);
        const char * gdn_env = std::getenv("GGML_XRT_GDN");
        if (gdn_env == nullptr || std::strcmp(gdn_env, "0") != 0) {
            for (const auto & arg : meta.get_args()) {
                if (arg.get_name() == "st0_rd") ctx->qc_gdn = true;
            }
            ctx->qc_gdn = ctx->qc_gdn && (meta.get_num_args() == size_t(act_arg) + 3 + 2 * XRT_QC_STATE_PORTS ||
                                          meta.get_num_args() == size_t(act_arg) + 4 + 2 * XRT_QC_STATE_PORTS);
        }
        // Kernels with the CONV epilog read act a second time through act_rd (last argument, same buffer).
        int act_rd_arg = -1;
        for (const auto & arg : meta.get_args()) {
            if (arg.get_name() == "act_rd") act_rd_arg = int(arg.get_index());
        }
        if (act_rd_arg >= 0 && (!ctx->qc_gdn || act_rd_arg != act_arg + 3 + int(2 * XRT_QC_STATE_PORTS))) {
            throw std::runtime_error("gemvqc16: unexpected position of act_rd in the xclbin");
        }
        ctx->qc_act_bo = xrt_qc_mem(xrt::bo(ctx->device, ctx->qc_gdn ? XRT_QC_ACT_BYTES_GDN : XRT_QC_ACT_BYTES,
                                            bank(act_arg)));
        ctx->qc_cmd_bo = xrt_qc_mem(xrt::bo(ctx->device, XRT_QC_CMD_BYTES, bank(act_arg + 1)));
        // st<p>_rd and st<p>_wr share one AXI port and get the same buffer.
        for (unsigned p = 0; ctx->qc_gdn && p < XRT_QC_STATE_PORTS; ++p) {
            ctx->qc_state_bos.emplace_back(xrt::bo(ctx->device, XRT_QC_STATE_BYTES, bank(act_arg + 3 + int(2 * p))));
        }
        const auto buffer_of = [&](int arg) -> const xrt::bo & {
            if (arg < act_arg) return ctx->qc_weight_bos[arg].bo;
            if (arg == act_arg) return ctx->qc_act_bo.bo;
            if (arg == act_arg + 1) return ctx->qc_cmd_bo.bo;
            if (arg == act_rd_arg) return ctx->qc_act_bo.bo;
            return ctx->qc_state_bos[(arg - act_arg - 3) / 2].bo;
        };
        const int n_buffer_args = act_rd_arg >= 0 ? act_rd_arg + 1
                                                  : act_arg + 2 + (ctx->qc_gdn ? int(2 * XRT_QC_STATE_PORTS) + 1 : 0);
        const char * direct_env = std::getenv("GGML_XRT_DIRECT_START");
        ctx->qc_direct = direct_env != nullptr && std::strcmp(direct_env, "0") != 0;
        if (ctx->qc_direct) {
            if (xrt_ert_disable(ctx->device) != 1) {
                throw std::runtime_error("GGML_XRT_DIRECT_START needs ert_disable=1 (load the xclbin with ert_disable=0 first)");
            }
            std::vector<uint32_t> reg(meta.get_num_args());
            for (const auto & arg : meta.get_args()) reg.at(arg.get_index()) = uint32_t(arg.get_offset());
            ctx->kernels.clear();  // frees the shared CU context for the exclusive register access
            ctx->qc_ip = xrt::ip(ctx->device, uuid, ctx->kernel_name + ":{" + ctx->kernel_name + "_1}");
            for (int arg = 0; arg < n_buffer_args; ++arg) {
                if (arg == act_arg + 2) continue;  // n_cmds, written per launch
                const uint64_t address = buffer_of(arg).address();
                ctx->qc_ip.write_register(reg.at(arg), uint32_t(address));
                ctx->qc_ip.write_register(reg.at(arg) + 4, uint32_t(address >> 32));
            }
            ctx->qc_ncmds_reg = reg.at(act_arg + 2);
        } else {
            ctx->qc_run = xrt::run(ctx->kernels[0]);
            for (int arg = 0; arg < n_buffer_args; ++arg) {
                if (arg != act_arg + 2) ctx->qc_run.set_arg(arg, buffer_of(arg));
            }
        }
        ctx->qc_features = xrt_qc_probe_features(ctx);
        xrt_activate_backend(ctx);
        const std::string formats = std::string("Q4_0/Q8_0") +
                                    ((ctx->qc_features & XRT_QC_FEATURE_Q4_1) ? "/Q4_1" : "") +
                                    ((ctx->qc_features & XRT_QC_FEATURE_Q5_K) ? "/Q5_K" : "") +
                                    ((ctx->qc_features & XRT_QC_FEATURE_Q6_K) ? "/Q6_K" : "");
        fprintf(stderr, "XRT Backend: kernel initialized: %s (%u engines, %zu MiB weight arena per engine, %s%s%s).\n",
                ctx->kernel_name.c_str(), ctx->qc_engines, ctx->qc_arena_bytes >> 20, formats.c_str(),
                ctx->qc_gdn ? ", DeltaNet opcodes, recurrent states on the card" : "",
                ctx->qc_direct ? ", direct register start" : "");
    } catch (const std::exception& e) {
        fprintf(stderr, "XRT Backend: initialization failed: %s\n", e.what());
        fprintf(stderr, "Set GGML_XRT_XCLBIN to the gemvqc16 xclbin (see docs/BUILD.md).\n");
        delete ctx;
        free(backend);
        return nullptr;
    }

    backend->iface = xrt_backend_iface;
    backend->device = dev;
    backend->context = (void*)ctx;

    return backend;
}

// Weights stay in host memory (the backend uploads its own repacked copy), so the scheduler needs no device buffers.
static ggml_backend_buffer_type_t ggml_backend_xrt_dev_get_buffer_type(ggml_backend_dev_t dev) { (void)dev; return ggml_backend_cpu_buffer_type(); }
static ggml_backend_buffer_type_t ggml_backend_xrt_dev_get_host_buffer_type(ggml_backend_dev_t dev) { (void)dev; return ggml_backend_cpu_buffer_type(); }

extern "C" ggml_backend_buffer_type_t ggml_backend_xrt_rs_buffer_type(ggml_backend_dev_t dev) {
    auto * device_ctx = xrt_device_ctx(dev);
    if (device_ctx == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(device_ctx->mutex);
    return device_ctx->active_backend != nullptr && device_ctx->active_backend->qc_gdn ? &xrt_rs_buft : nullptr;
}

static bool ggml_backend_xrt_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    if (op->op == GGML_OP_NONE && op->buffer != nullptr && op->buffer->buft == &xrt_rs_buft) {
        // Persistent recurrent-state tensors are preallocated storage, not a compute node.
        return true;
    }
    const ggml_backend_buffer_t op_buffer = op->buffer != nullptr ? op->buffer :
        (op->view_src != nullptr ? op->view_src->buffer : nullptr);
    if ((op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE ||
          op->op == GGML_OP_TRANSPOSE || op->op == GGML_OP_RESHAPE) && op_buffer != nullptr &&
        op_buffer->buft == &xrt_rs_buft) {
        return true;
    }
    if (xrt_active_qc_mode()) {
        return xrt_active_qc_gdn() ? xrt_qc_supports_gdn(op) : xrt_qc_supports(op);
    }
    return false;
}

static bool ggml_backend_xrt_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return ggml_backend_xrt_dev_supports_op(dev, op);
}

static bool ggml_backend_xrt_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    // Any host buffer is usable in place. This matters for the mmap-ed model weights ("CPU_Mapped", a different
    // buffer type object than the plain CPU one): if they are reported as unsupported, the scheduler copies every
    // weight tensor into a backend buffer before each split although the backend uploads its own device copy.
    return ggml_backend_buft_is_host(buft) || buft == &xrt_rs_buft;
}

extern "C" {
    __attribute__((used)) ggml_backend_reg_t ggml_backend_xrt_reg(void) {
        static bool init = false;
        if (!init) {
            memset(&xrt_reg_instance, 0, sizeof(xrt_reg_instance));
            xrt_reg_instance.api_version = GGML_BACKEND_API_VERSION;

            xrt_reg_instance.iface.get_name = [](ggml_backend_reg_t) -> const char* { return "XRT"; };
            xrt_reg_instance.iface.get_device_count = [](ggml_backend_reg_t) -> size_t { return 1; };
            xrt_reg_instance.iface.get_device = [](ggml_backend_reg_t, size_t i) -> ggml_backend_dev_t {
                return (i == 0) ? &ggml_xrt_device_inst : nullptr;
            };

            memset(&ggml_xrt_device_inst, 0, sizeof(ggml_xrt_device_inst));
            ggml_xrt_device_inst.iface.get_name = ggml_backend_xrt_dev_get_name;
            ggml_xrt_device_inst.iface.get_description = ggml_backend_xrt_dev_get_description;
            ggml_xrt_device_inst.iface.get_memory = ggml_backend_xrt_dev_get_memory;
            ggml_xrt_device_inst.iface.get_type = ggml_backend_xrt_dev_get_type;
            ggml_xrt_device_inst.iface.get_props = ggml_backend_xrt_dev_get_props;
            ggml_xrt_device_inst.iface.init_backend = ggml_backend_xrt_dev_init_backend;
            ggml_xrt_device_inst.iface.get_buffer_type = ggml_backend_xrt_dev_get_buffer_type;
            ggml_xrt_device_inst.iface.get_host_buffer_type = ggml_backend_xrt_dev_get_host_buffer_type;
            ggml_xrt_device_inst.iface.supports_op = ggml_backend_xrt_dev_supports_op;
            ggml_xrt_device_inst.iface.offload_op = ggml_backend_xrt_dev_offload_op;
            ggml_xrt_device_inst.iface.supports_buft = ggml_backend_xrt_dev_supports_buft;

            ggml_xrt_device_inst.reg = &xrt_reg_instance;
            ggml_xrt_device_inst.context = &xrt_device_context_inst;

            init = true;
        }
        return &xrt_reg_instance;
    }
}
#pragma GCC diagnostic pop
