// SPDX-License-Identifier: Apache-2.0
// Element-wise opcodes of gemvqc16 on FP32 vectors in the 512-bit activation buffer (16 lanes per word).
// Every op reads its operands with bursts into local buffers, computes, and writes back with a burst. The order
// of operations follows ggml: RMSNorm as (x * scale) * w, SiLU as x / (1 + exp(-x)).
#ifndef FPGA_LLAMA_GGUF_GEMVQC_OPS_H
#define FPGA_LLAMA_GGUF_GEMVQC_OPS_H
#include "gemvqw_engine.h"
#ifdef __SYNTHESIS__
#include <hls_math.h>
#else
#include <cmath>
namespace hls {
inline float sqrtf(float value) { return std::sqrt(value); }
inline float expf(float value) { return std::exp(value); }
inline float logf(float value) { return std::log(value); }
} // namespace hls
#endif

namespace gemvqc_ops {
namespace {

using gemvqw_engine::word_lane_float;
typedef ap_uint<512> vword;
// On-chip vector buffers hold up to 4096 words (65536 floats); a 512-bit URAM array costs the same up to this depth.
constexpr unsigned BUF_WORDS = 4096;
constexpr unsigned MAX_NORM_ROWS = 4096;
constexpr unsigned MAX_CONV_CHANNELS = 16384;

// RMSNORM flags
constexpr unsigned NORM_L2 = 1U;         // scale = 1 / max(sqrt(sum x^2), eps) (ggml_l2_norm)
constexpr unsigned NORM_NO_WEIGHT = 2U;  // y = x * scale
constexpr unsigned NORM_DEFER = 4U;      // one row kept on chip for the next GEMVs on it (see norm_defer)
constexpr unsigned NX_WORDS = 1024;      // longest deferred row: 16384 floats
// UNARY kinds
constexpr unsigned UNARY_SILU = 1U, UNARY_SIGMOID = 2U, UNARY_SOFTPLUS = 3U, UNARY_COPY = 4U;
// CONV flags
constexpr unsigned CONV_SILU = 1U;

uint32_t lane_bits(float value) {
#pragma HLS INLINE
    union { uint32_t bits; float value; } c;
    c.value = value;
    return c.bits;
}

void set_lane(vword & w, unsigned lane, float value) {
#pragma HLS INLINE
    w.range(lane * 32 + 31, lane * 32) = lane_bits(value);
}

float tree16(const float v[16]) {
#pragma HLS INLINE
    return (((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))) +
           (((v[8] + v[9]) + (v[10] + v[11])) + ((v[12] + v[13]) + (v[14] + v[15])));
}

float tree8(const float v[8]) {
#pragma HLS INLINE
    return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
}

// Each non-linear scalar op lives in its own pipelined function: every call site gets its own instance with
// its own operators, so the calling module is not tied to a shared top-level operator through a state mux.
float silu_f(float x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    return x / (1.0f + hls::expf(-x));
}

float unary_f(unsigned kind, float x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    const float e = hls::expf(kind == UNARY_SOFTPLUS ? x : -x);
    float r;
    if (kind == UNARY_SILU) r = x / (1.0f + e);
    else if (kind == UNARY_SIGMOID) r = 1.0f / (1.0f + e);
    else r = x > 20.0f ? x : hls::logf(1.0f + e);
    return r;
}

// Lane datapaths of the element-wise ops, one function per op (see silu_f).
float add_mul_f(float a, float b, bool mul) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    return mul ? a * b : a + b;
}

float silu_mul_f(float g, float u) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    return (g / (1.0f + hls::expf(-g))) * u;
}

float sumsq16(const vword & w) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    float sq[16];
#pragma HLS ARRAY_PARTITION variable=sq complete
    for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
        const float v = word_lane_float(w, l);
        sq[l] = v * v;
    }
    return tree16(sq);
}

float add_f(float a, float b) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    return a + b;
}

float row_scale(float p0, float p1, float p2, float p3, float p4, float p5, float p6, float p7, unsigned n, float eps,
                bool l2) {
#pragma HLS INLINE off
    const float part[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
    const float sum = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
    return l2 ? 1.0f / (hls::sqrtf(sum) > eps ? hls::sqrtf(sum) : eps) : 1.0f / hls::sqrtf(sum / float(n) + eps);
}

float norm_apply_f(float x, float g, float s, bool weighted) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    const float v = x * s;
    return weighted ? v * g : v;
}

float conv_chan(float in0, float in1, float in2, float x, float k0, float k1, float k2, float k3) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    float sum = 0.0f;
    sum += in0 * k0;
    sum += in1 * k1;
    sum += in2 * k2;
    sum += x * k3;
    return sum;
}

void load_words(const vword * src, unsigned words, vword * dst) {
    LOAD_WORDS: for (unsigned i = 0; i < words; ++i) {
#pragma HLS PIPELINE II=1
        dst[i] = src[i];
    }
}

void store_words(const vword * src, unsigned words, vword * dst) {
    STORE_WORDS: for (unsigned i = 0; i < words; ++i) {
#pragma HLS PIPELINE II=1
        dst[i] = src[i];
    }
}

// rows rows of row_words words, row r at src + r * stride words (stride == row_words: one burst), packed into dst.
void load_rows(const vword * src, unsigned row_words, unsigned rows, unsigned stride, vword * dst) {
    if (stride == row_words) {
        load_words(src, row_words * rows, dst);
    } else {
        LOAD_ROWS: for (unsigned r = 0; r < rows; ++r) load_words(src + r * stride, row_words, dst + r * row_words);
    }
}

// rows rows of n floats each (row r at x_off + r * stride words, stride 0: n / 16, packed rows); y is packed. The
// weight (n floats) applies to every row.
// RMS: y = (x * scale) * w with scale = 1 / sqrt(mean(x^2) + eps). L2: scale = 1 / max(sqrt(sum(x^2)), eps).
// Per word the 16 squares are summed by a tree; eight rotating accumulators per row hide the FP add latency.
void rownorm(vword * act, unsigned n, unsigned rows, unsigned x_off, unsigned w_off, unsigned y_off, float eps,
             unsigned flags, unsigned stride, vword * buf_a, vword * buf_b) {
    const unsigned wpr = n / 16U, total = wpr * rows;
    const unsigned live = wpr < 8U ? wpr : 8U;
    const bool l2 = (flags & NORM_L2) != 0, weighted = (flags & NORM_NO_WEIGHT) == 0;
    float scale[MAX_NORM_ROWS];
    load_rows(act + x_off / 16U, wpr, rows, stride == 0 ? wpr : stride, buf_a);
    // Eight interleaved partial sums: word p of a row adds to slot p % 8 (the chain of one slot is 8 words long, so the
    // FP adder may take several cycles). acc is written and read only 8 iterations apart, hence no inter-iteration
    // dependence for the scheduler.
    float acc[8];
#pragma HLS ARRAY_PARTITION variable=acc complete
#pragma HLS DEPENDENCE variable=acc inter false
    for (unsigned k = 0; k < 8; ++k) {
#pragma HLS UNROLL
        acc[k] = 0.0f;
    }
    unsigned pos = 0, row = 0;
    NORM_SUM: for (unsigned i = 0; i < total; ++i) {
#pragma HLS PIPELINE II=1
        const vword w = buf_a[i];
        const unsigned slot = pos & 7U;
        const float next = add_f(pos < 8U ? 0.0f : acc[slot], sumsq16(w));
        acc[slot] = next;
        if (pos == wpr - 1U) {
            float part[8];
#pragma HLS ARRAY_PARTITION variable=part complete
            for (unsigned k = 0; k < 8; ++k) {
#pragma HLS UNROLL
                part[k] = k < live ? acc[(pos - k) & 7U] : 0.0f;  // k-th newest word of the row
            }
            scale[row] = row_scale(part[0], part[1], part[2], part[3], part[4], part[5], part[6], part[7], n, eps, l2);
            pos = 0;
            ++row;
        } else {
            ++pos;
        }
    }
    if (weighted) load_words(act + w_off / 16U, wpr, buf_b);
    pos = 0;
    row = 0;
    // Four lanes per cycle: normalization is elementwise, so the result does not depend on the folding.
    vword y = 0;
    NORM_APPLY: for (unsigned j = 0; j < total * 4U; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=buf_a inter false
        const unsigned i = j >> 2, q = j & 3U;
        const vword x = buf_a[i], g = buf_b[pos];
        const float s = scale[row];
        for (unsigned l = 0; l < 4; ++l) {
#pragma HLS UNROLL
            const unsigned lane = q * 4U + l;
            y.range(lane * 32 + 31, lane * 32) =
                lane_bits(norm_apply_f(word_lane_float(x, lane), word_lane_float(g, lane), s, weighted));
        }
        if (q == 3U) {
            buf_a[i] = y;
            if (pos == wpr - 1U) { pos = 0; ++row; } else { ++pos; }
        }
    }
    store_words(buf_a, total, act + y_off / 16U);
}

// A deferred RMSNORM / L2 norm of one row: x (and the weight) go to the on-chip copies nx / nw, nothing is written to
// act; returns the scale, summed exactly like rownorm (same word sums, slots and row_scale), so a quantizer that
// applies (x * scale) * w (x_source) sees the values rownorm would have stored.
float norm_defer(vword * act, unsigned n, unsigned x_off, unsigned w_off, float eps, unsigned flags, vword * nx,
                 vword * nw) {
    const unsigned wpr = n / 16U;
    const unsigned live = wpr < 8U ? wpr : 8U;
    const bool l2 = (flags & NORM_L2) != 0, weighted = (flags & NORM_NO_WEIGHT) == 0;
    load_words(act + x_off / 16U, wpr, nx);
    if (weighted) load_words(act + w_off / 16U, wpr, nw);
    float acc[8];
#pragma HLS ARRAY_PARTITION variable=acc complete
#pragma HLS DEPENDENCE variable=acc inter false
    for (unsigned k = 0; k < 8; ++k) {
#pragma HLS UNROLL
        acc[k] = 0.0f;
    }
    float scale = 1.0f;
    NORM_DEFER_SUM: for (unsigned pos = 0; pos < wpr; ++pos) {
#pragma HLS PIPELINE II=1
        const unsigned slot = pos & 7U;
        const float next = add_f(pos < 8U ? 0.0f : acc[slot], sumsq16(nx[pos]));
        acc[slot] = next;
        if (pos == wpr - 1U) {
            float part[8];
#pragma HLS ARRAY_PARTITION variable=part complete
            for (unsigned k = 0; k < 8; ++k) {
#pragma HLS UNROLL
                part[k] = k < live ? acc[(pos - k) & 7U] : 0.0f;
            }
            scale = row_scale(part[0], part[1], part[2], part[3], part[4], part[5], part[6], part[7], n, eps, l2);
        }
    }
    return scale;
}

// RoPE of `rows` rows (heads) of ne0 floats, NEOX pairing as ggml (NEOX, MROPE, IMROPE): for ic < n_dims / 2,
// x0 = x[ic], x1 = x[ic + n_dims / 2], y[ic] = x0 * cos - x1 * sin, y[ic + n_dims / 2] = x0 * sin + x1 * cos; the
// floats from n_dims on are copied. cs: n_dims floats, cos and sin of pair ic at 2 ic and 2 ic + 1 (ggml's rope cache,
// computed by the host for the token's position).
constexpr unsigned MAX_ROPE_DIMS = 128;
void rope(vword * act, unsigned n_dims, unsigned x_off, unsigned cs_off, unsigned y_off, unsigned ne0, unsigned rows,
          vword * buf_a, vword * buf_b) {
    const unsigned wpr = ne0 / 16U, half = n_dims / 2U, hw = half / 16U;
    load_words(act + x_off / 16U, wpr * rows, buf_a);
    load_words(act + cs_off / 16U, n_dims / 16U, buf_b);
    float c[MAX_ROPE_DIMS / 2], s[MAX_ROPE_DIMS / 2];
#pragma HLS ARRAY_PARTITION variable=c cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=s cyclic factor=16
    ROPE_CACHE: for (unsigned ic = 0; ic < half; ++ic) {
#pragma HLS PIPELINE II=1
        c[ic] = word_lane_float(buf_b[(2U * ic) / 16U], (2U * ic) % 16U);
        s[ic] = word_lane_float(buf_b[(2U * ic + 1U) / 16U], (2U * ic + 1U) % 16U);
    }
    // One word of pairs (16 x0 lanes and the 16 matching x1 lanes) per iteration, in place.
    ROPE_PAIRS: for (unsigned t = 0; t < rows * hw; ++t) {
#pragma HLS PIPELINE II=4
#pragma HLS DEPENDENCE variable=buf_a inter false
        const unsigned r = t / hw, h = t % hw;
        const unsigned i0 = r * wpr + h, i1 = i0 + hw;
        const vword a = buf_a[i0], b = buf_a[i1];
        vword ya = 0, yb = 0;
        for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
            const float x0 = word_lane_float(a, l), x1 = word_lane_float(b, l);
            const float cs = c[h * 16U + l], sn = s[h * 16U + l];
            set_lane(ya, l, x0 * cs - x1 * sn);
            set_lane(yb, l, x0 * sn + x1 * cs);
        }
        buf_a[i0] = ya;
        buf_a[i1] = yb;
    }
    store_words(buf_a, wpr * rows, act + y_off / 16U);
}

// y = a + b (mul == false) or y = a * b (mul == true)
void add_mul(vword * act, unsigned n, unsigned a_off, unsigned b_off, unsigned y_off, bool mul, vword * buf_a,
             vword * buf_b) {
    const unsigned words = n / 16U;
    load_words(act + a_off / 16U, words, buf_a);
    load_words(act + b_off / 16U, words, buf_b);
    // Four lanes per cycle (the vectors are short): a quarter of the operators of a full-width word.
    vword y = 0;
    ADD_APPLY: for (unsigned j = 0; j < words * 4U; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=buf_a inter false
        const unsigned i = j >> 2, q = j & 3U;
        const vword a = buf_a[i], b = buf_b[i];
        for (unsigned l = 0; l < 4; ++l) {
#pragma HLS UNROLL
            const unsigned lane = q * 4U + l;
            y.range(lane * 32 + 31, lane * 32) =
                lane_bits(add_mul_f(word_lane_float(a, lane), word_lane_float(b, lane), mul));
        }
        if (q == 3U) buf_a[i] = y;
    }
    store_words(buf_a, words, act + y_off / 16U);
}

// y = g / (1 + exp(-g)) * u, four lanes per cycle to keep the number of exp units small.
void silu_mul(vword * act, unsigned n, unsigned g_off, unsigned u_off, unsigned y_off, vword * buf_a,
              vword * buf_b) {
    const unsigned words = n / 16U;
    load_words(act + g_off / 16U, words, buf_a);
    load_words(act + u_off / 16U, words, buf_b);
    vword y = 0;
    SILU_APPLY: for (unsigned j = 0; j < words * 4U; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=buf_a inter false
        const unsigned i = j >> 2, q = j & 3U;
        const vword g = buf_a[i], u = buf_b[i];
        for (unsigned l = 0; l < 4; ++l) {
#pragma HLS UNROLL
            const unsigned lane = q * 4U + l;
            const float gv = word_lane_float(g, lane);
            y.range(lane * 32 + 31, lane * 32) = lane_bits(silu_mul_f(gv, word_lane_float(u, lane)));
        }
        if (q == 3U) buf_a[i] = y;
    }
    store_words(buf_a, words, act + y_off / 16U);
}

// One lane per cycle (one exp, one log): SiLU x / (1 + e^-x), sigmoid 1 / (1 + e^-x), softplus as ggml
// (x > 20 ? x : log(1 + e^x)). Used for the short gate vectors of DeltaNet.
// x may be strided: rows of row_n floats every stride floats (row_n 0: one contiguous vector); y is packed.
void unary(vword * act, unsigned n, unsigned x_off, unsigned kind, unsigned y_off, unsigned row_n, unsigned stride,
           vword * buf_a) {
    const unsigned words = n / 16U;
    if (row_n == 0) load_words(act + x_off / 16U, words, buf_a);
    else load_rows(act + x_off / 16U, row_n / 16U, n / row_n, stride / 16U, buf_a);
    if (kind == UNARY_COPY) {  // ggml_cont of a strided view: the packed rows as they are
        store_words(buf_a, words, act + y_off / 16U);
        return;
    }
    vword y = 0;
    UNARY_APPLY: for (unsigned j = 0; j < n; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=buf_a inter false
        const unsigned i = j >> 4, lane = j & 15U;
        const float x = word_lane_float(buf_a[i], lane);
        set_lane(y, lane, unary_f(kind, x));
        if (lane == 15U) buf_a[i] = y;
    }
    store_words(buf_a, words, act + y_off / 16U);
}

// Depthwise causal conv of ggml_ssm_conv for one token, d_conv = 4: per channel c the window is the stored state
// (3 floats at s_src + 3c) followed by x[c]; y[c] = sum_t window[t] * w[4c + t] (summed from t = 0 like ggml),
// optionally SiLU. The new state (window[1..3]) goes to s_dst, which may equal s_src.
void conv(vword * act, unsigned ch, unsigned x_off, unsigned w_off, unsigned y_off, unsigned s_src, unsigned s_dst,
          unsigned flags, vword * buf_a, vword * buf_b, vword * buf_c) {
    const unsigned groups = ch / 16U;
    load_words(act + s_src / 16U, groups * 3U, buf_a);
    load_words(act + x_off / 16U, groups, buf_b);
    load_words(act + w_off / 16U, groups * 4U, buf_c);
#pragma HLS ALLOCATION function instances=conv_chan limit=4
#pragma HLS ALLOCATION function instances=silu_f limit=4
    CONV_GROUPS: for (unsigned g = 0; g < groups; ++g) {
#pragma HLS PIPELINE II=4
#pragma HLS DEPENDENCE variable=buf_a inter false
#pragma HLS DEPENDENCE variable=buf_b inter false
        float s[48], k[64];
#pragma HLS ARRAY_PARTITION variable=s complete
#pragma HLS ARRAY_PARTITION variable=k complete
        for (unsigned i = 0; i < 3; ++i) {
            const vword sw = buf_a[g * 3U + i];
            for (unsigned l = 0; l < 16; ++l) s[i * 16U + l] = word_lane_float(sw, l);
        }
        for (unsigned i = 0; i < 4; ++i) {
            const vword kw = buf_c[g * 4U + i];
            for (unsigned l = 0; l < 16; ++l) k[i * 16U + l] = word_lane_float(kw, l);
        }
        const vword xw = buf_b[g];
        vword y = 0, ns[3] = {0, 0, 0};
        for (unsigned c = 0; c < 16; ++c) {
            const float x = word_lane_float(xw, c);
            const float in0 = s[c * 3U], in1 = s[c * 3U + 1U], in2 = s[c * 3U + 2U];
            const float sum = conv_chan(in0, in1, in2, x, k[c * 4U], k[c * 4U + 1U], k[c * 4U + 2U], k[c * 4U + 3U]);
            set_lane(y, c, (flags & CONV_SILU) ? silu_f(sum) : sum);
            const float next[3] = {in1, in2, x};
            for (unsigned t = 0; t < 3; ++t) {
                const unsigned f = c * 3U + t;
                set_lane(ns[f / 16U], f % 16U, next[t]);
            }
        }
        buf_b[g] = y;
        for (unsigned i = 0; i < 3; ++i) buf_a[g * 3U + i] = ns[i];
    }
    store_words(buf_b, groups, act + y_off / 16U);
    store_words(buf_a, groups * 3U, act + s_dst / 16U);
}

} // namespace
} // namespace gemvqc_ops
#endif
