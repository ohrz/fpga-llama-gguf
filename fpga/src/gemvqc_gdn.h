// SPDX-License-Identifier: Apache-2.0
// Gated DeltaNet decode step (ggml GATED_DELTA_NET, one token, scalar gate, K = 1) for gemvqc16.
// Per value head h with key head kh = h % H_k (ggml's tiled broadcast) and state rows j = 0..127 (row j holds
// column j of S, as ggml stores it):  x = S_j * exp(g_h);  delta = (v_j - x . k) * beta_h;  S_j' = x + k * delta;
// y_j = (S_j' . q) * scale.  The state lives in HBM: 512-byte row r (global index) lives on state port r % U at
// word (r / U) * 8, so unit u owns the rows j = u (mod U) of every head on its own port. A step computes on chip:
// unit u reads its rows from an input buffer and writes the new rows to an output buffer. Both stages of a unit
// are processes of the GEMV pass of gemvqc16: stage A first fills the input buffer from the src rows (prefetch),
// stage B first writes the output buffer to the dst rows (write-back), then both compute a step if the pass has
// one. So the state traffic and the step run next to the engines. q, k, v, g, beta and y are FP32 vectors in act.
#ifndef FPGA_LLAMA_GGUF_GEMVQC_GDN_H
#define FPGA_LLAMA_GGUF_GEMVQC_GDN_H
#include "gemvqc_ops.h"
#include <hls_stream.h>

#ifndef GEMVQC_GDN_UNITS
#define GEMVQC_GDN_UNITS 4
#endif

namespace gemvqc_gdn {
namespace {

using gemvqc_ops::vword;
using gemvqc_ops::set_lane;
using gemvqc_ops::tree8;
using gemvqw_engine::word_lane_float;

constexpr unsigned U = GEMVQC_GDN_UNITS;
constexpr unsigned S = 128;                // S_k = S_v
constexpr unsigned SW = S / 16;            // words per state row and per head vector
constexpr unsigned UROWS = S / U;          // rows of one head handled by one unit
constexpr unsigned MAX_HEADS = 32;          // the state buffers hold one layer of up to 32 value heads
constexpr unsigned BUF_WORDS = MAX_HEADS * UROWS * SW;  // words per unit and buffer (one port's share of a layer)
#ifndef GEMVQC_GATED_LANES
#define GEMVQC_GATED_LANES 1
#endif
// Lanes of the gated output per clock: one word in 16 clocks keeps up with the step (512 clocks per head and 8 words).
constexpr unsigned GATED_LANES = GEMVQC_GATED_LANES;
static_assert(16 % GATED_LANES == 0, "GATED_LANES must divide a word");
static_assert(16 % U == 0, "state units must divide 16");
#ifndef GEMVQC_GDN_LANES
#define GEMVQC_GDN_LANES 8
#endif
// FP lanes per unit and stage: 16 (one state word per clock) or 8 (one word in two clocks, half the operators).
constexpr unsigned GL = GEMVQC_GDN_LANES, GH = 16 / GL;
static_assert(GL == 8 || GL == 16, "GDN_LANES must be 8 or 16");

// Sum of a word's 16 products in the order of tree16 (= tree8 of the low half + tree8 of the high half): with 8
// lanes the low half's sum is kept from the first clock of the word.
float word_sum(const float * prod, unsigned half, float & lo) {
#pragma HLS INLINE
    const float s = tree8(prod);
    if (GL == 16) return s + tree8(prod + 8);
    if (half == 0) lo = s;
    return lo + s;
}

// State work of one pass, the same for every unit: stage B writes wb_count words of the output buffer from wb_start
// to the rows at wb_word, stage A reads pf_count words from the rows at pf_word into the input buffer from pf_start;
// then a step over hv value heads (hv = 0: none). The command processor never lets the two transfers of one pass
// touch the same rows.
struct gdn_pass {
    uint64_t wb_word;
    unsigned wb_start, wb_count;
    uint64_t pf_word;
    unsigned pf_start, pf_count;
    unsigned hv, hk;
    float scale;
    bool gated;     // y leaves through the gated norm (gdn_gate)
    float rms_eps;
};

// Stage A of one unit: prefetch, then decay the state row, dot with k, emit the decayed words and delta for stage B.
void stage_a(const vword * st_rd, vword * st, gdn_pass pass, const vword * kbuf, const vword * vbuf,
             const float * decay, const float * beta, unsigned unit, hls::stream<vword> & xs,
             hls::stream<float> & deltas) {
    STATE_PF: for (unsigned i = 0; i < pass.pf_count; ++i) {
#pragma HLS PIPELINE II=1
        st[pass.pf_start + i] = st_rd[pass.pf_word + pass.pf_start + i];
    }
    float part[SW];
#pragma HLS ARRAY_PARTITION variable=part complete
    unsigned w = 0, m = 0, h = 0, kh = 0;
    float lo = 0.0f;
    vword s = 0, kw = 0, x = 0;
    const unsigned hk = pass.hk, total = pass.hv * UROWS * SW * GH;
    // GH clocks per state word, GL lanes each; the lanes leave s and kw and enter x through shift registers.
    GDN_A: for (unsigned c = 0; c < total; ++c) {
#pragma HLS PIPELINE II=1 style=frp
        const unsigned half = c % GH;
        if (half == 0) {
            s = st[c / GH];
            kw = kbuf[kh * SW + w];
        }
        const float d = decay[h];
        float prod[GL];
#pragma HLS ARRAY_PARTITION variable=prod complete
        for (unsigned l = 0; l < GL; ++l) {
#pragma HLS UNROLL
            const float xv = word_lane_float(s, l) * d;
            x = (x >> 32) | (vword(gemvqc_ops::lane_bits(xv)) << 480);
            prod[l] = xv * word_lane_float(kw, l);
        }
        if (GH > 1) {
            s >>= 32 * GL;
            kw >>= 32 * GL;
        }
        const float sum = word_sum(prod, half, lo);
        if (half == GH - 1U) {
            part[w] = sum;
            xs.write(x);
            if (w == SW - 1U) {
                const unsigned j = m * U + unit;
                const float vj = word_lane_float(vbuf[h * SW + j / 16U], j % 16U);
                deltas.write((vj - tree8(part)) * beta[h]);
                w = 0;
                if (m == UROWS - 1U) {
                    m = 0;
                    ++h;
                    kh = kh == hk - 1U ? 0U : kh + 1U;
                } else {
                    ++m;
                }
            } else {
                ++w;
            }
        }
    }
}

// Stage B of one unit: write-back, then S' = x + k * delta into the output buffer, y_j = (S' . q) * scale into ys
// (the unit's rows of each head in order).
void stage_b(vword * st_wr, vword * st, gdn_pass pass, const vword * kbuf, const vword * qbuf,
             hls::stream<vword> & xs, hls::stream<float> & deltas, hls::stream<float> & ys) {
    STATE_WB: for (unsigned i = 0; i < pass.wb_count; ++i) {
#pragma HLS PIPELINE II=1
        st_wr[pass.wb_word + pass.wb_start + i] = st[pass.wb_start + i];
    }
    float part[SW];
#pragma HLS ARRAY_PARTITION variable=part complete
    unsigned w = 0, m = 0, kh = 0;
    float delta = 0.0f, lo = 0.0f;
    vword x = 0, kw = 0, qw = 0, out = 0;
    const unsigned hk = pass.hk, total = pass.hv * UROWS * SW * GH;
    const float scale = pass.scale;
    // GH clocks per state word, GL lanes each (shift registers as in stage A).
    GDN_B: for (unsigned c = 0; c < total; ++c) {
#pragma HLS PIPELINE II=1 style=frp
        const unsigned half = c % GH;
        if (half == 0) {
            if (w == 0) delta = deltas.read();
            x = xs.read();
            kw = kbuf[kh * SW + w];
            qw = qbuf[kh * SW + w];
        }
        float prod[GL];
#pragma HLS ARRAY_PARTITION variable=prod complete
        for (unsigned l = 0; l < GL; ++l) {
#pragma HLS UNROLL
            const float u = word_lane_float(x, l) + word_lane_float(kw, l) * delta;
            out = (out >> 32) | (vword(gemvqc_ops::lane_bits(u)) << 480);
            prod[l] = u * word_lane_float(qw, l);
        }
        if (GH > 1) {
            x >>= 32 * GL;
            kw >>= 32 * GL;
            qw >>= 32 * GL;
        }
        const float sum = word_sum(prod, half, lo);
        if (half == GH - 1U) {
            st[c / GH] = out;
            part[w] = sum;
            if (w == SW - 1U) {
                ys.write(tree8(part) * scale);
                w = 0;
                if (m == UROWS - 1U) {
                    m = 0;
                    kh = kh == hk - 1U ? 0U : kh + 1U;
                } else {
                    ++m;
                }
            } else {
                ++w;
            }
        }
    }
}

// Both stages of unit u inside a dataflow region that has the state ports, the buffers in/out, the step inputs
// ka, kb, qb, v, decay, beta, the streams xs/ds/ys and the pass description `pass`.
#define GEMVQC_GDN_UNIT(u)                                                                                        \
    gemvqc_gdn::stage_a(st##u##_rd, in[u], pass, ka[u], v[u], decay[u], beta[u], u, xs[u], ds[u]);              \
    gemvqc_gdn::stage_b(st##u##_wr, out[u], pass, kb[u], qb[u], xs[u], ds[u], ys[u])
static_assert(U == 4, "the GEMV pass of gemvqc16 is written out for four state ports");

// Per-head scales of `heads` rows of 128 floats in buf (8 words each), exactly as the RMSNORM opcode (rownorm) does
// it: sumsq16 per word, row_scale over the 8 word sums (newest first). l2: 1 / max(sqrt(sum), eps), else
// 1 / sqrt(mean + eps).
void head_scales(const vword * buf, unsigned heads, float eps, bool l2, float * scale) {
    float part[SW];
#pragma HLS ARRAY_PARTITION variable=part complete
    GDN_HEAD_SUM: for (unsigned i = 0; i < heads * SW; ++i) {
#pragma HLS PIPELINE II=1
        part[i % SW] = gemvqc_ops::sumsq16(buf[i]);
        if (i % SW == SW - 1U) {
            scale[i / SW] = gemvqc_ops::row_scale(part[7], part[6], part[5], part[4], part[3], part[2], part[1], part[0],
                                                  S, eps, l2);
        }
    }
}
static_assert(SW == 8, "head_scales passes eight word sums to row_scale");

vword scale_word(const vword & w, float s, bool on) {
#pragma HLS INLINE
    vword out = 0;
    for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
        set_lane(out, l, on ? word_lane_float(w, l) * s : word_lane_float(w, l));
    }
    return out;
}

// Step inputs into the units' local copies (before the pass). q_off, k_off: H_k * 128 floats; v_off: H_v * 128
// floats; g_off, beta_off: H_v floats. l2: q and k are L2-normalized per head on the way (ggml_l2_norm with
// l2_eps, the arithmetic of the RMSNORM opcode with flags 3). gates: g and beta are the raw projections; per head
// beta = sigmoid(beta) and g = softplus(g + dt) * a (dt, a: H_v floats), the arithmetic of the UNARY, ADD and MUL
// opcodes.
void gdn_load(const vword * act, unsigned hv, unsigned hk, unsigned q_off, unsigned k_off, unsigned v_off,
              unsigned g_off, unsigned beta_off, bool l2, float l2_eps, bool gates, unsigned dt_off, unsigned a_off,
              vword ka[U][MAX_HEADS * SW],
              vword kb[U][MAX_HEADS * SW], vword qb[U][MAX_HEADS * SW], vword v[U][MAX_HEADS * SW],
              float decay[U][MAX_HEADS], float beta[U][MAX_HEADS], vword * buf_a) {
    const unsigned kwords = hk * SW, vwords = hv * SW;
    constexpr unsigned QK = 64;  // q and k words in buf_a from here, the gate words below
    float hs[2 * MAX_HEADS];
    gemvqc_ops::load_words(act + q_off / 16U, kwords, buf_a + QK);
    gemvqc_ops::load_words(act + k_off / 16U, kwords, buf_a + QK + kwords);
    if (l2) head_scales(buf_a + QK, 2U * hk, l2_eps, true, hs);
    GDN_LOAD_QK: for (unsigned i = 0; i < kwords; ++i) {
#pragma HLS PIPELINE II=1
        const vword q = scale_word(buf_a[QK + i], hs[i / SW], l2);
        const vword k = scale_word(buf_a[QK + kwords + i], hs[hk + i / SW], l2);
        for (unsigned u = 0; u < U; ++u) {
#pragma HLS UNROLL
            qb[u][i] = q;
            ka[u][i] = k;
            kb[u][i] = k;
        }
    }
    GDN_LOAD_V: for (unsigned i = 0; i < vwords; ++i) {
#pragma HLS PIPELINE II=1
        const vword vw = act[v_off / 16U + i];
        for (unsigned u = 0; u < U; ++u) {
#pragma HLS UNROLL
            v[u][i] = vw;
        }
    }
    const unsigned gwords = (hv + 15U) / 16U;
    gemvqc_ops::load_words(act + g_off / 16U, gwords, buf_a);
    gemvqc_ops::load_words(act + beta_off / 16U, gwords, buf_a + gwords);
    if (gates) {
        gemvqc_ops::load_words(act + dt_off / 16U, gwords, buf_a + 2U * gwords);
        gemvqc_ops::load_words(act + a_off / 16U, gwords, buf_a + 3U * gwords);
    }
    GDN_LOAD_GATE: for (unsigned h = 0; h < hv; ++h) {
#pragma HLS PIPELINE II=1
        const float gr = word_lane_float(buf_a[h / 16U], h % 16U), br = word_lane_float(buf_a[gwords + h / 16U], h % 16U);
        const float dt = word_lane_float(buf_a[2U * gwords + h / 16U], h % 16U);
        const float a = word_lane_float(buf_a[3U * gwords + h / 16U], h % 16U);
        const float gv = gates ? gemvqc_ops::add_mul_f(
                                     gemvqc_ops::unary_f(gemvqc_ops::UNARY_SOFTPLUS, gemvqc_ops::add_mul_f(gr, dt, false)), a, true)
                               : gr;
        const float d = hls::expf(gv);
        const float bt = gates ? gemvqc_ops::unary_f(gemvqc_ops::UNARY_SIGMOID, br) : br;
        for (unsigned u = 0; u < U; ++u) {
#pragma HLS UNROLL
            decay[u][h] = d;
            beta[u][h] = bt;
        }
    }
}
static_assert(4 * ((MAX_HEADS + 15) / 16) <= 64, "the gate words must fit below the q/k words in buf_a");

// The step's y as words, head by head, while the step runs: unit u gives rows j = u (mod U) of every head in order,
// so four values per clock fill a word's lanes 4 (j % 16 / 4) + u. gated: instead of y, per head
// (y * scale_h) * w * silu(z) with scale_h = 1 / sqrt(mean(y^2) + rms_eps) and w (128 floats) in wbuf, z (hv * 128
// floats) in zbuf: the RMSNORM opcode with weight, then the SILU_MUL opcode. The squares are summed while the
// values arrive, four per clock, in the order of sumsq16 (tree16 = pairs of four-lane sums), and row_scale gets the
// word sums as head_scales passes them, so scale_h has the bits of the RMSNORM opcode. Every word goes to gy (for
// act) and, with to_x, to gx (the x of the pass's GEMV).
void gdn_gate(hls::stream<float> ys[U], unsigned hv, bool gated, float rms_eps, const vword * zbuf,
              const vword * wbuf, bool to_x, hls::stream<vword> & gy, hls::stream<vword> & gx) {
    vword head[SW];
    float part[SW];
#pragma HLS ARRAY_PARTITION variable=part complete
    GDN_GATE_HEADS: for (unsigned h = 0; h < hv; ++h) {
        vword cur = 0;
        float q4[4];  // four-lane sums of the current word
#pragma HLS ARRAY_PARTITION variable=q4 complete
        GDN_GATE_COLLECT: for (unsigned m = 0; m < UROWS; ++m) {
#pragma HLS PIPELINE II=1
            float sq[U];
#pragma HLS ARRAY_PARTITION variable=sq complete
            for (unsigned u = 0; u < U; ++u) {
#pragma HLS UNROLL
                const float yv = ys[u].read();
                sq[u] = yv * yv;
                cur = (cur >> 32) | (vword(gemvqc_ops::lane_bits(yv)) << 480);
            }
            q4[m % 4U] = (sq[0] + sq[1]) + (sq[2] + sq[3]);
            if (m % 4U == 3U) {
                head[m / 4U] = cur;
                part[m / 4U] = (q4[0] + q4[1]) + (q4[2] + q4[3]);
            }
        }
        const float hs = gemvqc_ops::row_scale(part[7], part[6], part[5], part[4], part[3], part[2], part[1], part[0],
                                               S, rms_eps, false);
        constexpr unsigned GL = GATED_LANES, GQ = 16U / GL;
        // The lanes move through shift registers (no lane selected by a variable index).
        vword raw = 0, x = 0, z = 0, g = 0, out = 0;
        GDN_GATE_APPLY: for (unsigned j = 0; j < SW * GQ; ++j) {
#pragma HLS PIPELINE II=1
            const unsigned i = j / GQ, q = j % GQ;
            if (q == 0) {
                raw = head[i];
                x = raw;
                z = zbuf[h * SW + i];
                g = wbuf[i];
            }
            for (unsigned l = 0; l < GL; ++l) {
#pragma HLS UNROLL
                const float nv = gemvqc_ops::norm_apply_f(word_lane_float(x, l), word_lane_float(g, l), hs, true);
                const float o = gemvqc_ops::silu_mul_f(word_lane_float(z, l), nv);
                out = (out >> 32) | (vword(gemvqc_ops::lane_bits(o)) << 480);
            }
            x >>= 32 * GL;
            z >>= 32 * GL;
            g >>= 32 * GL;
            if (q == GQ - 1U) {
                const vword o = gated ? out : raw;
                gy.write(o);
                if (to_x) gx.write(o);
            }
        }
    }
}

} // namespace
} // namespace gemvqc_gdn
#endif
