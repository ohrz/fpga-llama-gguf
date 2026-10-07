// SPDX-License-Identifier: Apache-2.0
// Gated DeltaNet decode step (ggml GATED_DELTA_NET, one token, scalar gate, K = 1) for gemvqc16.
// Per value head h with key head kh = h % H_k (ggml's tiled broadcast) and state rows j = 0..127 (row j holds
// column j of S, as ggml stores it):  x = S_j * exp(g_h);  delta = (v_j - x . k) * beta_h;  S_j' = x + k * delta;
// y_j = (S_j' . q) * scale.  The state stays in HBM: 512-byte row r (global index) lives on state port r % U at
// word (r / U) * 8, so unit u streams the rows j = u (mod U) of every head from its own port, reads and writes in
// place (or from src to dst rows), and all units run in parallel. q, k, v, g, beta and y are FP32 vectors in act.
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
using gemvqc_ops::tree16;
using gemvqc_ops::tree8;
using gemvqw_engine::word_lane_float;

constexpr unsigned U = GEMVQC_GDN_UNITS;
constexpr unsigned S = 128;                // S_k = S_v
constexpr unsigned SW = S / 16;            // words per state row and per head vector
constexpr unsigned UROWS = S / U;          // rows of one head handled by one unit
constexpr unsigned MAX_HEADS = 64;
static_assert(16 % U == 0, "state units must divide 16");

// Stage A of one unit: decay the state row, dot with k, emit the decayed words and delta for stage B.
void stage_a(const vword * st, uint64_t first_word, unsigned hv, unsigned hk, const vword * kbuf,
             const vword * vbuf, const float * decay, const float * beta, unsigned unit,
             hls::stream<vword> & xs, hls::stream<float> & deltas) {
    float part[SW];
#pragma HLS ARRAY_PARTITION variable=part complete
    unsigned w = 0, m = 0, h = 0, kh = 0;
    const unsigned total = hv * UROWS * SW;
    GDN_A: for (unsigned t = 0; t < total; ++t) {
#pragma HLS PIPELINE II=1
        const vword s = st[first_word + t];
        const vword kw = kbuf[kh * SW + w];
        const float d = decay[h];
        vword x = 0;
        float prod[16];
#pragma HLS ARRAY_PARTITION variable=prod complete
        for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
            const float xv = word_lane_float(s, l) * d;
            set_lane(x, l, xv);
            prod[l] = xv * word_lane_float(kw, l);
        }
        part[w] = tree16(prod);
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

// Stage B of one unit: S' = x + k * delta written back, y_j = (S' . q) * scale.
void stage_b(vword * st, uint64_t first_word, unsigned hv, unsigned hk, const vword * kbuf, const vword * qbuf,
             float scale, hls::stream<vword> & xs, hls::stream<float> & deltas, float * yloc) {
    float part[SW];
#pragma HLS ARRAY_PARTITION variable=part complete
    unsigned w = 0, row = 0, m = 0, kh = 0;
    float delta = 0.0f;
    const unsigned total = hv * UROWS * SW;
    GDN_B: for (unsigned t = 0; t < total; ++t) {
#pragma HLS PIPELINE II=1
        if (w == 0) delta = deltas.read();
        const vword x = xs.read();
        const vword kw = kbuf[kh * SW + w], qw = qbuf[kh * SW + w];
        vword out = 0;
        float prod[16];
#pragma HLS ARRAY_PARTITION variable=prod complete
        for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
            const float u = word_lane_float(x, l) + word_lane_float(kw, l) * delta;
            set_lane(out, l, u);
            prod[l] = u * word_lane_float(qw, l);
        }
        st[first_word + t] = out;
        part[w] = tree16(prod);
        if (w == SW - 1U) {
            yloc[row] = tree8(part) * scale;
            ++row;
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

#define GEMVQC_GDN_UNIT(u)                                                                                        \
    stage_a(st##u##_rd, src_word, hv, hk, ka[u], v[u], decay[u], beta[u], u, xs[u], ds[u]);                      \
    stage_b(st##u##_wr, dst_word, hv, hk, kb[u], qb[u], scale, xs[u], ds[u], y[u])

void units(const vword * st0_rd, vword * st0_wr, const vword * st1_rd, vword * st1_wr, const vword * st2_rd,
           vword * st2_wr, const vword * st3_rd, vword * st3_wr, uint64_t src_word, uint64_t dst_word, unsigned hv,
           unsigned hk, float scale, vword ka[U][MAX_HEADS * SW], vword kb[U][MAX_HEADS * SW],
           vword qb[U][MAX_HEADS * SW], vword v[U][MAX_HEADS * SW], float decay[U][MAX_HEADS],
           float beta[U][MAX_HEADS], float y[U][MAX_HEADS * UROWS]) {
    hls::stream<vword> xs[U];
    hls::stream<float> ds[U];
#pragma HLS STREAM variable=xs depth=64
#pragma HLS STREAM variable=ds depth=16
#pragma HLS DATAFLOW
    GEMVQC_GDN_UNIT(0);
    GEMVQC_GDN_UNIT(1);
    GEMVQC_GDN_UNIT(2);
    GEMVQC_GDN_UNIT(3);
}
static_assert(U == 4, "units() is written out for four state ports");

// q_off, k_off: H_k * 128 floats; v_off, y_off: H_v * 128 floats; g_off, beta_off: H_v floats.
// src_row, dst_row: global 512-byte state row of head 0, row 0 (multiples of U).
void gdn(vword * act, const vword * st0_rd, vword * st0_wr, const vword * st1_rd, vword * st1_wr,
         const vword * st2_rd, vword * st2_wr, const vword * st3_rd, vword * st3_wr, unsigned hv, unsigned hk,
         unsigned q_off, unsigned k_off, unsigned v_off, unsigned g_off, unsigned beta_off, unsigned y_off,
         unsigned src_row, unsigned dst_row, float scale, vword ka[U][MAX_HEADS * SW], vword kb[U][MAX_HEADS * SW],
         vword qb[U][MAX_HEADS * SW], vword v[U][MAX_HEADS * SW], float decay[U][MAX_HEADS],
         float beta[U][MAX_HEADS], float y[U][MAX_HEADS * UROWS], vword * buf_a) {
    const unsigned kwords = hk * SW, vwords = hv * SW;
    GDN_LOAD_QK: for (unsigned i = 0; i < kwords; ++i) {
#pragma HLS PIPELINE II=1
        const vword q = act[q_off / 16U + i];
        for (unsigned u = 0; u < U; ++u) {
#pragma HLS UNROLL
            qb[u][i] = q;
        }
    }
    GDN_LOAD_K: for (unsigned i = 0; i < kwords; ++i) {
#pragma HLS PIPELINE II=1
        const vword k = act[k_off / 16U + i];
        for (unsigned u = 0; u < U; ++u) {
#pragma HLS UNROLL
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
    GDN_LOAD_GATE: for (unsigned h = 0; h < hv; ++h) {
#pragma HLS PIPELINE II=1
        const float d = hls::expf(word_lane_float(buf_a[h / 16U], h % 16U));
        const float bt = word_lane_float(buf_a[gwords + h / 16U], h % 16U);
        for (unsigned u = 0; u < U; ++u) {
#pragma HLS UNROLL
            decay[u][h] = d;
            beta[u][h] = bt;
        }
    }
    units(st0_rd, st0_wr, st1_rd, st1_wr, st2_rd, st2_wr, st3_rd, st3_wr, uint64_t(src_row / U) * SW,
          uint64_t(dst_row / U) * SW, hv, hk, scale, ka, kb, qb, v, decay, beta, y);
    // y word (h, jw) holds rows j = 16 jw + l; row j came from unit j % U = l % U at local row h * UROWS + j / U.
    GDN_STORE_Y: for (unsigned i = 0; i < vwords; ++i) {
#pragma HLS PIPELINE II=1
        const unsigned h = i / SW, jw = i % SW;
        vword out = 0;
        for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
            set_lane(out, l, y[l % U][h * UROWS + (jw * 16U + l) / U]);
        }
        act[y_off / 16U + i] = out;
    }
}

} // namespace
} // namespace gemvqc_gdn
#endif
