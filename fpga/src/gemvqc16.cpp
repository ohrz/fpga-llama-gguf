// SPDX-License-Identifier: Apache-2.0
// 16-engine word-wide Q4_0/Q4_1/Q8_0/Q5_K/Q6_K GEMV plus element-wise ops (RMSNorm/L2 norm, Add, Mul, SiLU*Mul, unary, conv)
// and the Gated DeltaNet step, driven by a command list: one launch executes n_cmds commands in order.
// Row r of a matrix lives in engine r % 16; every engine has its own weight port (own HBM channel) holding its
// rows in the word layout of gemvqw_layout.h from the command's word offset. x and y are FP32 vectors inside
// one activation buffer, so a command may consume the output of an earlier one. Consecutive GEMVs on the same x
// run as one group (x quantized once, results written straight to act). The GDN units are processes of the same
// pass: it carries a share of the DeltaNet state traffic (prefetch of the next step's state, write-back of the
// last one) and can run a GDN step next to a GEMV. There is one call of the pass and of the quantizer, so their
// hardware exists once.
#include "gemvqc16.h"
#include "gemvqw_engine.h"
#include "gemvqc_ops.h"
#include "gemvqc_gdn.h"
#include "hls_fence.h"

namespace {

using namespace gemvqw_engine;

constexpr int E = GEMVQC16_ENGINES;
static_assert(E == 16 || E == 20, "the activation broadcast chain is written out for 16 and 20 engines");

// One GEMV of a group. format: ggml type; swiglu: g of a SwiGLU pair (not written, the next member is u).
// conv: the rows are channels of a CONV epilog (kernel at c_w, state from c_src to c_dst, SiLU if silu); y_off is
// then the CONV's y. add: y + r (r at r_off, in rbuf during the pass) goes to y_off, the ADD's y. Engine e streams words + (e < rem ? wpr : 0) words (rows / 16 or one row more), so no engine
// multiplies. blocks: column blocks of the matrix (1, or cols / 256 for the column-blocked GEMV of a GDN pass, whose
// wpr then counts the words of all blocks of a row); bwords, bwpr: words and wpr of one column block.
struct gemv_member {
    unsigned format, rows;
    uint64_t w_off;
    unsigned y_off;
    bool swiglu, conv, silu, add;
    unsigned c_w, c_src, c_dst, r_off;
    unsigned words, wpr, rem, blocks, bwords, bwpr;
};

unsigned engine_words(const gemv_member & m, int engine) {
#pragma HLS INLINE
    return m.words + (unsigned(engine) < m.rem ? m.wpr : 0U);
}

gemv_member pick(unsigned i, gemv_member m0, gemv_member m1, gemv_member m2, gemv_member m3) {
#pragma HLS INLINE
    return i == 0 ? m0 : (i == 1 ? m1 : (i == 2 ? m2 : m3));
}
static_assert(GEMVQC_GROUP == 4, "pick() is written out for four group members");

#define GEMVQC_MEMBERS m0, m1, m2, m3
#define GEMVQC_MEMBER_PARAMS gemv_member m0, gemv_member m1, gemv_member m2, gemv_member m3

// The engine's words of all members back to back (one burst per member).
void read_group(const ap_uint<512> * w, GEMVQC_MEMBER_PARAMS, unsigned n, int engine, unsigned blocks_per_row,
                hls::stream<ap_uint<512>> & words) {
    READ_GROUP: for (unsigned i = 0; i < n; ++i) {
        const gemv_member m = pick(i, GEMVQC_MEMBERS);
        read_words(w, m.w_off, engine_words(m, engine), words);
    }
}

// x comes in m0.blocks blocks of per words (see xcopy_forward), each copied into the engine's buffer before the
// block is computed; blocks_per_row counts the 32-column blocks of one column block. A column-blocked GEMV is the
// only member of its pass; its rows sum up in racc from block to block.
void compute_group(hls::stream<ap_uint<512>> & words, hls::stream<xmsg> & xloc, unsigned per, GEMVQC_MEMBER_PARAMS,
                   unsigned n, int engine, unsigned blocks_per_row, hls::stream<float> & row_results) {
    acc_t racc[MAX_BLOCKED_ROWS];
    xq_word local_xq[XWORDS];
    xs_word local_xs[XWORDS];
    const unsigned blocks = n == 0 ? 0U : m0.blocks;
    COMPUTE_BLOCKS: for (unsigned b = 0; b < blocks; ++b) {
        X_LOCAL: for (unsigned i = 0; i < per; ++i) {
#pragma HLS PIPELINE II=1
            const xmsg m = xloc.read();
            local_xq[i] = m.q;
            local_xs[i] = m.s;
        }
        COMPUTE_GROUP: for (unsigned i = 0; i < n; ++i) {
            const gemv_member m = pick(i, GEMVQC_MEMBERS);
            compute_words(words, local_xq, local_xs, m.format, m.bwords + (unsigned(engine) < m.rem ? m.bwpr : 0U),
                          blocks_per_row, row_results, racc, b == 0, b == blocks - 1U);
        }
    }
}

uint32_t float_bits(float value) {
#pragma HLS INLINE
    union { uint32_t bits; float value; } c;
    c.value = value;
    return c.bits;
}

constexpr unsigned MAX_ROWS = 65536;
constexpr unsigned PAIR_ROWS = 16384;  // rows of a SwiGLU pair (on-chip buffer of g)
constexpr unsigned ADD_ROWS = 16384;   // rows of an add member (on-chip buffer of the residual)

#if GEMVQC16_ENGINES == 16
// Output word w of a member: row r comes from engine r % 16 and lands in lane r % 16; lanes past the last row are 0.
gemvqc_ops::vword gather(hls::stream<float> (&row_results)[E], unsigned w, unsigned rows) {
#pragma HLS INLINE
    const unsigned left = rows - w * 16U;  // rows from this word on, at least 1
    gemvqc_ops::vword word = 0;
    for (int l = 15; l >= 0; --l) {
#pragma HLS UNROLL
        const uint32_t bits = unsigned(l) < left ? float_bits(row_results[l].read()) : 0U;
        word = (word << 32) | gemvqc_ops::vword(bits);
    }
    return word;
}
#else
// Output word w of a member: row r = 16 w + l (lane l) comes from engine r % E. Each engine gives at most one row per
// word, and the words are gathered in order, so every engine's rows leave its stream in row order. Lanes past the
// last row are 0.
gemvqc_ops::vword gather(hls::stream<float> (&row_results)[E], unsigned w, unsigned rows) {
#pragma HLS INLINE
    const unsigned left = rows - w * 16U;  // rows from this word on, at least 1
    const unsigned base = (w * 16U) % unsigned(E);  // engine of lane 0
    float v[E];
#pragma HLS ARRAY_PARTITION variable=v complete
    for (unsigned e = 0; e < unsigned(E); ++e) {
#pragma HLS UNROLL
        const unsigned l = e >= base ? e - base : e + unsigned(E) - base;  // lane of engine e in this word
        v[e] = l < 16U && l < left ? row_results[e].read() : 0.0f;
    }
    gemvqc_ops::vword word = 0;
    for (int l = 15; l >= 0; --l) {
#pragma HLS UNROLL
        const unsigned e = base + unsigned(l) < unsigned(E) ? base + unsigned(l) : base + unsigned(l) - unsigned(E);
        const uint32_t bits = unsigned(l) < left ? float_bits(v[e]) : 0U;
        word = (word << 32) | gemvqc_ops::vword(bits);
    }
    return word;
}
#endif

// Conv state and kernel of 16 channels (one output word): the 3 state words (channel c at floats 3c..3c+2) and the
// 4 kernel words (channel c at floats 4c..4c+3) of the CONV command's layout.
struct conv_state {
    gemvqc_ops::vword w[3];
};
struct conv_kernel {
    gemvqc_ops::vword w[4];
};

// The words as one little-endian bit string shifted right by `bits` (< 512); the low `bits` bits of `in` enter
// at the top.
template <int N>
void shift_words(gemvqc_ops::vword (&w)[N], int bits, const gemvqc_ops::vword & in) {
#pragma HLS INLINE
    for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
        const gemvqc_ops::vword next = i + 1 < N ? w[i + 1] : in;
        w[i] = (w[i] >> bits) | (next << (512 - bits));
    }
}

float low_float(uint32_t bits) {
#pragma HLS INLINE
    return gemvqw_engine::bits_to_float(bits);
}

// State and kernel of the conv members through act_rd, member by member like store_rows, in bursts of 16 output
// words, so the epilog never waits on a single read.
void conv_fetch(const gemvqc_ops::vword * act_rd, GEMVQC_MEMBER_PARAMS, unsigned n, hls::stream<conv_state> & sst,
                hls::stream<conv_kernel> & skw) {
    CONV_MEMBERS: for (unsigned i = 0; i < n; ++i) {
        const gemv_member m = pick(i, GEMVQC_MEMBERS);
        if (!m.conv) continue;
        const unsigned groups = m.rows / 16U;
        CONV_FETCH: for (unsigned g0 = 0; g0 < groups; g0 += 16U) {
            const unsigned cnt = groups - g0 < 16U ? groups - g0 : 16U;
            conv_state s;
            FETCH_STATE: for (unsigned j = 0; j < cnt * 3U; ++j) {
#pragma HLS PIPELINE II=1
                s.w[0] = s.w[1];
                s.w[1] = s.w[2];
                s.w[2] = act_rd[m.c_src / 16U + g0 * 3U + j];
                if (j % 3U == 2U) sst.write(s);
            }
            conv_kernel k;
            FETCH_KERNEL: for (unsigned j = 0; j < cnt * 4U; ++j) {
#pragma HLS PIPELINE II=1
                k.w[0] = k.w[1];
                k.w[1] = k.w[2];
                k.w[2] = k.w[3];
                k.w[3] = act_rd[m.c_w / 16U + g0 * 4U + j];
                if (j % 4U == 3U) skw.write(k);
            }
        }
    }
}

// Engine streams -> act, member by member, 16 rows per 512-bit word. Only this process touches act during the
// pass (the quantizer reads x from on-chip copies), so results go straight to their place. g of a SwiGLU pair goes to gbuf; for its u
// the epilog computes silu(g) * u one lane per cycle (silu_mul_f, the arithmetic of the SILU_MUL opcode), which
// keeps up with the engines as long as a row takes at least 16 words. A conv member gets the CONV epilog the same
// way: per channel conv_chan over the stored window and the new value, optionally silu_f (the CONV opcode's
// functions), the new state {in1, in2, x} goes to c_dst, three words written in the first cycles of the next output
// word. The lanes move through shift registers, so no lane is selected by a variable index.
// An add member adds its residual from rbuf four lanes per cycle (add_mul_f, the ADD opcode's arithmetic). The
// residual comes to rbuf first, while the engines still wait for x and their first weights.
// The GDN step's output words (gy, gdn_words of them, to gdn_y) are stored before the members if gdn_first (their
// GEMV needs all of them first), else after them (gy holds a whole step, so the step never waits for the members).
void store_rows(hls::stream<float> (&row_results)[E], gemvqc_ops::vword * act, gemvqc_ops::vword * gbuf,
                gemvqc_ops::vword * rbuf, hls::stream<conv_state> & sst, hls::stream<conv_kernel> & skw,
                GEMVQC_MEMBER_PARAMS, unsigned n, hls::stream<gemvqc_ops::vword> & gy, unsigned gdn_words,
                unsigned gdn_y, bool gdn_first) {
    if (n != 0 && m0.add) {  // an add member is always the first one
        LOAD_RESIDUAL: for (unsigned w = 0; w < m0.rows / 16U; ++w) {
#pragma HLS PIPELINE II=1
            rbuf[w] = act[m0.r_off / 16U + w];
        }
    }
    if (gdn_first) {
        STORE_GDN_FIRST: for (unsigned w = 0; w < gdn_words; ++w) {
#pragma HLS PIPELINE II=1
            act[gdn_y / 16U + w] = gy.read();
        }
    }
    bool pair = false;
    STORE_MEMBERS: for (unsigned i = 0; i < n; ++i) {
        const gemv_member m = pick(i, GEMVQC_MEMBERS);
        const unsigned words = (m.rows + 15U) / 16U;
        gemvqc_ops::vword * y = act + m.y_off / 16U;
        if (m.swiglu) {
            STORE_G: for (unsigned w = 0; w < words; ++w) {
#pragma HLS PIPELINE II=1
                gbuf[w] = gather(row_results, w, m.rows);
            }
        } else if (pair) {
            gemvqc_ops::vword u = 0, g = 0, out = 0;
            STORE_SWIGLU: for (unsigned j = 0; j < words * 16U; ++j) {
#pragma HLS PIPELINE II=1
                const unsigned w = j / 16U, lane = j % 16U;
                if (lane == 0) {
                    u = gather(row_results, w, m.rows);
                    g = gbuf[w];
                }
                const float v = gemvqc_ops::silu_mul_f(word_lane_float(g, 0), word_lane_float(u, 0));
                g >>= 32;
                u >>= 32;
                out = (out >> 32) | (gemvqc_ops::vword(gemvqc_ops::lane_bits(v)) << 480);
                if (lane == 15U) y[w] = out;
            }
        } else if (m.conv) {
            gemvqc_ops::vword u = 0, out = 0;
            conv_state s, ns, ns_prev;
            conv_kernel k;
            for (unsigned i2 = 0; i2 < 3U; ++i2) s.w[i2] = ns.w[i2] = ns_prev.w[i2] = 0;
            for (unsigned i2 = 0; i2 < 4U; ++i2) k.w[i2] = 0;
            gemvqc_ops::vword * st = act + m.c_dst / 16U;
            STORE_CONV: for (unsigned j = 0; j < words * 16U; ++j) {
#pragma HLS PIPELINE II=1
                const unsigned w = j / 16U, lane = j % 16U;
                if (lane == 0) {
                    u = gather(row_results, w, m.rows);
                    s = sst.read();
                    k = skw.read();
                }
                const ap_uint<32> xb = u.range(31, 0), s0 = s.w[0].range(31, 0), s1 = s.w[0].range(63, 32),
                                  s2 = s.w[0].range(95, 64);
                const float sum = gemvqc_ops::conv_chan(low_float(s0), low_float(s1), low_float(s2), low_float(xb),
                                                        low_float(k.w[0].range(31, 0)), low_float(k.w[0].range(63, 32)),
                                                        low_float(k.w[0].range(95, 64)), low_float(k.w[0].range(127, 96)));
                const float yv = m.silu ? gemvqc_ops::silu_f(sum) : sum;
                out = (out >> 32) | (gemvqc_ops::vword(gemvqc_ops::lane_bits(yv)) << 480);
                shift_words(ns.w, 96, gemvqc_ops::vword(ap_uint<96>((xb, s2, s1))));
                u >>= 32;
                shift_words(s.w, 96, gemvqc_ops::vword(0));
                shift_words(k.w, 128, gemvqc_ops::vword(0));
                if (w > 0 && lane < 3U) {
                    st[(w - 1U) * 3U + lane] = lane == 0 ? ns_prev.w[0] : (lane == 1U ? ns_prev.w[1] : ns_prev.w[2]);
                }
                if (lane == 15U) {
                    y[w] = out;
                    ns_prev = ns;
                }
            }
            CONV_LAST_STATE: for (unsigned i2 = 0; i2 < 3U; ++i2) {
#pragma HLS PIPELINE II=1
                st[(words - 1U) * 3U + i2] = i2 == 0 ? ns_prev.w[0] : (i2 == 1U ? ns_prev.w[1] : ns_prev.w[2]);
            }
        } else if (m.add) {
            gemvqc_ops::vword u = 0, r = 0, out = 0;
            STORE_ADD: for (unsigned j = 0; j < words * 4U; ++j) {
#pragma HLS PIPELINE II=1
                const unsigned w = j / 4U, q = j % 4U;
                if (q == 0) {
                    u = gather(row_results, w, m.rows);
                    r = rbuf[w];
                }
                ap_uint<128> sum = 0;
                for (unsigned l = 0; l < 4; ++l) {
#pragma HLS UNROLL
                    const float s = gemvqc_ops::add_mul_f(word_lane_float(u, l), word_lane_float(r, l), false);
                    sum.range(l * 32 + 31, l * 32) = gemvqc_ops::lane_bits(s);
                }
                u >>= 128;
                r >>= 128;
                out = (out >> 128) | (gemvqc_ops::vword(sum) << 384);
                if (q == 3U) y[w] = out;
            }
        } else {
            STORE_Y: for (unsigned w = 0; w < words; ++w) {
#pragma HLS PIPELINE II=1
                y[w] = gather(row_results, w, m.rows);
            }
        }
        pair = m.swiglu;
    }
    if (!gdn_first) {
        STORE_GDN_LAST: for (unsigned w = 0; w < gdn_words; ++w) {
#pragma HLS PIPELINE II=1
            act[gdn_y / 16U + w] = gy.read();
        }
    }
}

#define GEMVQC16_READ(n) read_group(w##n, GEMVQC_MEMBERS, n_members, n, blocks_per_row, words[n])
#define GEMVQC16_COMPUTE(n) \
    compute_group(words[n], xloc[n], xwords, GEMVQC_MEMBERS, n_members, n, blocks_per_row, row_results[n])

// One pass: a GEMV group (n_members may be 0) and, on the state ports, the state work of `pass` (transfers sized by
// the command processor so that they finish no later than the engines, and possibly a GDN step).
// x is quantized inside the pass from on-chip copies (see x_source_onchip; k_quant selects Q8_K, units are its
// blocks or superblocks, xwords the broadcast words per block of x_blocks, x_words the words of x). Only the compute stages wait for the
// quantized x (each copies it from its FIFO); the weight readers start at once, so the first bursts are in flight while x is
// quantized.
void gemv_core(const ap_uint<512> * w0, const ap_uint<512> * w1, const ap_uint<512> * w2, const ap_uint<512> * w3,
               const ap_uint<512> * w4, const ap_uint<512> * w5, const ap_uint<512> * w6, const ap_uint<512> * w7,
               const ap_uint<512> * w8, const ap_uint<512> * w9, const ap_uint<512> * w10,
               const ap_uint<512> * w11, const ap_uint<512> * w12, const ap_uint<512> * w13,
               const ap_uint<512> * w14, const ap_uint<512> * w15 GEMVQC16_EXTRA_W_PARAMS, const gemvqc_ops::vword * nx,
               const gemvqc_ops::vword * nw, const gemvqc_ops::vword * xbuf, float scale, bool weighted,
               bool from_norm, bool k_quant, bool q8_1, unsigned units, unsigned x_blocks, unsigned xwords, unsigned x_words,
               GEMVQC_MEMBER_PARAMS, unsigned n_members, unsigned blocks_per_row,
               gemvqc_ops::vword * act, gemvqc_ops::vword * gbuf, gemvqc_ops::vword * rbuf
               , const gemvqc_ops::vword * st0_rd, gemvqc_ops::vword * st0_wr, const gemvqc_ops::vword * st1_rd,
               gemvqc_ops::vword * st1_wr, const gemvqc_ops::vword * st2_rd, gemvqc_ops::vword * st2_wr,
               const gemvqc_ops::vword * st3_rd, gemvqc_ops::vword * st3_wr,
               gemvqc_ops::vword in[gemvqc_gdn::U][gemvqc_gdn::BUF_WORDS],
               gemvqc_ops::vword out[gemvqc_gdn::U][gemvqc_gdn::BUF_WORDS],
               const gemvqc_ops::vword ka[gemvqc_gdn::U][gemvqc_gdn::MAX_HEADS * gemvqc_gdn::SW],
               const gemvqc_ops::vword kb[gemvqc_gdn::U][gemvqc_gdn::MAX_HEADS * gemvqc_gdn::SW],
               const gemvqc_ops::vword qb[gemvqc_gdn::U][gemvqc_gdn::MAX_HEADS * gemvqc_gdn::SW],
               const gemvqc_ops::vword v[gemvqc_gdn::U][gemvqc_gdn::MAX_HEADS * gemvqc_gdn::SW],
               const float decay[gemvqc_gdn::U][gemvqc_gdn::MAX_HEADS],
               const float beta[gemvqc_gdn::U][gemvqc_gdn::MAX_HEADS], gemvqc_gdn::gdn_pass pass,
               const gemvqc_ops::vword * zbuf, const gemvqc_ops::vword * wbuf, unsigned gdn_y, bool x_from_gdn
               , const gemvqc_ops::vword * act_rd) {
    hls::stream<ap_uint<512>> xin;
    hls::stream<xmsg> link[E], xloc[E];
#pragma HLS STREAM variable=xin depth=4
#pragma HLS STREAM variable=link depth=2
#pragma HLS STREAM variable=xloc depth=2
    // Shallow FIFOs (LUT shift registers): the m_axi adapters buffer the outstanding bursts.
    hls::stream<ap_uint<512>> words[E];
#pragma HLS STREAM variable=words depth=16
    hls::stream<float> row_results[E];
#pragma HLS STREAM variable=row_results depth=16
    hls::stream<conv_state> sst;
    hls::stream<conv_kernel> skw;
#pragma HLS STREAM variable=sst depth=32
#pragma HLS STREAM variable=skw depth=32
    hls::stream<gemvqc_ops::vword> xs[gemvqc_gdn::U];
    hls::stream<float> ds[gemvqc_gdn::U];
    hls::stream<float> ys[gemvqc_gdn::U];
#pragma HLS STREAM variable=xs depth=64
#pragma HLS STREAM variable=ds depth=16
#pragma HLS STREAM variable=ys depth=32
    // The step's output: gy to act (a whole step deep, see store_rows), gx to the quantizer (x_from_gdn).
    hls::stream<gemvqc_ops::vword> gy, gx;
#pragma HLS STREAM variable=gy depth=256
#pragma HLS STREAM variable=gx depth=16
    static_assert(gemvqc_gdn::MAX_HEADS * gemvqc_gdn::SW <= 256, "gy holds the output of a whole step");
#pragma HLS DATAFLOW
    GEMVQC_GDN_UNIT(0);
    GEMVQC_GDN_UNIT(1);
    GEMVQC_GDN_UNIT(2);
    GEMVQC_GDN_UNIT(3);
    gemvqc_gdn::gdn_gate(ys, pass.hv, pass.gated, pass.rms_eps, zbuf, wbuf, x_from_gdn, gy, gx);
    x_source_onchip(nx, nw, xbuf, scale, weighted, from_norm, x_from_gdn, x_words, gx, xin);
    quantize_any_stream(xin, k_quant, units, link[0], q8_1);
#if GEMVQC16_ENGINES == 20
    XCOPY_CHAIN20(x_blocks * xwords);
#else
    XCOPY_CHAIN(x_blocks * xwords);
#endif
    GEMVQC16_READ(0);
    GEMVQC16_READ(1);
    GEMVQC16_READ(2);
    GEMVQC16_READ(3);
    GEMVQC16_READ(4);
    GEMVQC16_READ(5);
    GEMVQC16_READ(6);
    GEMVQC16_READ(7);
    GEMVQC16_READ(8);
    GEMVQC16_READ(9);
    GEMVQC16_READ(10);
    GEMVQC16_READ(11);
    GEMVQC16_READ(12);
    GEMVQC16_READ(13);
    GEMVQC16_READ(14);
    GEMVQC16_READ(15);
#if GEMVQC16_ENGINES == 20
    GEMVQC16_READ(16);
    GEMVQC16_READ(17);
    GEMVQC16_READ(18);
    GEMVQC16_READ(19);
#endif
    GEMVQC16_COMPUTE(0);
    GEMVQC16_COMPUTE(1);
    GEMVQC16_COMPUTE(2);
    GEMVQC16_COMPUTE(3);
    GEMVQC16_COMPUTE(4);
    GEMVQC16_COMPUTE(5);
    GEMVQC16_COMPUTE(6);
    GEMVQC16_COMPUTE(7);
    GEMVQC16_COMPUTE(8);
    GEMVQC16_COMPUTE(9);
    GEMVQC16_COMPUTE(10);
    GEMVQC16_COMPUTE(11);
    GEMVQC16_COMPUTE(12);
    GEMVQC16_COMPUTE(13);
    GEMVQC16_COMPUTE(14);
    GEMVQC16_COMPUTE(15);
#if GEMVQC16_ENGINES == 20
    GEMVQC16_COMPUTE(16);
    GEMVQC16_COMPUTE(17);
    GEMVQC16_COMPUTE(18);
    GEMVQC16_COMPUTE(19);
#endif
    conv_fetch(act_rd, GEMVQC_MEMBERS, n_members, sst, skw);
    store_rows(row_results, act, gbuf, rbuf, sst, skw, GEMVQC_MEMBERS, n_members, gy, pass.hv * gemvqc_gdn::SW, gdn_y,
               x_from_gdn);
}

constexpr unsigned MAX_CMDS = 128;  // 4 KiB command buffer

// Command i (zeros past the list) from the on-chip copy of the list: one cycle, the copy is split into 8 banks.
void read_cmd(const unsigned cbuf[MAX_CMDS * GEMVQC_CMD_WORDS], unsigned i, unsigned n_cmds,
              unsigned out[GEMVQC_CMD_WORDS]) {
#pragma HLS INLINE
    for (unsigned k = 0; k < GEMVQC_CMD_WORDS; ++k) {
#pragma HLS UNROLL
        out[k] = i < n_cmds ? cbuf[i * GEMVQC_CMD_WORDS + k] : 0U;
    }
}

// Activation type a GEMV format needs: 0 Q8_0 (Q4_0, Q8_0), 1 Q8_1 (Q4_1), 2 Q8_K (Q5_K, Q6_K).
unsigned act_type(unsigned format) {
#pragma HLS INLINE
    return format == GEMVQ_Q5_K || format == GEMVQ_Q6_K ? 2U : (format == GEMVQ_Q4_1 ? 1U : 0U);
}

bool gemv_valid(const unsigned cmd[GEMVQC_CMD_WORDS]) {
#pragma HLS INLINE
    const unsigned format = cmd[1] & 0xffU, rows = cmd[2], cols = cmd[3];
    const bool q6_k = GEMVQ_HAS_Q6_K && format == GEMVQ_Q6_K, q4_1 = GEMVQ_HAS_Q4_1 && format == GEMVQ_Q4_1;
    const bool k_quant = format == GEMVQ_Q5_K || q6_k;
    const unsigned flags = cmd[1] & ~0xffU;
    return cmd[0] == GEMVQC_OP_GEMV && (flags & ~(GEMVQC_GEMV_SWIGLU | GEMVQC_GEMV_CONV | GEMVQC_GEMV_ADD)) == 0 &&
           (flags & (flags - 1U)) == 0 && rows != 0 &&
           rows <= MAX_ROWS && cols != 0 && cols <= MAX_COLS && (cols % (k_quant ? 256U : 128U)) == 0 &&
           (format == GEMVQ_Q8_0 || format == GEMVQ_Q4_0 || q4_1 || k_quant) && (cmd[6] % 16U) == 0 &&
           (cmd[7] % 16U) == 0;
}

// y of m (rounded up to whole words, plus the new conv state of a conv member) touches x: later GEMVs on x must not
// share the quantized copy.
bool overlaps_x(const gemv_member & m, unsigned x_off, unsigned cols) {
#pragma HLS INLINE
    return (m.y_off < x_off + cols && x_off < m.y_off + (m.rows + 15U) / 16U * 16U) ||
           (m.conv && m.c_dst < x_off + cols && x_off < m.c_dst + 3U * m.rows);
}

// The ADD command ad behind add member m: one operand is m's output, the other the residual; fills m's add fields.
bool attach_add(gemv_member & m, const unsigned ad[GEMVQC_CMD_WORDS]) {
#pragma HLS INLINE
    const bool ok = ad[0] == GEMVQC_OP_ADD && ad[1] == m.rows && (ad[2] == m.y_off || ad[3] == m.y_off) &&
                    (m.rows % 16U) == 0 && m.rows <= ADD_ROWS && (ad[2] % 16U) == 0 && (ad[3] % 16U) == 0 &&
                    (ad[4] % 16U) == 0;
    m.r_off = ad[2] == m.y_off ? ad[3] : ad[2];
    m.y_off = ad[4];
    return ok;
}

// The CONV command cv behind conv member m: it must take m's output (all rows as channels); fills m's conv fields.
bool attach_conv(gemv_member & m, const unsigned cv[GEMVQC_CMD_WORDS]) {
#pragma HLS INLINE
    const bool ok = cv[0] == GEMVQC_OP_CONV && cv[1] == m.rows && cv[2] == m.y_off && (cv[1] % 16U) == 0 &&
                    cv[1] <= gemvqc_ops::MAX_CONV_CHANNELS && (cv[3] % 16U) == 0 && (cv[4] % 16U) == 0 &&
                    (cv[5] % 16U) == 0 && (cv[6] % 16U) == 0 && cv[7] <= 1U;
    m.c_w = cv[3];
    m.y_off = cv[4];
    m.c_src = cv[5];
    m.c_dst = cv[6];
    m.silu = (cv[7] & gemvqc_ops::CONV_SILU) != 0;
    return ok;
}

gemv_member member_of(const unsigned cmd[GEMVQC_CMD_WORDS]) {
#pragma HLS INLINE
    gemv_member m;
    m.format = cmd[1] & 0xffU;
    m.rows = cmd[2];
    m.w_off = uint64_t(cmd[4]) | (uint64_t(cmd[5]) << 32);
    m.y_off = cmd[7];
    m.swiglu = (cmd[1] & GEMVQC_GEMV_SWIGLU) != 0;
    m.conv = (cmd[1] & GEMVQC_GEMV_CONV) != 0;
    m.add = (cmd[1] & GEMVQC_GEMV_ADD) != 0;
    m.silu = false;
    m.c_w = m.c_src = m.c_dst = m.r_off = 0;
    m.wpr = words_per_row(cmd[3] / 32U, m.format);
    m.words = (m.rows / unsigned(E)) * m.wpr;
    m.rem = m.rows % unsigned(E);
    m.blocks = 1;
    m.bwords = m.words;
    m.bwpr = m.wpr;
    return m;
}

// Cycle trace (TRACE opcode): the command processor sends an event at every iteration start, pass start and pass
// end and at the end of the list; trace_clock, a process next to it, stamps each event with the cycle it arrives
// in. The processor only writes to the stream and never waits for the clock (the C simulation runs the two
// processes one after the other, so there the stamps number the events).
constexpr unsigned TRACE_ITER = 0, TRACE_PASS_START = 1, TRACE_PASS_END = 2, TRACE_END = 3;

ap_uint<32> trace_tag(unsigned c, unsigned kind) {
#pragma HLS INLINE
    return ap_uint<32>((c & 0xffffU) | (kind << 16));
}

void trace_clock(hls::stream<ap_uint<32>> & ev, ap_uint<64> log[GEMVQC_TRACE_ENTRIES], unsigned & stored,
                 unsigned & total) {
    ap_uint<32> t = 0;
    unsigned k = 0;
    bool stop = false;
    TRACE_CLOCK: while (!stop) {
#pragma HLS PIPELINE II=1
        ap_uint<32> e;
        if (ev.read_nb(e)) {
            if (k < GEMVQC_TRACE_ENTRIES) log[k] = (e, t);
            ++k;
            stop = e.range(17, 16) == TRACE_END;
        }
        ++t;
    }
    stored = k < GEMVQC_TRACE_ENTRIES ? k : GEMVQC_TRACE_ENTRIES;
    total = k;
}

// The command list (all of gemvqc16 except the feature probe and the length check).
void run_list(const ap_uint<512> * w0, const ap_uint<512> * w1, const ap_uint<512> * w2,
              const ap_uint<512> * w3, const ap_uint<512> * w4, const ap_uint<512> * w5, const ap_uint<512> * w6,
              const ap_uint<512> * w7, const ap_uint<512> * w8, const ap_uint<512> * w9, const ap_uint<512> * w10,
              const ap_uint<512> * w11, const ap_uint<512> * w12, const ap_uint<512> * w13, const ap_uint<512> * w14,
              const ap_uint<512> * w15 GEMVQC16_EXTRA_W_PARAMS, ap_uint<512> * act, const unsigned * cmds,
              unsigned n_cmds, const ap_uint<512> * st0_rd, ap_uint<512> * st0_wr, const ap_uint<512> * st1_rd,
              ap_uint<512> * st1_wr, const ap_uint<512> * st2_rd, ap_uint<512> * st2_wr, const ap_uint<512> * st3_rd,
              ap_uint<512> * st3_wr, const ap_uint<512> * act_rd,
              hls::stream<ap_uint<32>> & ev, int & status_out, bool & trace_on, unsigned & trace_off) {
    gemvqc_ops::vword buf_a[gemvqc_ops::BUF_WORDS];
    gemvqc_ops::vword buf_b[gemvqc_ops::BUF_WORDS];
    gemvqc_ops::vword buf_c[gemvqc_ops::BUF_WORDS];
#pragma HLS BIND_STORAGE variable=buf_a type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=buf_b type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=buf_c type=ram_2p impl=uram
    gemvqc_ops::vword gbuf[PAIR_ROWS / 16];
#pragma HLS BIND_STORAGE variable=gbuf type=ram_2p impl=uram
    gemvqc_ops::vword rbuf[ADD_ROWS / 16];
#pragma HLS BIND_STORAGE variable=rbuf type=ram_2p impl=uram
    // Deferred norm (RMSNORM with NORM_DEFER): its row and weight on chip, its scale; GEMVs on its y read from here.
    gemvqc_ops::vword nx[gemvqc_ops::NX_WORDS], nw[gemvqc_ops::NX_WORDS];
#pragma HLS BIND_STORAGE variable=nx type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=nw type=ram_2p impl=uram
    bool nd_valid = false, nd_weighted = false;
    unsigned nd_y = 0, nd_n = 0;
    float nd_scale = 1.0f;
    using namespace gemvqc_gdn;
    gemvqc_ops::vword gdn_ka[U][MAX_HEADS * SW], gdn_kb[U][MAX_HEADS * SW], gdn_qb[U][MAX_HEADS * SW];
    gemvqc_ops::vword gdn_v[U][MAX_HEADS * SW];
    float gdn_decay[U][MAX_HEADS], gdn_beta[U][MAX_HEADS];
#pragma HLS ARRAY_PARTITION variable=gdn_ka complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_kb complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_qb complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_v complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_decay complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_beta complete dim=1
    // State of one GDN step on chip: gdn_in holds the old rows (port u's share in gdn_in[u]), gdn_out the new ones.
    gemvqc_ops::vword gdn_in[U][BUF_WORDS], gdn_out[U][BUF_WORDS];
#pragma HLS ARRAY_PARTITION variable=gdn_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_out complete dim=1
#pragma HLS BIND_STORAGE variable=gdn_in type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=gdn_out type=ram_2p impl=uram
    // Pending state traffic in words per port. Prefetch: the rows of (pf_src, pf_hv) into gdn_in, pf_done words
    // so far, valid until the next GDN step. Write-back: gdn_out to the rows at wb_dst, wb_done of wb_total words.
    bool pf_valid = false;
    unsigned pf_src = 0, pf_hv = 0, pf_done = 0, wb_dst = 0, wb_rows = 0, wb_total = 0, wb_done = 0;
    // The whole list comes on chip in one burst; every later look at a command (group collection, attached CONV/ADD,
    // the GDN slot and its GEMV) is a one-cycle read instead of an m_axi round trip.
    unsigned cbuf[MAX_CMDS * GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=cbuf cyclic factor=8
    LOAD_CMDS: for (unsigned i = 0; i < n_cmds * GEMVQC_CMD_WORDS; ++i) {
#pragma HLS PIPELINE II=1
        cbuf[i] = cmds[i];
    }
    int status = 0;
    trace_on = false;
    trace_off = 0;
    unsigned c = 0;
    bool done = false;
    // Each iteration runs one command (or a group, or a GDN step with its GEMV); everything that needs the engines
    // or the state ports ends in the one pass at the bottom. The iteration after the last command (or an invalid
    // one) finishes the write-back, so the state is complete in HBM when the launch ends.
    COMMANDS: while (!done) {
        ev.write(trace_tag(c, TRACE_ITER));
        gemv_member mem[GEMVQC_GROUP];
#pragma HLS ARRAY_PARTITION variable=mem complete
        unsigned n = 0, x_off = 0, cols = 128, advance = 1;
        bool run = false;
        gdn_pass pass;
        pass.wb_word = uint64_t(wb_dst / U) * SW;
        pass.wb_start = wb_done;
        pass.wb_count = 0;
        pass.pf_word = uint64_t(pf_src / U) * SW;
        pass.pf_start = pf_done;
        pass.pf_count = 0;
        pass.hv = 0;
        pass.hk = 1;
        pass.scale = 0.0f;
        pass.gated = false;
        pass.rms_eps = 0.0f;
        unsigned step_y = 0, step_dst = 0;
        bool out_gemv = false;  // the pass's GEMV is the column-blocked one on the step's output
        if (c >= n_cmds || status != 0) {
            done = true;
            pass.wb_count = wb_total - wb_done;
            run = pass.wb_count != 0;
        } else {
            unsigned cmd[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=cmd complete
            read_cmd(cbuf, c, n_cmds, cmd);
            // Stop at the first invalid command; the return value names it (1-based), 0 means all ran.
            const unsigned opcode = cmd[0];
            const unsigned max_n = gemvqc_ops::BUF_WORDS * 16U;
            if (opcode == GEMVQC_OP_GEMV) {
                cols = cmd[3];
                x_off = cmd[6];
                mem[0] = member_of(cmd);
                n = 1;
                // A conv member takes the CONV command right behind it; end[k] is the command after member k.
                unsigned end[GEMVQC_GROUP];
#pragma HLS ARRAY_PARTITION variable=end complete
                unsigned next_at = c + 1U;
                bool bad_epilog = false;
                if (mem[0].conv) {
                    unsigned cv[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=cv complete
                    read_cmd(cbuf, next_at, n_cmds, cv);
                    bad_epilog = !attach_conv(mem[0], cv);
                    ++next_at;
                } else if (mem[0].add) {
                    unsigned ad[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=ad complete
                    read_cmd(cbuf, next_at, n_cmds, ad);
                    bad_epilog = !attach_add(mem[0], ad);
                    ++next_at;
                }
                end[0] = next_at;
                for (unsigned k = 1; k < GEMVQC_GROUP; ++k) {
                    mem[k] = mem[0];
                    end[k] = next_at;
                }
                // Collect the group: following valid GEMVs on the same x with the same activation type.
                bool open = !bad_epilog && !overlaps_x(mem[0], x_off, cols);
                GROUP: for (unsigned k = 1; k < GEMVQC_GROUP; ++k) {
                    if (!open || next_at >= n_cmds) break;
                    unsigned next[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=next complete
                    read_cmd(cbuf, next_at, n_cmds, next);
                    if (!gemv_valid(next) || next[3] != cols || next[6] != x_off ||
                        act_type(next[1] & 0xffU) != act_type(mem[0].format)) {
                        break;
                    }
                    gemv_member m = member_of(next);
                    if (m.add) break;  // an add member starts its own group (its residual is loaded before the pass)
                    if (m.conv) {
                        if (next_at + 1U >= n_cmds) break;
                        unsigned cv[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=cv complete
                        read_cmd(cbuf, next_at + 1U, n_cmds, cv);
                        if (!attach_conv(m, cv)) break;  // reported when the GEMV comes first
                        next_at += 2U;
                    } else {
                        ++next_at;
                    }
                    mem[k] = m;
                    end[k] = next_at;
                    n = k + 1;
                    open = !overlaps_x(m, x_off, cols);
                }
                // A SwiGLU g needs its u (same rows) right behind it in the group; an unpaired g at the end of a full
                // group starts the next group instead, an unpaired g in front is invalid.
                bool bad_pair = false;
                unsigned i = 0;
                PAIRS: while (i < n) {
                    if (!mem[i].swiglu) {
                        ++i;
                    } else if (i + 1 < n && mem[i + 1].rows == mem[i].rows && !mem[i + 1].swiglu &&
                               !mem[i + 1].conv && !mem[i + 1].add && mem[i].rows <= PAIR_ROWS) {
                        i += 2;
                    } else {
                        bad_pair = i == 0;
                        n = i;
                    }
                }
                if (!gemv_valid(cmd) || bad_pair || bad_epilog) {
                    status = int(c + 1);
                } else {
                    run = true;
                    advance = end[n - 1U] - c;
                    // Transfers get as many words per port as engine 0 (the busiest one) streams, write-back and
                    // prefetch together. The prefetch waits while a pending write-back touches its rows.
                    unsigned budget = 0;
                    for (unsigned k = 0; k < GEMVQC_GROUP; ++k) {
                        if (k < n) budget += engine_words(mem[k], 0);
                    }
                    const unsigned wb_left = wb_total - wb_done;
                    const unsigned pf_left = pf_valid ? pf_hv * UROWS * SW - pf_done : 0U;
                    const bool clash = wb_left != 0 && pf_src < wb_dst + wb_rows && wb_dst < pf_src + pf_hv * S;
                    pass.wb_count = wb_left < budget ? wb_left : budget;
                    pass.pf_count = clash ? 0U : (pf_left < budget - pass.wb_count ? pf_left : budget - pass.wb_count);
                }
            } else if (opcode == GEMVQC_OP_RMSNORM) {
                const unsigned nn = cmd[1], xo = cmd[2], w_off = cmd[3], y_off = cmd[4], flags = cmd[7];
                const unsigned rows = cmd[6] == 0 ? 1U : cmd[6];
                const bool defer = (flags & gemvqc_ops::NORM_DEFER) != 0;
                const unsigned stride = (flags >> 8) & 0xffffU;  // row stride in words, 0: packed rows
                if (nn == 0 || (nn % 16U) != 0 || rows > gemvqc_ops::MAX_NORM_ROWS ||
                    (nn / 16U) * rows > gemvqc_ops::BUF_WORDS || (xo % 16U) != 0 || (w_off % 16U) != 0 ||
                    (y_off % 16U) != 0 || (flags & 0xff0000f8U) != 0 || (stride != 0 && stride < nn / 16U) ||
                    (defer && (rows != 1U || nn > gemvqc_ops::NX_WORDS * 16U))) {
                    status = int(c + 1);
                } else if (defer) {
                    // y is not written: the GEMVs on it normalize on the fly (x_source).
                    nd_scale = gemvqc_ops::norm_defer(act, act_rd, nn, xo, w_off, gemvqw_engine::bits_to_float(cmd[5]), flags, nx, nw);
                    nd_valid = true;
                    nd_weighted = (flags & gemvqc_ops::NORM_NO_WEIGHT) == 0;
                    nd_y = y_off;
                    nd_n = nn;
                } else {
                    gemvqc_ops::rownorm(act, nn, rows, xo, w_off, y_off, gemvqw_engine::bits_to_float(cmd[5]),
                                        flags & 0xffU, stride, buf_a, buf_b);
                }
            } else if (opcode == GEMVQC_OP_ADD || opcode == GEMVQC_OP_SILU_MUL || opcode == GEMVQC_OP_MUL ||
                       opcode == GEMVQC_OP_UNARY) {
                const unsigned nn = cmd[1], a_off = cmd[2], b_off = cmd[3], y_off = cmd[4];
                const bool unary = opcode == GEMVQC_OP_UNARY;  // b is the kind
                const bool silu_mul = opcode == GEMVQC_OP_SILU_MUL;
                // strided x (UNARY) or g (SILU_MUL): rows of row floats every stride floats
                const unsigned row = unary ? cmd[5] : (silu_mul ? cmd[6] : 0U), stride = unary ? cmd[6] : cmd[7];
                if (nn == 0 || nn > max_n || (nn % 16U) != 0 || (a_off % 16U) != 0 || (y_off % 16U) != 0 ||
                    (unary ? (b_off == 0 || b_off > gemvqc_ops::UNARY_COPY) : (b_off % 16U) != 0) ||
                    (silu_mul && (cmd[5] & ~GEMVQC_SILU_MUL_SIGMOID) != 0) ||
                    (row != 0 && ((row % 16U) != 0 || (nn % row) != 0 || (stride % 16U) != 0 || stride < row))) {
                    status = int(c + 1);
                } else if (silu_mul) {
                    gemvqc_ops::silu_mul(act, nn, a_off, b_off, y_off, (cmd[5] & GEMVQC_SILU_MUL_SIGMOID) != 0, row,
                                         stride, buf_a, buf_b);
                } else if (unary) {
                    gemvqc_ops::unary(act, nn, a_off, b_off, y_off, row, stride, buf_a);
                } else {
                    gemvqc_ops::add_mul(act, nn, a_off, b_off, y_off, opcode == GEMVQC_OP_MUL, buf_a, buf_b);
                }
            } else if (opcode == GEMVQC_OP_ROPE) {
                const unsigned nd = cmd[1], xo = cmd[2], cs = cmd[3], y_off = cmd[4], ne0 = cmd[5], rows = cmd[6];
                if (nd == 0 || (nd % 32U) != 0 || nd > gemvqc_ops::MAX_ROPE_DIMS || nd > ne0 || (ne0 % 16U) != 0 ||
                    rows == 0 || (ne0 / 16U) * rows > gemvqc_ops::BUF_WORDS || (xo % 16U) != 0 || (cs % 16U) != 0 ||
                    (y_off % 16U) != 0 || cmd[7] != 0) {
                    status = int(c + 1);
                } else {
                    gemvqc_ops::rope(act, nd, xo, cs, y_off, ne0, rows, buf_a, buf_b);
                }
            } else if (opcode == GEMVQC_OP_TRACE) {
                if ((cmd[1] % 16U) != 0 || (cmd[2] | cmd[3] | cmd[4] | cmd[5] | cmd[6] | cmd[7]) != 0) {
                    status = int(c + 1);
                } else {
                    trace_on = true;
                    trace_off = cmd[1];
                }
            } else if (opcode == GEMVQC_OP_CONV) {
                const unsigned ch = cmd[1], xo = cmd[2], w_off = cmd[3], y_off = cmd[4], s_src = cmd[5],
                               s_dst = cmd[6], flags = cmd[7];
                if (ch == 0 || ch > gemvqc_ops::MAX_CONV_CHANNELS || (ch % 16U) != 0 || (xo % 16U) != 0 ||
                    (w_off % 16U) != 0 || (y_off % 16U) != 0 || (s_src % 16U) != 0 || (s_dst % 16U) != 0 ||
                    flags > 1U) {
                    status = int(c + 1);
                } else {
                    gemvqc_ops::conv(act, ch, xo, w_off, y_off, s_src, s_dst, flags, buf_a, buf_b, buf_c);
                }
            } else if (opcode == GEMVQC_OP_GDN) {
                const unsigned hv = cmd[1] & 0xffffU, hk = cmd[1] >> 16;
                unsigned ext[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=ext complete
                read_cmd(cbuf, c + 1U, n_cmds, ext);
                // With GEMVQC_GDN_GATES a third slot [dt_off, a_off] follows.
                const bool gates = (ext[3] & GEMVQC_GDN_GATES) != 0;
                const unsigned slots = gates ? 3U : 2U;
                unsigned ext2[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=ext2 complete
                read_cmd(cbuf, c + 2U, n_cmds, ext2);
                const unsigned words = hv * UROWS * SW, rows = hv * S;
                // The write-back still pending and this step's prefetch run at the same time in the step's pass;
                // if they touch the same rows, a pass with only the write-back goes first (the step is not consumed).
                const bool wb_pending = wb_done != wb_total;
                const bool clash = wb_pending && ext[0] < wb_dst + wb_rows && wb_dst < ext[0] + rows;
                if (c + slots > n_cmds || hv == 0 || hv > MAX_HEADS || hk == 0 || hk > hv || (hv % hk) != 0 ||
                    (cmd[2] % 16U) != 0 || (cmd[3] % 16U) != 0 || (cmd[4] % 16U) != 0 || (cmd[5] % 16U) != 0 ||
                    (cmd[6] % 16U) != 0 || (cmd[7] % 16U) != 0 || (ext[0] % U) != 0 || (ext[1] % U) != 0 ||
                    ext[3] > (GEMVQC_GDN_WITH_GEMV | GEMVQC_GDN_L2_QK | GEMVQC_GDN_GATED | GEMVQC_GDN_GATES |
                              GEMVQC_GDN_WITH_OUT) ||
                    ((ext[3] & GEMVQC_GDN_GATED) != 0 && ((ext[6] % 16U) != 0 || (ext[7] % 16U) != 0)) ||
                    (gates && ((ext2[0] % 16U) != 0 || (ext2[1] % 16U) != 0))) {
                    status = int(c + 1);
                } else if (clash) {
                    pass.wb_count = wb_total - wb_done;
                    run = true;
                    advance = 0;
                } else {
                    // The rest of the previous write-back (stage B, before it overwrites gdn_out) and of this step's
                    // prefetch (stage A); without a matching STATE_LOAD the whole state is read here.
                    const bool hit = pf_valid && pf_src == ext[0] && pf_hv == hv;
                    pass.wb_count = wb_total - wb_done;
                    pass.pf_word = uint64_t(ext[0] / U) * SW;
                    pass.pf_start = hit ? pf_done : 0U;
                    pass.pf_count = words - pass.pf_start;
                    pass.hv = hv;
                    pass.hk = hk;
                    pass.scale = gemvqw_engine::bits_to_float(ext[2]);
                    pass.gated = (ext[3] & GEMVQC_GDN_GATED) != 0;
                    pass.rms_eps = gemvqw_engine::bits_to_float(ext[5]);
                    step_y = cmd[7];
                    step_dst = ext[1];
                    gdn_load(act, hv, hk, cmd[2], cmd[3], cmd[4], cmd[5], cmd[6], (ext[3] & GEMVQC_GDN_L2_QK) != 0,
                             gemvqw_engine::bits_to_float(ext[4]), gates, ext2[0], ext2[1], gdn_ka, gdn_kb, gdn_qb,
                             gdn_v, gdn_decay, gdn_beta, buf_a);
                    // The gated norm runs in the pass: z and its weight come on chip with the other inputs.
                    if (pass.gated) {
                        gemvqc_ops::load_words(act + ext[7] / 16U, hv * SW, buf_b);
                        gemvqc_ops::load_words(act + ext[6] / 16U, SW, buf_c);
                    }
                    run = true;
                    advance = slots;
                    // The GEMV behind the step joins the pass unless its x or y touches the step's y.
                    if ((ext[3] & GEMVQC_GDN_WITH_GEMV) != 0 && c + slots < n_cmds) {
                        unsigned next[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=next complete
                        read_cmd(cbuf, c + slots, n_cmds, next);
                        const gemv_member m = member_of(next);
                        const unsigned yr = (m.rows + 15U) / 16U * 16U;
                        if (gemv_valid(next) && !m.swiglu && !m.conv && !m.add &&
                            !(next[6] < step_y + rows && step_y < next[6] + next[3]) &&
                            !(m.y_off < step_y + rows && step_y < m.y_off + yr)) {
                            mem[0] = m;
                            for (unsigned k = 1; k < GEMVQC_GROUP; ++k) mem[k] = m;
                            n = 1;
                            cols = next[3];
                            x_off = next[6];
                            advance = slots + 1U;
                        }
                    }
                    // The GEMV on the step's output (and its residual ADD) joins the pass column block by column block.
                    if ((ext[3] & GEMVQC_GDN_WITH_OUT) != 0) {
                        unsigned next[GEMVQC_CMD_WORDS], ad[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=next complete
#pragma HLS ARRAY_PARTITION variable=ad complete
                        read_cmd(cbuf, c + slots, n_cmds, next);
                        read_cmd(cbuf, c + slots + 1U, n_cmds, ad);
                        gemv_member m = member_of(next);
                        bool ok = (ext[3] & GEMVQC_GDN_WITH_GEMV) == 0 && gemv_valid(next) && !m.swiglu && !m.conv &&
                                  next[6] == step_y && next[3] == rows && (rows % 256U) == 0 &&
                                  m.rows <= MAX_BLOCKED_ROWS * unsigned(E);
                        if (m.add) {
                            // The residual is loaded when the pass starts, before the step's y is stored.
                            ok = ok && attach_add(m, ad) && !(m.r_off < step_y + rows && step_y < m.r_off + m.rows);
                        }
                        if (!ok) {
                            status = int(c + 1);
                            run = false;
                        } else {
                            m.blocks = rows / 256U;
                            m.bwpr = words_per_row(8U, m.format);
                            m.bwords = (m.rows / unsigned(E)) * m.bwpr;
                            m.wpr = m.bwpr * m.blocks;
                            m.words = (m.rows / unsigned(E)) * m.wpr;
                            for (unsigned k = 0; k < GEMVQC_GROUP; ++k) mem[k] = m;
                            n = 1;
                            cols = rows;
                            x_off = step_y;
                            out_gemv = true;
                            advance = slots + (m.add ? 2U : 1U);
                        }
                    }
                }
            } else if (opcode == GEMVQC_OP_STATE_LOAD) {
                if (cmd[1] == 0 || cmd[1] > MAX_HEADS || (cmd[2] % U) != 0) {
                    status = int(c + 1);
                } else {
                    pf_valid = true;
                    pf_hv = cmd[1];
                    pf_src = cmd[2];
                    pf_done = 0;
                }
            } else {
                status = int(c + 1);
            }
        }
        if (run) {
            // x for the quantizer in the pass: the deferred norm's row (x is its y), else a copy of x in buf_a.
            // mem is not set in a pass without GEMVs; its format must not reach the quantizer's control flow then
            // (an undefined value stalls the RTL simulation, though not the hardware).
            const unsigned format = n != 0 ? mem[0].format : unsigned(GEMVQ_Q8_0);
            const bool k_quant = format == GEMVQ_Q5_K || format == GEMVQ_Q6_K;
            const bool from_nd = n != 0 && !out_gemv && nd_valid && x_off == nd_y && cols == nd_n;
            if (n != 0 && !from_nd && !out_gemv) gemvqc_ops::load_words(act + x_off / 16U, cols / 16U, buf_a);
            // The fences keep the pass events at the pass boundaries (they have no data dependence on the pass).
            hls::fence(act, ev);
            ev.write(trace_tag(c, TRACE_PASS_START));
            hls::fence(ev, act, w0, st0_rd, st0_wr);
            gemv_core(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15 GEMVQC16_EXTRA_W_NAMES, nx,
                      nw, buf_a, nd_scale,
                      nd_weighted, from_nd, k_quant, GEMVQ_HAS_Q4_1 && format == GEMVQ_Q4_1,
                      n == 0 ? 0U : (k_quant ? cols / 256U : cols / 32U), n == 0 ? 0U : mem[0].blocks,
                      n == 0 ? 0U : (out_gemv ? 2U : cols / 128U),
                      n == 0 ? 0U : cols / 16U,
                      mem[0], mem[1], mem[2], mem[3], n, out_gemv ? 8U : cols / 32U, act, gbuf, rbuf
                      , st0_rd, st0_wr, st1_rd, st1_wr, st2_rd, st2_wr, st3_rd, st3_wr, gdn_in, gdn_out, gdn_ka,
                      gdn_kb, gdn_qb, gdn_v, gdn_decay, gdn_beta, pass, buf_b, buf_c, step_y, out_gemv
                      , act_rd);
            hls::fence(act, w0, st0_rd, st0_wr, ev);
            ev.write(trace_tag(c, TRACE_PASS_END));
            hls::fence(ev, act);
            wb_done += pass.wb_count;
            pf_done += pass.pf_count;
            if (pass.hv != 0) {
                pf_valid = false;
                wb_dst = step_dst;
                wb_rows = pass.hv * S;
                wb_total = pass.hv * UROWS * SW;
                wb_done = 0;
            }
        }
        c += advance;
    }
    hls::fence(act, ev);
    ev.write(trace_tag(c, TRACE_END));
    status_out = status;
}

void run_traced(const ap_uint<512> * w0, const ap_uint<512> * w1, const ap_uint<512> * w2,
                const ap_uint<512> * w3, const ap_uint<512> * w4, const ap_uint<512> * w5, const ap_uint<512> * w6,
                const ap_uint<512> * w7, const ap_uint<512> * w8, const ap_uint<512> * w9, const ap_uint<512> * w10,
                const ap_uint<512> * w11, const ap_uint<512> * w12, const ap_uint<512> * w13, const ap_uint<512> * w14,
                const ap_uint<512> * w15 GEMVQC16_EXTRA_W_PARAMS, ap_uint<512> * act, const unsigned * cmds,
                unsigned n_cmds, const ap_uint<512> * st0_rd, ap_uint<512> * st0_wr, const ap_uint<512> * st1_rd,
                ap_uint<512> * st1_wr, const ap_uint<512> * st2_rd, ap_uint<512> * st2_wr, const ap_uint<512> * st3_rd,
                ap_uint<512> * st3_wr, const ap_uint<512> * act_rd,
                int & status, bool & trace_on, unsigned & trace_off, ap_uint<64> log[GEMVQC_TRACE_ENTRIES],
                unsigned & stored, unsigned & total) {
#pragma HLS DATAFLOW
    hls::stream<ap_uint<32>> ev;
#pragma HLS STREAM variable=ev depth=8
    run_list(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15 GEMVQC16_EXTRA_W_NAMES, act, cmds,
             n_cmds, st0_rd, st0_wr, st1_rd, st1_wr, st2_rd, st2_wr, st3_rd, st3_wr, act_rd, ev, status, trace_on,
             trace_off);
    trace_clock(ev, log, stored, total);
}

// The trace into act at off: header word (stored, total), then 8 events per word, the first in the low bits.
void write_trace(ap_uint<512> * act, unsigned off, const ap_uint<64> log[GEMVQC_TRACE_ENTRIES], unsigned stored,
                 unsigned total) {
    ap_uint<512> head = 0;
    head.range(31, 0) = stored;
    head.range(63, 32) = total;
    act[off / 16U] = head;
    ap_uint<512> w = 0;
    TRACE_WRITE: for (unsigned i = 0; i < (stored + 7U) / 8U * 8U; ++i) {
#pragma HLS PIPELINE II=1
        const ap_uint<64> e = i < stored ? log[i] : ap_uint<64>(0);
        w = (e, ap_uint<448>(w.range(511, 64)));
        if (i % 8U == 7U) act[off / 16U + 1U + i / 8U] = w;
    }
}

} // namespace

extern "C" int gemvqc16(const ap_uint<512> * w0, const ap_uint<512> * w1, const ap_uint<512> * w2,
                        const ap_uint<512> * w3, const ap_uint<512> * w4, const ap_uint<512> * w5,
                        const ap_uint<512> * w6, const ap_uint<512> * w7, const ap_uint<512> * w8,
                        const ap_uint<512> * w9, const ap_uint<512> * w10, const ap_uint<512> * w11,
                        const ap_uint<512> * w12, const ap_uint<512> * w13, const ap_uint<512> * w14,
                        const ap_uint<512> * w15 GEMVQC16_EXTRA_W_PARAMS, ap_uint<512> * act, const unsigned * cmds,
                        unsigned n_cmds
                        , const ap_uint<512> * st0_rd, ap_uint<512> * st0_wr, const ap_uint<512> * st1_rd,
                        ap_uint<512> * st1_wr, const ap_uint<512> * st2_rd, ap_uint<512> * st2_wr,
                        const ap_uint<512> * st3_rd, ap_uint<512> * st3_wr
                        , const ap_uint<512> * act_rd) {
#pragma HLS INTERFACE m_axi port=w0 offset=slave bundle=gmem0 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem1 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem2 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w3 offset=slave bundle=gmem3 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w4 offset=slave bundle=gmem4 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w5 offset=slave bundle=gmem5 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w6 offset=slave bundle=gmem6 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w7 offset=slave bundle=gmem7 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w8 offset=slave bundle=gmem8 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w9 offset=slave bundle=gmem9 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w10 offset=slave bundle=gmem10 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w11 offset=slave bundle=gmem11 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w12 offset=slave bundle=gmem12 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w13 offset=slave bundle=gmem13 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w14 offset=slave bundle=gmem14 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w15 offset=slave bundle=gmem15 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#if GEMVQC16_ENGINES == 20
#pragma HLS INTERFACE m_axi port=w16 offset=slave bundle=gmem16 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w17 offset=slave bundle=gmem17 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w18 offset=slave bundle=gmem18 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w19 offset=slave bundle=gmem19 depth=8192 max_read_burst_length=64 num_read_outstanding=8
#endif
#pragma HLS INTERFACE s_axilite port=w0 bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=w3 bundle=control
#pragma HLS INTERFACE s_axilite port=w4 bundle=control
#pragma HLS INTERFACE s_axilite port=w5 bundle=control
#pragma HLS INTERFACE s_axilite port=w6 bundle=control
#pragma HLS INTERFACE s_axilite port=w7 bundle=control
#pragma HLS INTERFACE s_axilite port=w8 bundle=control
#pragma HLS INTERFACE s_axilite port=w9 bundle=control
#pragma HLS INTERFACE s_axilite port=w10 bundle=control
#pragma HLS INTERFACE s_axilite port=w11 bundle=control
#pragma HLS INTERFACE s_axilite port=w12 bundle=control
#pragma HLS INTERFACE s_axilite port=w13 bundle=control
#pragma HLS INTERFACE s_axilite port=w14 bundle=control
#pragma HLS INTERFACE s_axilite port=w15 bundle=control
#if GEMVQC16_ENGINES == 20
#pragma HLS INTERFACE s_axilite port=w16 bundle=control
#pragma HLS INTERFACE s_axilite port=w17 bundle=control
#pragma HLS INTERFACE s_axilite port=w18 bundle=control
#pragma HLS INTERFACE s_axilite port=w19 bundle=control
#endif
#pragma HLS INTERFACE m_axi port=act offset=slave bundle=gmema depth=8192 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=cmds offset=slave bundle=gmemc depth=1024 max_read_burst_length=8
#pragma HLS INTERFACE s_axilite port=act bundle=control
#pragma HLS INTERFACE s_axilite port=cmds bundle=control
#pragma HLS INTERFACE s_axilite port=n_cmds bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE m_axi port=act_rd offset=slave bundle=gmemr depth=8192 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE s_axilite port=act_rd bundle=control
#pragma HLS INTERFACE m_axi port=st0_rd offset=slave bundle=gmems0 depth=16384 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=st0_wr offset=slave bundle=gmems0 depth=16384 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=st1_rd offset=slave bundle=gmems1 depth=16384 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=st1_wr offset=slave bundle=gmems1 depth=16384 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=st2_rd offset=slave bundle=gmems2 depth=16384 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=st2_wr offset=slave bundle=gmems2 depth=16384 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=st3_rd offset=slave bundle=gmems3 depth=16384 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=st3_wr offset=slave bundle=gmems3 depth=16384 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE s_axilite port=st0_rd bundle=control
#pragma HLS INTERFACE s_axilite port=st0_wr bundle=control
#pragma HLS INTERFACE s_axilite port=st1_rd bundle=control
#pragma HLS INTERFACE s_axilite port=st1_wr bundle=control
#pragma HLS INTERFACE s_axilite port=st2_rd bundle=control
#pragma HLS INTERFACE s_axilite port=st2_wr bundle=control
#pragma HLS INTERFACE s_axilite port=st3_rd bundle=control
#pragma HLS INTERFACE s_axilite port=st3_wr bundle=control
    if (n_cmds == 0) {
        ap_uint<512> features = 0;
        features.range(31, 0) = GEMVQC_FEATURES;
        features.range(63, 32) = GEMVQC_FEATURES2;
        act[0] = features;
        return 0;
    }
    if (n_cmds > MAX_CMDS) return int(MAX_CMDS + 1U);
    ap_uint<64> trace_log[GEMVQC_TRACE_ENTRIES];
    unsigned trace_stored = 0, trace_total = 0, trace_off = 0;
    bool trace_on = false;
    int status = 0;
    run_traced(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15 GEMVQC16_EXTRA_W_NAMES, act, cmds,
               n_cmds, st0_rd, st0_wr, st1_rd, st1_wr, st2_rd, st2_wr, st3_rd, st3_wr, act_rd, status, trace_on,
               trace_off, trace_log, trace_stored, trace_total);
    if (trace_on) write_trace(act, trace_off, trace_log, trace_stored, trace_total);
    return status;
}
