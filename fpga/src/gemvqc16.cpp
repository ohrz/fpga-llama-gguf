// SPDX-License-Identifier: Apache-2.0
// 16-engine word-wide Q4_0/Q4_1/Q8_0/Q5_K/Q6_K GEMV plus element-wise ops (RMSNorm/L2 norm, Add, Mul, SiLU*Mul, unary, conv)
// and the Gated DeltaNet step, driven by a command list: one launch executes n_cmds commands in order.
// Row r of a matrix lives in engine r % 16; every engine has its own weight port (own HBM channel) holding its
// rows in the word layout of gemvqw_layout.h from the command's word offset. x and y are FP32 vectors inside
// one activation buffer, so a command may consume the output of an earlier one.
#include "gemvqc16.h"
#include "gemvqw_engine.h"
#include "gemvqc_ops.h"
#include "gemvqc_gdn.h"

namespace {

using namespace gemvqw_engine;

constexpr int E = GEMVQC16_ENGINES;

unsigned engine_rows(unsigned rows, int engine) {
#pragma HLS INLINE
    return rows > unsigned(engine) ? (rows - unsigned(engine) + E - 1U) / E : 0U;
}

void run_engine(const ap_uint<512> * w, uint64_t offset_words, const xq_word * local_xq, const xs_word * local_xs,
                unsigned format, unsigned rows, int engine, unsigned blocks_per_row,
                hls::stream<float> & row_results) {
    const unsigned word_count = engine_rows(rows, engine) * words_per_row(blocks_per_row, format);
    // Shallow FIFO (LUT shift register): the m_axi adapter already buffers the outstanding bursts.
    hls::stream<ap_uint<512>> words;
#pragma HLS STREAM variable=words depth=16
#pragma HLS DATAFLOW
    read_words(w, offset_words, word_count, words);
    compute_words(words, local_xq, local_xs, format, word_count, blocks_per_row, row_results);
}

uint32_t float_bits(float value) {
#pragma HLS INLINE
    union { uint32_t bits; float value; } c;
    c.value = value;
    return c.bits;
}

// Results of one GEMV are collected on chip and copied to act after the dataflow region, so no m_axi port is
// read and written by different processes at the same time.
constexpr unsigned MAX_ROWS = 65536;
constexpr unsigned YWORDS = MAX_ROWS / 16;

// Engine streams -> y in row order, 16 rows per 512-bit word: row r comes from engine r % E and lands in lane
// r % 16, so word w holds engines (w % (E / 16)) * 16 .. + 15. The last word is padded with zeros.
void write_rows(hls::stream<float> (&row_results)[E], ap_uint<512> * y, unsigned rows) {
    WRITE_ROWS: for (unsigned w = 0; w < (rows + 15U) / 16U; ++w) {
#pragma HLS PIPELINE II=1
        const unsigned bank = w % unsigned(E / 16);
        ap_uint<512> word = 0;
        for (unsigned b = 0; b < unsigned(E / 16); ++b) {
#pragma HLS UNROLL
            for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
                if (b == bank && w * 16U + l < rows) {
                    word.range(l * 32 + 31, l * 32) = float_bits(row_results[b * 16U + l].read());
                }
            }
        }
        y[w] = word;
    }
}

#define GEMVQC16_RUN(n) \
    run_engine(w##n, offset_words, local_xq[n], local_xs[n], format, rows, n, blocks_per_row, row_results[n])

void gemv_core(const ap_uint<512> * w0, const ap_uint<512> * w1, const ap_uint<512> * w2, const ap_uint<512> * w3,
               const ap_uint<512> * w4, const ap_uint<512> * w5, const ap_uint<512> * w6, const ap_uint<512> * w7,
               const ap_uint<512> * w8, const ap_uint<512> * w9, const ap_uint<512> * w10,
               const ap_uint<512> * w11, const ap_uint<512> * w12, const ap_uint<512> * w13,
               const ap_uint<512> * w14, const ap_uint<512> * w15, uint64_t offset_words,
               xq_word (*local_xq)[XWORDS], xs_word (*local_xs)[XWORDS], unsigned format, unsigned rows,
               unsigned blocks_per_row, ap_uint<512> * ybuf) {
    hls::stream<float> row_results[E];
#pragma HLS STREAM variable=row_results depth=16
#pragma HLS DATAFLOW
    GEMVQC16_RUN(0);
    GEMVQC16_RUN(1);
    GEMVQC16_RUN(2);
    GEMVQC16_RUN(3);
    GEMVQC16_RUN(4);
    GEMVQC16_RUN(5);
    GEMVQC16_RUN(6);
    GEMVQC16_RUN(7);
    GEMVQC16_RUN(8);
    GEMVQC16_RUN(9);
    GEMVQC16_RUN(10);
    GEMVQC16_RUN(11);
    GEMVQC16_RUN(12);
    GEMVQC16_RUN(13);
    GEMVQC16_RUN(14);
    GEMVQC16_RUN(15);
    write_rows(row_results, ybuf, rows);
}

} // namespace

extern "C" int gemvqc16(const ap_uint<512> * w0, const ap_uint<512> * w1, const ap_uint<512> * w2,
                        const ap_uint<512> * w3, const ap_uint<512> * w4, const ap_uint<512> * w5,
                        const ap_uint<512> * w6, const ap_uint<512> * w7, const ap_uint<512> * w8,
                        const ap_uint<512> * w9, const ap_uint<512> * w10, const ap_uint<512> * w11,
                        const ap_uint<512> * w12, const ap_uint<512> * w13, const ap_uint<512> * w14,
                        const ap_uint<512> * w15, ap_uint<512> * act, const unsigned * cmds, unsigned n_cmds
                        , const ap_uint<512> * st0_rd, ap_uint<512> * st0_wr, const ap_uint<512> * st1_rd,
                        ap_uint<512> * st1_wr, const ap_uint<512> * st2_rd, ap_uint<512> * st2_wr,
                        const ap_uint<512> * st3_rd, ap_uint<512> * st3_wr
                        ) {
#pragma HLS INTERFACE m_axi port=w0 offset=slave bundle=gmem0 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem1 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem2 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w3 offset=slave bundle=gmem3 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w4 offset=slave bundle=gmem4 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w5 offset=slave bundle=gmem5 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w6 offset=slave bundle=gmem6 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w7 offset=slave bundle=gmem7 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w8 offset=slave bundle=gmem8 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w9 offset=slave bundle=gmem9 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w10 offset=slave bundle=gmem10 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w11 offset=slave bundle=gmem11 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w12 offset=slave bundle=gmem12 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w13 offset=slave bundle=gmem13 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w14 offset=slave bundle=gmem14 depth=4096 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=w15 offset=slave bundle=gmem15 depth=4096 max_read_burst_length=64 num_read_outstanding=8
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
#pragma HLS INTERFACE m_axi port=act offset=slave bundle=gmema depth=8192 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=cmds offset=slave bundle=gmemc depth=256 max_read_burst_length=8
#pragma HLS INTERFACE s_axilite port=act bundle=control
#pragma HLS INTERFACE s_axilite port=cmds bundle=control
#pragma HLS INTERFACE s_axilite port=n_cmds bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
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
    xq_word local_xq[E][XWORDS];
    xs_word local_xs[E][XWORDS];
#pragma HLS ARRAY_PARTITION variable=local_xq complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_xs complete dim=1
    gemvqc_ops::vword buf_a[gemvqc_ops::BUF_WORDS];
    gemvqc_ops::vword buf_b[gemvqc_ops::BUF_WORDS];
    gemvqc_ops::vword buf_c[gemvqc_ops::BUF_WORDS];
#pragma HLS BIND_STORAGE variable=buf_a type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=buf_b type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=buf_c type=ram_2p impl=uram
    using namespace gemvqc_gdn;
    gemvqc_ops::vword gdn_ka[U][MAX_HEADS * SW], gdn_kb[U][MAX_HEADS * SW], gdn_qb[U][MAX_HEADS * SW];
    gemvqc_ops::vword gdn_v[U][MAX_HEADS * SW];
    float gdn_decay[U][MAX_HEADS], gdn_beta[U][MAX_HEADS], gdn_y[U][MAX_HEADS * UROWS];
#pragma HLS ARRAY_PARTITION variable=gdn_ka complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_kb complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_qb complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_v complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_decay complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_beta complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_y complete dim=1
#pragma HLS ARRAY_PARTITION variable=gdn_y cyclic factor=4 dim=2
    ap_uint<512> ybuf[YWORDS];
#pragma HLS BIND_STORAGE variable=ybuf type=ram_2p impl=uram
    if (n_cmds == 0) {
        act[0] = ap_uint<512>(GEMVQC_FEATURES);
        return 0;
    }
    unsigned c = 0;
    COMMANDS: while (c < n_cmds) {
        unsigned cmd[GEMVQC_CMD_WORDS];
#pragma HLS ARRAY_PARTITION variable=cmd complete
        for (unsigned i = 0; i < GEMVQC_CMD_WORDS; ++i) {
#pragma HLS PIPELINE II=1
            cmd[i] = cmds[c * GEMVQC_CMD_WORDS + i];
        }
        // Stop at the first invalid command; the return value names it (1-based), 0 means all ran.
        const unsigned opcode = cmd[0];
        const unsigned max_n = gemvqc_ops::BUF_WORDS * 16U;
        if (opcode == GEMVQC_OP_GEMV) {
            const unsigned format = cmd[1], rows = cmd[2], cols = cmd[3];
            const uint64_t offset_words = uint64_t(cmd[4]) | (uint64_t(cmd[5]) << 32);
            const unsigned x_off = cmd[6], y_off = cmd[7];
            const bool q6_k = GEMVQ_HAS_Q6_K && format == GEMVQ_Q6_K, q4_1 = GEMVQ_HAS_Q4_1 && format == GEMVQ_Q4_1;
            const bool k_quant = format == GEMVQ_Q5_K || q6_k;
            if (rows == 0 || rows > MAX_ROWS || cols == 0 || cols > MAX_COLS || (cols % (k_quant ? 256U : 128U)) != 0 ||
                (format != GEMVQ_Q8_0 && format != GEMVQ_Q4_0 && !q4_1 && !k_quant) ||
                (x_off % 16U) != 0 || (y_off % 16U) != 0) {
                return int(c + 1);
            }
            if (k_quant) {
                quantize_x_k<E>(act + x_off / 16U, cols / 256U, local_xq, local_xs);
            } else {
                quantize_x<E>(act + x_off / 16U, cols / 32U, local_xq, local_xs, q4_1);
            }
            gemv_core(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, offset_words, local_xq,
                      local_xs, format, rows, cols / 32U, ybuf);
            gemvqc_ops::store_words(ybuf, (rows + 15U) / 16U, act + y_off / 16U);
        } else if (opcode == GEMVQC_OP_RMSNORM) {
            const unsigned n = cmd[1], x_off = cmd[2], w_off = cmd[3], y_off = cmd[4], flags = cmd[7];
            const unsigned rows = cmd[6] == 0 ? 1U : cmd[6];
            if (n == 0 || (n % 16U) != 0 || rows > gemvqc_ops::MAX_NORM_ROWS || (n / 16U) * rows > gemvqc_ops::BUF_WORDS ||
                (x_off % 16U) != 0 || (w_off % 16U) != 0 || (y_off % 16U) != 0 || flags > 3U) {
                return int(c + 1);
            }
            gemvqc_ops::rownorm(act, n, rows, x_off, w_off, y_off, gemvqw_engine::bits_to_float(cmd[5]), flags,
                                buf_a, buf_b);
        } else if (opcode == GEMVQC_OP_ADD || opcode == GEMVQC_OP_SILU_MUL || opcode == GEMVQC_OP_MUL ||
                   opcode == GEMVQC_OP_UNARY) {
            const unsigned n = cmd[1], a_off = cmd[2], b_off = cmd[3], y_off = cmd[4];
            const bool unary = opcode == GEMVQC_OP_UNARY;  // b is the kind
            if (n == 0 || n > max_n || (n % 16U) != 0 || (a_off % 16U) != 0 || (y_off % 16U) != 0 ||
                (unary ? (b_off == 0 || b_off > gemvqc_ops::UNARY_SOFTPLUS) : (b_off % 16U) != 0)) {
                return int(c + 1);
            }
            if (opcode == GEMVQC_OP_SILU_MUL) {
                gemvqc_ops::silu_mul(act, n, a_off, b_off, y_off, buf_a, buf_b);
            } else if (unary) {
                gemvqc_ops::unary(act, n, a_off, b_off, y_off, buf_a);
            } else {
                gemvqc_ops::add_mul(act, n, a_off, b_off, y_off, opcode == GEMVQC_OP_MUL, buf_a, buf_b);
            }
        } else if (opcode == GEMVQC_OP_CONV) {
            const unsigned ch = cmd[1], x_off = cmd[2], w_off = cmd[3], y_off = cmd[4], s_src = cmd[5],
                           s_dst = cmd[6], flags = cmd[7];
            if (ch == 0 || ch > gemvqc_ops::MAX_CONV_CHANNELS || (ch % 16U) != 0 || (x_off % 16U) != 0 ||
                (w_off % 16U) != 0 || (y_off % 16U) != 0 || (s_src % 16U) != 0 || (s_dst % 16U) != 0 || flags > 1U) {
                return int(c + 1);
            }
            gemvqc_ops::conv(act, ch, x_off, w_off, y_off, s_src, s_dst, flags, buf_a, buf_b, buf_c);
        } else if (opcode == GEMVQC_OP_GDN) {
            const unsigned hv = cmd[1] & 0xffffU, hk = cmd[1] >> 16;
            if (c + 1U >= n_cmds || hv == 0 || hv > MAX_HEADS || hk == 0 || hk > hv || (hv % hk) != 0 ||
                (cmd[2] % 16U) != 0 || (cmd[3] % 16U) != 0 || (cmd[4] % 16U) != 0 || (cmd[5] % 16U) != 0 ||
                (cmd[6] % 16U) != 0 || (cmd[7] % 16U) != 0) {
                return int(c + 1);
            }
            unsigned ext[3];
#pragma HLS ARRAY_PARTITION variable=ext complete
            for (unsigned i = 0; i < 3; ++i) {
#pragma HLS PIPELINE II=1
                ext[i] = cmds[(c + 1U) * GEMVQC_CMD_WORDS + i];
            }
            if ((ext[0] % U) != 0 || (ext[1] % U) != 0) return int(c + 1);
            gdn(act, st0_rd, st0_wr, st1_rd, st1_wr, st2_rd, st2_wr, st3_rd, st3_wr, hv, hk, cmd[2], cmd[3], cmd[4],
                cmd[5], cmd[6], cmd[7], ext[0], ext[1], gemvqw_engine::bits_to_float(ext[2]), gdn_ka, gdn_kb, gdn_qb,
                gdn_v, gdn_decay, gdn_beta, gdn_y, buf_a);
            ++c;  // the second slot
        } else {
            return int(c + 1);
        }
        ++c;
    }
    return 0;
}
