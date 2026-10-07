// SPDX-License-Identifier: Apache-2.0
#ifndef FPGA_LLAMA_GGUF_GEMVQC16_H
#define FPGA_LLAMA_GGUF_GEMVQC16_H
#include "gemvqw.h"
#define GEMVQC16_ENGINES 16
// Command list: GEMVQC_CMD_WORDS unsigned words per command. act holds FP32 vectors as 512-bit words (16 lanes,
// lane 0 in bits 31:0); all act offsets count floats and must be multiples of 16, all lengths n as well.
//   GEMV      [1, format (ggml 2 Q4_0, 3 Q4_1, 8 Q8_0, 13 Q5_K, 14 Q6_K), rows (<= 65536), cols (multiple of
//             128, of 256 for Q5_K/Q6_K), w_off lo, w_off hi, x_off, y_off]  x is quantized like ggml's vec_dot
//             type: quantize_row_q8_0 (Q4_0, Q8_0), quantize_row_q8_1 (Q4_1), quantize_row_q8_K (Q5_K, Q6_K).
//             Row r of the matrix lives in engine r % 16 (w<engine>) in the word layout of gemvqw_layout.h;
//             y is written rounded up to a multiple of 16 floats, the padding lanes are zero.
//   RMSNORM   [2, n, x_off, w_off, y_off, eps (FP32 bits), rows, flags]  per row of n floats (rows 0 means 1):
//             y = x / sqrt(mean(x^2) + eps) * w; flags bit0: L2 norm, y = x / max(sqrt(sum(x^2)), eps);
//             flags bit1: no weight. The weight (n floats) applies to every row.
//   ADD       [3, n, a_off, b_off, y_off, 0, 0, 0]                  y = a + b
//   SILU_MUL  [4, n, g_off, u_off, y_off, 0, 0, 0]                  y = g / (1 + exp(-g)) * u
//   MUL       [5, n, a_off, b_off, y_off, 0, 0, 0]                  y = a * b
//   UNARY     [6, n, x_off, kind, y_off, 0, 0, 0]                   kind 1 SiLU, 2 sigmoid, 3 softplus (as ggml)
//   GDN       two slots: [7, H_v | H_k << 16, q_off, k_off, v_off, g_off, beta_off, y_off]
//             + [src_row, dst_row, scale (FP32 bits), 0, 0, 0, 0, 0]: Gated DeltaNet decode step, head size 128,
//             state rows in the state ports (see gemvqc_gdn.h), y = H_v * 128 floats.
//   CONV      [8, channels, x_off, w_off, y_off, s_src, s_dst, flags]  ggml_ssm_conv for one token, d_conv 4:
//             state [3, channels] at s_src, kernel [4, channels] at w_off (channel-major); new state to s_dst;
//             flags bit0: SiLU on y.
// Norm weights and other FP32 constants live in act like any other vector.
#define GEMVQC_CMD_WORDS 8U
#define GEMVQC_OP_GEMV 1U
#define GEMVQC_OP_RMSNORM 2U
#define GEMVQC_OP_ADD 3U
#define GEMVQC_OP_SILU_MUL 4U
#define GEMVQC_OP_MUL 5U
#define GEMVQC_OP_UNARY 6U
#define GEMVQC_OP_GDN 7U
#define GEMVQC_OP_CONV 8U
// Returns 0 when all commands ran, else the 1-based index of the first invalid command (nothing after it runs).
// A launch with n_cmds = 0 writes GEMVQC_FEATURES to the first float of act (older kernels leave it unchanged).
// 0x6B51 marker | GEMV formats: bit 0 Q5_K, bit 1 Q6_K, bit 2 Q4_1
#define GEMVQC_FEATURES (0x6B510001U | (GEMVQ_HAS_Q6_K ? 2U : 0U) | (GEMVQ_HAS_Q4_1 ? 4U : 0U))
extern "C" int gemvqc16(const ap_uint<512> * w0, const ap_uint<512> * w1, const ap_uint<512> * w2,
                        const ap_uint<512> * w3, const ap_uint<512> * w4, const ap_uint<512> * w5,
                        const ap_uint<512> * w6, const ap_uint<512> * w7, const ap_uint<512> * w8,
                        const ap_uint<512> * w9, const ap_uint<512> * w10, const ap_uint<512> * w11,
                        const ap_uint<512> * w12, const ap_uint<512> * w13, const ap_uint<512> * w14,
                        const ap_uint<512> * w15, ap_uint<512> * act, const unsigned * cmds, unsigned n_cmds
                        // State ports: st<u>_rd and st<u>_wr share one AXI port and get the same buffer.
                        , const ap_uint<512> * st0_rd, ap_uint<512> * st0_wr, const ap_uint<512> * st1_rd,
                        ap_uint<512> * st1_wr, const ap_uint<512> * st2_rd, ap_uint<512> * st2_wr,
                        const ap_uint<512> * st3_rd, ap_uint<512> * st3_wr
                        );
#endif
