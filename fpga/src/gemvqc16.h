// SPDX-License-Identifier: Apache-2.0
#ifndef FPGA_LLAMA_GGUF_GEMVQC16_H
#define FPGA_LLAMA_GGUF_GEMVQC16_H
#include "gemvqw.h"
// Engines (one weight port each): 16, or 20 with -DGEMVQC16_ENGINES=20 (ports w16..w19 follow w15).
#ifndef GEMVQC16_ENGINES
#define GEMVQC16_ENGINES 16
#endif
#if GEMVQC16_ENGINES == 20
#define GEMVQC16_EXTRA_W_PARAMS , const ap_uint<512> * w16, const ap_uint<512> * w17, const ap_uint<512> * w18, \
                                const ap_uint<512> * w19
#define GEMVQC16_EXTRA_W_DATA(w) , w[16].data(), w[17].data(), w[18].data(), w[19].data()
#define GEMVQC16_EXTRA_W_PTRS(w) , w[16], w[17], w[18], w[19]
#define GEMVQC16_EXTRA_W_NAMES , w16, w17, w18, w19
#elif GEMVQC16_ENGINES == 16
#define GEMVQC16_EXTRA_W_PARAMS
#define GEMVQC16_EXTRA_W_DATA(w)
#define GEMVQC16_EXTRA_W_PTRS(w)
#define GEMVQC16_EXTRA_W_NAMES
#else
#error "GEMVQC16_ENGINES must be 16 or 20"
#endif
// Command list: GEMVQC_CMD_WORDS unsigned words per command. act holds FP32 vectors as 512-bit words (16 lanes,
// lane 0 in bits 31:0); all act offsets count floats and must be multiples of 16, all lengths n as well.
//   GEMV      [1, format (ggml 2 Q4_0, 3 Q4_1, 8 Q8_0, 13 Q5_K, 14 Q6_K), rows (<= 65536), cols (multiple of
//             128, of 256 for Q5_K/Q6_K), w_off lo, w_off hi, x_off, y_off]  x is quantized like ggml's vec_dot
//             type: quantize_row_q8_0 (Q4_0, Q8_0), quantize_row_q8_1 (Q4_1), quantize_row_q8_K (Q5_K, Q6_K).
//             Row r of the matrix lives in engine r % GEMVQC16_ENGINES (w<engine>) in the word layout of gemvqw_layout.h;
//             y is written rounded up to a multiple of 16 floats, the padding lanes are zero.
//             Format word bits 7:0 the format, bit 8 GEMVQC_GEMV_SWIGLU: this matrix is g of a SwiGLU pair, the
//             next command (same x, same rows) is u; y = silu(g) * u goes to the u command's y_off, g itself is
//             not written.
//             Format word bit 9 GEMVQC_GEMV_CONV: the next command is a CONV on this GEMV's output (x_off = this
//             y_off, channels = rows); it runs in the same pass on each output word, this GEMV's y is not written,
//             and the CONV is consumed. Same arithmetic as the CONV command. Conv state and kernel are read through
//             act_rd (a second read port on the activation buffer).
//             Format word bit 10 GEMVQC_GEMV_ADD: the next command is an ADD of this GEMV's output and another vector
//             r (the residual, rows floats, rows a multiple of 16 and <= 16384); the pass writes y + r to the ADD's
//             y_off, this GEMV's y is not written, and the ADD is consumed. Such a GEMV is the first of its group.
//             At most one of the bits 8 to 10 per GEMV.
//             GEMV group: up to GEMVQC_GROUP consecutive GEMVs with the same x_off, cols and activation type
//             (Q8_0 for Q4_0/Q8_0, Q8_1 for Q4_1, Q8_K for Q5_K/Q6_K) run as one pass: x is quantized once and
//             the engines stream the matrices back to back. A group ends after a GEMV whose y overlaps x, so the
//             result is the one of running the commands one by one.
//   RMSNORM   [2, n, x_off, w_off, y_off, eps (FP32 bits), rows, flags]  per row of n floats (rows 0 means 1):
//             y = x / sqrt(mean(x^2) + eps) * w; flags bit0: L2 norm, y = x / max(sqrt(sum(x^2)), eps);
//             flags bit1: no weight. The weight (n floats) applies to every row. flags bit2 (deferred, rows 1,
//             n <= 16384): y is not written; the row stays on chip, and GEMVs with x_off = y_off and cols = n quantize
//             the normalized values directly, until the next deferred norm (the same values the command would store).
//             flags bits 23:8: row stride in words (0: rows packed); rows are read with it, y is packed.
//   ADD       [3, n, a_off, b_off, y_off, 0, 0, 0]                  y = a + b
//   SILU_MUL  [4, n, g_off, u_off, y_off, 0, 0, 0]                  y = g / (1 + exp(-g)) * u
//   MUL       [5, n, a_off, b_off, y_off, 0, 0, 0]                  y = a * b
//   UNARY     [6, n, x_off, kind, y_off, row, stride, 0]            kind 1 SiLU, 2 sigmoid, 3 softplus (as ggml), 4 copy;
//             row, stride (floats, multiples of 16): x is rows of `row` floats every `stride` floats (row 0: one
//             contiguous vector), y is packed
//   GDN       two slots: [7, H_v | H_k << 16, q_off, k_off, v_off, g_off, beta_off, y_off]
//             + [src_row, dst_row, scale (FP32 bits), flags, l2_eps, rms_eps, w_off, z_off]: Gated DeltaNet decode step, head size
//             128, H_v <= 32, state rows in the state ports (see gemvqc_gdn.h), y = H_v * 128 floats. The step
//             computes on an on-chip copy of the state; the new state is written back to dst_row while the
//             following GEMVs run (at the latest when the list ends). flags bit 0 (GEMVQC_GDN_WITH_GEMV): the next
//             command, a GEMV without the SwiGLU bit, runs in the same pass as the step (e.g. the z projection).
//             If its x or y touches the step's y, the two run one after the other instead.
//             flags bit 1 (GEMVQC_GDN_L2_QK): q and k are raw and are L2-normalized per head on loading (l2_eps, FP32
//             bits; the RMSNORM opcode with flags 3). flags bit 2 (GEMVQC_GDN_GATED): instead of y, y_off receives
//             per head (y * 1/sqrt(mean(y^2) + rms_eps)) * w * silu(z) with w (128 floats) at w_off and z (H_v * 128
//             floats) at z_off, the RMSNORM opcode with weight followed by SILU_MUL(z, .). A gated step with
//             GEMVQC_GDN_WITH_GEMV whose GEMV cannot join the pass is invalid (z would come too late).
//             flags bit 3 (GEMVQC_GDN_GATES): a third slot [dt_off, a_off, 0, ...] follows; g_off and beta_off hold
//             the raw projections, and per head beta = sigmoid(beta), g = softplus(g + dt) * a (dt, a: H_v floats),
//             the UNARY, ADD and MUL opcodes' arithmetic. The GEMV of GEMVQC_GDN_WITH_GEMV then follows that slot.
//   STATE_LOAD [9, H_v, src_row, 0, 0, 0, 0, 0]  starts reading the state of the next GDN step (same H_v and
//             src_row) into the chip while the following GEMVs run. Optional: without it, or if it does not
//             match, the GDN step reads its state itself.
//   CONV      [8, channels, x_off, w_off, y_off, s_src, s_dst, flags]  ggml_ssm_conv for one token, d_conv 4:
//             state [3, channels] at s_src, kernel [4, channels] at w_off (channel-major); new state to s_dst;
//             flags bit0: SiLU on y.
//   ROPE      [10, n_dims, x_off, cs_off, y_off, ne0, rows, 0]  RoPE of `rows` heads of ne0 floats with ggml's NEOX
//             pairing (NEOX, MROPE, IMROPE): pairs (i, i + n_dims / 2) for i < n_dims / 2 (n_dims a multiple of 32,
//             <= 128), rotated by the cos / sin at cs_off (n_dims floats: cos, sin of pair i at 2i, 2i + 1, ggml's
//             rope cache for the token's position, computed by the host); floats from n_dims on are copied.
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
#define GEMVQC_OP_STATE_LOAD 9U
#define GEMVQC_OP_ROPE 10U
#define GEMVQC_GEMV_SWIGLU 0x100U
#define GEMVQC_GROUP 4U
#define GEMVQC_GDN_WITH_GEMV 1U
#define GEMVQC_GDN_L2_QK 2U
#define GEMVQC_GDN_GATED 4U
#define GEMVQC_GDN_GATES 8U
#define GEMVQC_GEMV_CONV 0x200U
#define GEMVQC_GEMV_ADD 0x400U
// Returns 0 when all commands ran, else the 1-based index of the first invalid command (nothing after it runs).
// At most 128 commands per launch (4 KiB); a longer list returns 129 and runs nothing.
// A launch with n_cmds = 0 writes GEMVQC_FEATURES to the first float of act (older kernels leave it unchanged).
// 0x6B51 marker | GEMV formats: bit 0 Q5_K, bit 1 Q6_K, bit 2 Q4_1 | bit 3 GEMV groups and SwiGLU pairs |
// bit 4 STATE_LOAD and GDN on an on-chip state copy | bit 5 GDN step and a GEMV in one pass | bit 6 CONV epilog |
// bit 7 GDN with L2 norm of q/k and gated RMS norm of y | bit 8 residual ADD epilog | bit 9 deferred RMSNORM |
// bit 10 GDN gates from the raw projections | bit 11 strided rows in RMSNORM and UNARY | bit 12 ROPE |
// bit 13 packed Q6_K layout (three super-block metas per word, see gemvqw_layout.h)
#define GEMVQC_FEATURES (0x6B513FF9U | (GEMVQ_HAS_Q6_K ? 2U : 0U) | (GEMVQ_HAS_Q4_1 ? 4U : 0U))
extern "C" int gemvqc16(const ap_uint<512> * w0, const ap_uint<512> * w1, const ap_uint<512> * w2,
                        const ap_uint<512> * w3, const ap_uint<512> * w4, const ap_uint<512> * w5,
                        const ap_uint<512> * w6, const ap_uint<512> * w7, const ap_uint<512> * w8,
                        const ap_uint<512> * w9, const ap_uint<512> * w10, const ap_uint<512> * w11,
                        const ap_uint<512> * w12, const ap_uint<512> * w13, const ap_uint<512> * w14,
                        const ap_uint<512> * w15 GEMVQC16_EXTRA_W_PARAMS, ap_uint<512> * act, const unsigned * cmds,
                        unsigned n_cmds
                        // State ports: st<u>_rd and st<u>_wr share one AXI port and get the same buffer.
                        , const ap_uint<512> * st0_rd, ap_uint<512> * st0_wr, const ap_uint<512> * st1_rd,
                        ap_uint<512> * st1_wr, const ap_uint<512> * st2_rd, ap_uint<512> * st2_wr,
                        const ap_uint<512> * st3_rd, ap_uint<512> * st3_wr
                        // Read-only view of act (same buffer) for the CONV epilog, read while the pass writes act.
                        , const ap_uint<512> * act_rd
                        );
#endif
