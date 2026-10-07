// SPDX-License-Identifier: Apache-2.0
#ifndef FPGA_LLAMA_GGUF_GEMVQW_H
#define FPGA_LLAMA_GGUF_GEMVQW_H
#include <ap_int.h>
// ggml type ids
#define GEMVQ_Q4_0 2U
#define GEMVQ_Q4_1 3U
#define GEMVQ_Q8_0 8U
#define GEMVQ_Q5_K 13U
#define GEMVQ_Q6_K 14U
#define GEMVQ_MAX_COLS 16384U
// Optional GEMV formats of the gemvqc kernels. Q4_1 is left out by default: its per-lane FP32 min term costs
// ~70k LUT over 16 engines for 4 small matrices per token (HLS estimate).
#ifndef GEMVQ_HAS_Q4_1
#define GEMVQ_HAS_Q4_1 0
#endif
#ifndef GEMVQ_HAS_Q6_K
#define GEMVQ_HAS_Q6_K 1
#endif
// Weights in the word-aligned layout of gemvqw_layout.h; cols must be a multiple of 128. x is FP32 read as
// 512-bit words (16 lanes, lane 0 in bits 31:0).
extern "C" int gemvqw(const ap_uint<512> * w, const ap_uint<512> * x, float * y, unsigned rows, unsigned cols,
                      unsigned format, unsigned long long offset_words);
#endif
