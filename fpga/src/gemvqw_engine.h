// SPDX-License-Identifier: Apache-2.0
// Shared building blocks of the gemvqw kernels: llama.cpp-compatible Q8_0 activation quantization and the
// word-wide Q4_0/Q8_0 engine. One 64-byte weight word per cycle in the layout of gemvqw_layout.h: a scale word,
// then quant words with 2 Q8_0 or 4 Q4_0 blocks each. Exact int block dots, FP32 sum per word, fixed-point
// row sums.
#ifndef FPGA_LLAMA_GGUF_GEMVQW_ENGINE_H
#define FPGA_LLAMA_GGUF_GEMVQW_ENGINE_H
#include "gemvqw.h"
#include "stream_common.h"
#include <cstdint>

namespace gemvqw_engine {
namespace {

constexpr unsigned MAX_COLS = GEMVQ_MAX_COLS;

uint32_t float_to_bits(float value) {
#pragma HLS INLINE
    union { uint32_t bits; float value; } c;
    c.value = value;
    return c.bits;
}

ap_uint<16> fp32_to_fp16(float value) {
#pragma HLS INLINE
    const uint32_t bits = float_to_bits(value);
    const ap_uint<16> sign = ap_uint<16>((bits >> 16) & 0x8000U);
    const uint32_t exponent = (bits >> 23) & 0xffU;
    const uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        return sign | (mantissa == 0 ? ap_uint<16>(0x7c00) : ap_uint<16>(0x7e00));
    }
    int half_exponent = int(exponent) - 127 + 15;
    if (half_exponent >= 31) return sign | ap_uint<16>(0x7c00);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        const uint32_t normalized = mantissa | 0x800000U;
        const int shift = 14 - half_exponent;
        uint32_t rounded = normalized >> shift;
        const uint32_t remainder = normalized & ((uint32_t(1) << shift) - 1U);
        const uint32_t halfway = uint32_t(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U))) ++rounded;
        return sign | ap_uint<16>(rounded);
    }
    uint32_t rounded_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (rounded_mantissa & 1U))) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400U) {
            rounded_mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 31) return sign | ap_uint<16>(0x7c00);
        }
    }
    return sign | (ap_uint<16>(half_exponent) << 10) | ap_uint<16>(rounded_mantissa);
}

// Quantized x: one 1024-bit word per 4 activation blocks (block k at bits 256k..256k+255) and the 4 FP32
// block scales per 128-bit word, so an engine reads all activations of one weight word in one access.
typedef ap_uint<1024> xq_word;
typedef ap_uint<128> xs_word;
constexpr unsigned XWORDS = MAX_COLS / 128;

float word_lane_float(const ap_uint<512> & word, unsigned lane) {
#pragma HLS INLINE
    union { uint32_t bits; float value; } c;
    c.bits = word.range(lane * 32 + 31, lane * 32).to_uint();
    return c.value;
}

// Q8_0-quantize x like llama.cpp's quantize_row_q8_0: scale = amax / 127 stored as FP16, roundf(x / scale).
// x is read as 512-bit words (16 FP32 lanes, lane 0 in bits 31:0), two words per block.
// Writes N identical copies so N engines read in parallel. blocks must be a multiple of 4.
// The xs lane of a block holds the FP16-rounded scale as FP32; with q8_1 (Q8_1 for Q4_1, quantize_row_q8_1_ref)
// it holds the FP16 scale in bits 15:0 and FP16(sum of the quants * scale) in bits 31:16.
template <int N>
void quantize_x(const ap_uint<512> * x, unsigned blocks, xq_word (*local_xq)[XWORDS], xs_word (*local_xs)[XWORDS],
                bool q8_1 = false) {
    xq_word pending_q = 0;
    xs_word pending_s = 0;
    QUANTIZE_X: for (unsigned block = 0; block < blocks; ++block) {
#pragma HLS PIPELINE II=2
        float value[32];
#pragma HLS ARRAY_PARTITION variable=value complete
        float magnitude[32];
#pragma HLS ARRAY_PARTITION variable=magnitude complete
        const ap_uint<512> low = x[block * 2U];
        const ap_uint<512> high = x[block * 2U + 1U];
        for (unsigned j = 0; j < 32; ++j) {
#pragma HLS UNROLL
            value[j] = word_lane_float(j < 16 ? low : high, j & 15U);
            magnitude[j] = value[j] < 0.0f ? -value[j] : value[j];
        }
        for (unsigned width = 16; width >= 1; width >>= 1) {
#pragma HLS UNROLL
            for (unsigned i = 0; i < width; ++i) {
#pragma HLS UNROLL
                magnitude[i] = magnitude[2 * i] > magnitude[2 * i + 1] ? magnitude[2 * i] : magnitude[2 * i + 1];
            }
        }
        const float scale = magnitude[0] / 127.0f;
        const float reciprocal = scale != 0.0f ? 1.0f / scale : 0.0f;
        ap_uint<256> packed = 0;
        int quant_sum = 0;
        for (unsigned j = 0; j < 32; ++j) {
#pragma HLS UNROLL
            // Exact roundf(): truncate, then step away from zero when the remainder is at least one half.
            const float scaled = value[j] * reciprocal;
            const int32_t truncated = int32_t(scaled);
            const float remainder = scaled - float(truncated);
            const int8_t rounded = int8_t(truncated + (remainder >= 0.5f ? 1 : (remainder <= -0.5f ? -1 : 0)));
            packed.range(j * 8 + 7, j * 8) = ap_uint<8>(rounded);
            quant_sum += rounded;
        }
        const ap_uint<16> scale_half = fp32_to_fp16(scale);
        const uint32_t lane = q8_1 ? uint32_t(scale_half) | (uint32_t(fp32_to_fp16(float(quant_sum) * scale)) << 16)
                                   : float_to_bits(half_bits_to_float(scale_half));
        // Shift registers: after four blocks block 0 sits at the lowest position.
        pending_q = (pending_q >> 256) | (xq_word(packed) << 768);
        pending_s = (pending_s >> 32) | (xs_word(lane) << 96);
        if ((block & 3U) == 3U) {
            for (int e = 0; e < N; ++e) {
#pragma HLS UNROLL
                local_xq[e][block / 4U] = pending_q;
                local_xs[e][block / 4U] = pending_s;
            }
        }
    }
}

// Q8_K-quantize x like llama.cpp's quantize_row_q8_K_ref for the K-quant formats: per 256 values the value with
// the largest magnitude (the first one on ties) gives iscale = -127 / max, q = min(127, nearest_int(iscale * x)),
// round half to even through the FP32 addition of 1.5 * 2^23, and d = 1 / iscale (all zero for a zero block).
// local_xq as for Q8_0 (128 activations per word, element e of the word at bits 8e..8e+7); local_xs per word:
// bits 31:0 the super-block's d (FP32), bits 32+16k..47+16k the sum of the 32 quants of block k of the word.
template <int N>
void quantize_x_k(const ap_uint<512> * x, unsigned superblocks, xq_word (*local_xq)[XWORDS],
                  xs_word (*local_xs)[XWORDS]) {
    QXK_SUPER: for (unsigned sb = 0; sb < superblocks; ++sb) {
        ap_uint<512> raw[16];
#pragma HLS ARRAY_PARTITION variable=raw complete
        uint32_t amax = 0;
        float vmax = 0.0f;
        QXK_MAX: for (unsigned w = 0; w < 16; ++w) {
#pragma HLS PIPELINE II=1
            const ap_uint<512> word = x[sb * 16U + w];
            raw[w] = word;
            // |x| as the FP32 bits without the sign: for finite values the integer order is the float order.
            uint32_t mag[16];
            float val[16];
#pragma HLS ARRAY_PARTITION variable=mag complete
#pragma HLS ARRAY_PARTITION variable=val complete
            for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
                mag[l] = word.range(l * 32 + 30, l * 32).to_uint();
                val[l] = word_lane_float(word, l);
            }
            for (unsigned width = 8; width >= 1; width >>= 1) {
#pragma HLS UNROLL
                for (unsigned i = 0; i < width; ++i) {
#pragma HLS UNROLL
                    const bool left = mag[2 * i] >= mag[2 * i + 1];  // the earlier value wins ties
                    val[i] = left ? val[2 * i] : val[2 * i + 1];
                    mag[i] = left ? mag[2 * i] : mag[2 * i + 1];
                }
            }
            if (mag[0] > amax) {
                amax = mag[0];
                vmax = val[0];
            }
        }
        const float iscale = amax == 0 ? 0.0f : -127.0f / vmax;
        const float d = amax == 0 ? 0.0f : 1.0f / iscale;
        xq_word pending_q = 0;
        int block_sum[8];
#pragma HLS ARRAY_PARTITION variable=block_sum complete
        QXK_QUANT: for (unsigned w = 0; w < 16; ++w) {
#pragma HLS PIPELINE II=1
            const ap_uint<512> word = raw[w];
            ap_uint<128> packed = 0;
            int part = 0;
            for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
                const float biased = word_lane_float(word, l) * iscale + 12582912.0f;
                const int v = int(float_to_bits(biased) & 0x7fffffU) - 0x400000;
                const int q = v > 127 ? 127 : v;
                packed.range(l * 8 + 7, l * 8) = ap_uint<8>(q);
                part += q;
            }
            block_sum[w / 2U] = (w & 1U) ? block_sum[w / 2U] + part : part;
            // Shift register: after eight words the super-block half's first value sits at the lowest position.
            pending_q = (pending_q >> 128) | (xq_word(packed) << 896);
            if ((w & 7U) == 7U) {
                const unsigned half = w / 8U;
                xs_word xs = xs_word(float_to_bits(d));
                for (unsigned k = 0; k < 4; ++k) {
#pragma HLS UNROLL
                    xs.range(47 + 16 * k, 32 + 16 * k) = ap_uint<16>(block_sum[4U * half + k]);
                }
                for (int e = 0; e < N; ++e) {
#pragma HLS UNROLL
                    local_xq[e][sb * 2U + half] = pending_q;
                    local_xs[e][sb * 2U + half] = xs;
                }
            }
        }
    }
}

unsigned blocks_per_word(unsigned format) {
#pragma HLS INLINE
    return format == GEMVQ_Q8_0 ? 2U : 4U;
}

// Header words per group: Q4_1 (d word, m word) and Q6_K (meta word, high-bit word) have two, the others one.
unsigned header_words(unsigned format) {
#pragma HLS INLINE
    return (format == GEMVQ_Q4_1 || format == GEMVQ_Q6_K) ? 2U : 1U;
}

unsigned words_per_row(unsigned blocks_per_row, unsigned format) {
#pragma HLS INLINE
    if (format == GEMVQ_Q5_K) return 3U * (blocks_per_row / 8U);
    if (format == GEMVQ_Q6_K) return 4U * (blocks_per_row / 8U);
    const unsigned full = blocks_per_row / 32U, tail = blocks_per_row % 32U;
    if (format == GEMVQ_Q8_0) return full * 17U + (tail ? 1U + tail / 2U : 0U);
    const unsigned headers = header_words(format);
    return full * (headers + 8U) + (tail ? headers + tail / 4U : 0U);
}

float bits_to_float(uint32_t bits) {
#pragma HLS INLINE
    union { uint32_t bits; float value; } c;
    c.bits = bits;
    return c.value;
}

// One weight word per cycle. Lane k handles block (block_in_row + k): Q8_0 uses lanes 0-1 with 32 int8 weights
// each, Q4_0/Q4_1 lanes 0-3 with 16 bytes each (element i < 16 = low nibble of byte i, i >= 16 = high nibble of
// byte i - 16; Q4_0 minus 8). Every group of 32 blocks starts with its header words (Q4_1: d word and m word),
// one cycle each. Q4_1 adds m * s of the Q8_1 activations per block.
// K-quants (Q5_K, Q6_K): a super-block (8 blocks) starts with its header words, then two quant words in the
// Q4_0 nibble order. Q5_K: weight = nibble + 16 * high bit, scale and min per 32 weights. Q6_K: weight =
// (nibble | 2 high bits << 4) - 32, int8 scale per 16 weights. The super-block's integer sums S = sum sc * (q . q8)
// and M = sum m * (sum q8) give one FP32 term d * dx * S - dmin * dx * M after its second quant word.
void compute_words(hls::stream<ap_uint<512>> & words, const xq_word * local_xq, const xs_word * local_xs,
                   unsigned format, unsigned word_count, unsigned blocks_per_row,
                   hls::stream<float> & row_results) {
    const bool q8 = format == GEMVQ_Q8_0, q4 = format == GEMVQ_Q4_0, q41 = GEMVQ_HAS_Q4_1 && format == GEMVQ_Q4_1;
    const bool k5 = format == GEMVQ_Q5_K, k6 = GEMVQ_HAS_Q6_K && format == GEMVQ_Q6_K, kq = k5 || k6;
    const unsigned bpw = blocks_per_word(format);
    const unsigned headers = header_words(format);
    ap_uint<512> scales = 0;   // first header word: scale word, Q4_1 d word or K-quant meta word
    ap_uint<512> scales2 = 0;  // second header word: Q4_1 m word or Q6_K high-bit word
    unsigned header_left = headers;
    bool k_second = false;     // K-quants: the next quant word is the super-block's second one
    int k_s = 0, k_m = 0;      // K-quant integer sums of the first quant word
    unsigned block_in_row = 0;
    unsigned group_left = 0;
    acc_t accumulator = 0;
    COMPUTE_WORDS: for (unsigned n = 0; n < word_count; ++n) {
#pragma HLS PIPELINE II=1
        const ap_uint<512> word = words.read();
        if (header_left != 0) {
            if (header_left == headers) scales = word;
            else scales2 = word;
            const unsigned left = blocks_per_row - block_in_row;
            group_left = kq ? 8U : (left < 32U ? left : 32U);
            --header_left;
        } else {
            const xq_word xq = local_xq[block_in_row / 4U];
            const xs_word xs = local_xs[block_in_row / 4U];
            const unsigned x_base = block_in_row & 3U;   // 0 or 2 for Q8_0, 0 otherwise
            const unsigned s_base = block_in_row & 31U;
            float lane_term[4];
            int lane_s[4], lane_m[4];
#pragma HLS ARRAY_PARTITION variable=lane_term complete
#pragma HLS ARRAY_PARTITION variable=lane_s complete
#pragma HLS ARRAY_PARTITION variable=lane_m complete
            for (unsigned k = 0; k < 4; ++k) {
#pragma HLS UNROLL
                const unsigned xb = (x_base + k) & 3U;
                // Select the lane's activation block once with constant ranges: indexing bytes of the 1024-bit
                // word with a variable offset would build one wide mux per byte.
                ap_uint<256> x_block;
                switch (xb) {
                case 0: x_block = xq.range(255, 0); break;
                case 1: x_block = xq.range(511, 256); break;
                case 2: x_block = xq.range(767, 512); break;
                default: x_block = xq.range(1023, 768); break;
                }
                // K-quant high bits of this lane: Q5_K one bit per weight of sub-block k or 4 + k (meta word),
                // Q6_K two bits per weight of super-block weights 128w + 32k .. + 31 (high-bit word).
                const ap_uint<32> high5 = k_second ? scales.range(160 + 32 * (4 + k) + 31, 160 + 32 * (4 + k))
                                                   : scales.range(160 + 32 * k + 31, 160 + 32 * k);
                const ap_uint<64> high6 = k_second ? scales2.range(256 + 64 * k + 63, 256 + 64 * k)
                                                   : scales2.range(64 * k + 63, 64 * k);
                int products[32];
#pragma HLS ARRAY_PARTITION variable=products complete
                for (unsigned i = 0; i < 32; ++i) {
#pragma HLS UNROLL
                    // The weight as an 8-bit pattern, wired from nibble and high bits without arithmetic, so all
                    // formats share one 8x8 multiplier per weight: Q4_0 nibble - 8 = {~n3 x5, n2..n0}; Q5_K
                    // nibble + 16 h = {000, h, n}; Q6_K (h1 h0 n) - 32 = {~h1 x3, h0, n}; Q4_1 {0000, n}.
                    const ap_uint<8> q4_byte = word.range((k * 16 + (i & 15)) * 8 + 7, (k * 16 + (i & 15)) * 8);
                    const ap_uint<4> nibble = i < 16 ? q4_byte.range(3, 0) : q4_byte.range(7, 4);
                    const bool h1 = high6[2 * i + 1], h0 = k6 ? bool(high6[2 * i]) : (k5 && high5[i]);
                    ap_uint<8> pattern;
                    if (q8) {
                        pattern = k < 2 ? ap_uint<8>(word.range((k * 32 + i) * 8 + 7, (k * 32 + i) * 8)) : ap_uint<8>(0);
                    } else if (q4) {
                        pattern = (ap_uint<5>(nibble[3] ? 0 : 31), ap_uint<3>(nibble.range(2, 0)));
                    } else {
                        pattern = (ap_uint<3>(k6 && !h1 ? 7 : 0), ap_uint<1>(h0), nibble);
                    }
                    const int weight = int(ap_int<8>(pattern));
                    const int activation = int(ap_int<8>(x_block.range(i * 8 + 7, i * 8)));
                    products[i] = weight * activation;
                }
                int level[16];
#pragma HLS ARRAY_PARTITION variable=level complete
                for (unsigned i = 0; i < 16; ++i) {
#pragma HLS UNROLL
                    level[i] = products[2 * i] + products[2 * i + 1];
                }
                for (unsigned width = 8; width >= 2; width >>= 1) {
#pragma HLS UNROLL
                    for (unsigned i = 0; i < width; ++i) {
#pragma HLS UNROLL
                        level[i] = level[2 * i] + level[2 * i + 1];
                    }
                }
                const int half0 = level[0], half1 = level[1];  // weights 0..15 and 16..31 of the lane
                const int dot = half0 + half1;
                const unsigned si = (s_base + k) & 31U;
                const float weight_scale = half_bits_to_float(scales.range(si * 16 + 15, si * 16));
                const bool active = k < bpw;
                // Q4_0/Q8_0: xs holds the FP32 activation scale. Q4_1 (ggml: (d4 * d8) * sumi + m4 * s8): xs holds d8
                // and s8 as FP16, the m word the weight minima. One multiply path serves all three formats.
                const float x_scale = q41 ? half_bits_to_float(xs.range(xb * 32 + 15, xb * 32))
                                          : bits_to_float(xs.range(xb * 32 + 31, xb * 32).to_uint());
                const float min_term = q41 ? half_bits_to_float(scales2.range(si * 16 + 15, si * 16)) *
                                                 half_bits_to_float(xs.range(xb * 32 + 31, xb * 32 + 16))
                                           : 0.0f;
                const float product = (weight_scale * x_scale) * float(dot);
                lane_term[k] = active ? (q41 ? product + min_term : product) : 0.0f;
                // K-quant scales (meta bytes, read as int8): Q5_K one per lane (sub-block k or 4 + k, byte 4 + sub)
                // and its min (byte 12 + sub); Q6_K one per 16 weights (bytes 4 + 8w + 2k and the next one).
                const unsigned sub5 = k_second ? 4 + k : k;
                const unsigned sub6 = k_second ? 8 + 2 * k : 2 * k;
                const ap_int<8> sc5 = ap_int<8>(scales.range(32 + 8 * sub5 + 7, 32 + 8 * sub5));
                const ap_int<8> sc_a = k6 ? ap_int<8>(scales.range(32 + 8 * sub6 + 7, 32 + 8 * sub6)) : sc5;
                const ap_int<8> sc_b = k6 ? ap_int<8>(scales.range(40 + 8 * sub6 + 7, 40 + 8 * sub6)) : sc5;
                const ap_uint<6> mn5 = scales.range(96 + 8 * sub5 + 5, 96 + 8 * sub5);
                // Two scale products per lane (Q5_K: the same scale twice) and the Q5_K min product, in DSPs.
                const int scaled_a = int(sc_a * ap_int<20>(half0));
#pragma HLS BIND_OP variable=scaled_a op=mul impl=dsp
                const int scaled_b = int(sc_b * ap_int<20>(half1));
#pragma HLS BIND_OP variable=scaled_b op=mul impl=dsp
                const int min_product = int(mn5 * ap_int<16>(xs.range(47 + 16 * k, 32 + 16 * k)));
#pragma HLS BIND_OP variable=min_product op=mul impl=dsp
                lane_s[k] = scaled_a + scaled_b;
                lane_m[k] = k5 ? min_product : 0;
            }
            // The lanes of one word are summed in FP32 and converted once: a single variable shift per word
            // instead of four (~2.5k LUT less per engine). The row sum stays order-independent fixed point.
            float word_term = (lane_term[0] + lane_term[1]) + (lane_term[2] + lane_term[3]);
            if (kq) {
                const int s_sum = (k_second ? k_s : 0) + (lane_s[0] + lane_s[1]) + (lane_s[2] + lane_s[3]);
                const int m_sum = (k_second ? k_m : 0) + (lane_m[0] + lane_m[1]) + (lane_m[2] + lane_m[3]);
                const float dx = bits_to_float(xs.range(31, 0).to_uint());
                const float d = half_bits_to_float(scales.range(15, 0)) * dx;
                const float dmin = half_bits_to_float(scales.range(31, 16)) * dx;  // Q6_K: zero
                word_term = k_second ? float(s_sum) * d - float(m_sum) * dmin : 0.0f;
                k_s = s_sum;
                k_m = m_sum;
                k_second = !k_second;
            }
            const acc_t base = block_in_row == 0 ? acc_t(0) : accumulator;
            const acc_t next = base + float_to_fixed(word_term);
            accumulator = next;
            block_in_row += bpw;
            group_left -= bpw;
            if (group_left == 0) header_left = headers;
            if (block_in_row == blocks_per_row) {
                row_results.write(fixed_to_float(next));
                block_in_row = 0;
            }
        }
    }
}

} // namespace
} // namespace gemvqw_engine
#endif
