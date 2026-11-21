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

float bits_to_float(uint32_t bits) {
#pragma HLS INLINE
    union { uint32_t bits; float value; } c;
    c.bits = bits;
    return c.value;
}

// roundf(|v|) from the FP32 bits of v (|v| < 2^23, half away from zero): exact like truncating, subtracting and
// comparing in FP32, without a float-to-int, an int-to-float and a subtraction per lane.
uint32_t round_magnitude(uint32_t bits) {
#pragma HLS INLINE
    const uint32_t exponent = (bits >> 23) & 0xffU;
    if (exponent < 126U) return 0U;  // |v| < 0.5
    const uint32_t shift = 150U - exponent, mantissa = (bits & 0x7fffffU) | 0x800000U;
    const uint32_t q = mantissa >> shift, rem = mantissa & ((uint32_t(1) << shift) - 1U);
    const uint32_t half = uint32_t(1) << (shift - 1U);
    return q + (rem >= half ? 1U : 0U);
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
// One broadcast word of the activation chain (packed to 1152 bits by the stream).
struct xmsg {
    xq_word q;
    xs_word s;
};

float word_lane_float(const ap_uint<512> & word, unsigned lane) {
#pragma HLS INLINE
    union { uint32_t bits; float value; } c;
    c.bits = word.range(lane * 32 + 31, lane * 32).to_uint();
    return c.value;
}

// Q8_0-quantize x like llama.cpp's quantize_row_q8_0: scale = amax / 127 stored as FP16, roundf(x / scale).
// x is read as 512-bit words (16 FP32 lanes, lane 0 in bits 31:0), two words per block.
// Writes N identical copies so N engines read in parallel. blocks must be a multiple of 4.
// The xs lane of a block holds the FP16 scale in bits 15:0 and the sum of the block's 32 quants in bits 31:16 (the
// engine needs it for Q4_0, see compute_words); with q8_1 (Q8_1 for Q4_1, quantize_row_q8_1_ref) bits 31:16 hold
// FP16(sum of the quants * scale).
// CHAIN == false writes N copies into local_xq / local_xs, CHAIN == true sends every finished word into `link`
// instead (see quantize_any_stream). SRC_STREAM: the words of x come from `xin` (in order) instead of `x`.
template <int N, bool CHAIN, bool SRC_STREAM = false>
void quantize_x_impl(const ap_uint<512> * x, unsigned blocks, xq_word (*local_xq)[XWORDS], xs_word (*local_xs)[XWORDS],
                     bool q8_1, hls::stream<xmsg> & link, hls::stream<ap_uint<512>> * xin = nullptr) {
    xq_word pending_q = 0;
    xs_word pending_s = 0;
    QUANTIZE_X: for (unsigned block = 0; block < blocks; ++block) {
#pragma HLS PIPELINE II=2
        float value[32];
#pragma HLS ARRAY_PARTITION variable=value complete
        uint32_t magnitude[32];  // fabsf as FP32 bits: for finite values the integer order is the float order
#pragma HLS ARRAY_PARTITION variable=magnitude complete
        const ap_uint<512> low = SRC_STREAM ? xin->read() : x[block * 2U];
        const ap_uint<512> high = SRC_STREAM ? xin->read() : x[block * 2U + 1U];
        for (unsigned j = 0; j < 32; ++j) {
#pragma HLS UNROLL
            value[j] = word_lane_float(j < 16 ? low : high, j & 15U);
            magnitude[j] = float_to_bits(value[j]) & 0x7fffffffU;
        }
        for (unsigned width = 16; width >= 1; width >>= 1) {
#pragma HLS UNROLL
            for (unsigned i = 0; i < width; ++i) {
#pragma HLS UNROLL
                magnitude[i] = magnitude[2 * i] > magnitude[2 * i + 1] ? magnitude[2 * i] : magnitude[2 * i + 1];
            }
        }
        const float scale = bits_to_float(magnitude[0]) / 127.0f;
        const float reciprocal = scale != 0.0f ? 1.0f / scale : 0.0f;
        ap_uint<256> packed = 0;
        int quant_sum = 0;
        for (unsigned j = 0; j < 32; ++j) {
#pragma HLS UNROLL
            // roundf(): |x / scale| <= 127 (plus rounding), half away from zero.
            const uint32_t scaled = float_to_bits(value[j] * reciprocal);
            const int magnitude_q = int(round_magnitude(scaled));
            const int8_t rounded = int8_t((scaled >> 31) != 0U ? -magnitude_q : magnitude_q);
            packed.range(j * 8 + 7, j * 8) = ap_uint<8>(rounded);
            quant_sum += rounded;
        }
        const ap_uint<16> scale_half = fp32_to_fp16(scale);
        const uint32_t lane = uint32_t(scale_half) |
                              (uint32_t(q8_1 ? fp32_to_fp16(float(quant_sum) * scale) : ap_uint<16>(quant_sum)) << 16);
        // Shift registers: after four blocks block 0 sits at the lowest position.
        pending_q = (pending_q >> 256) | (xq_word(packed) << 768);
        pending_s = (pending_s >> 32) | (xs_word(lane) << 96);
        if ((block & 3U) == 3U) {
            if (CHAIN) {
                link.write(xmsg{pending_q, pending_s});
            } else {
                for (int e = 0; e < N; ++e) {
#pragma HLS UNROLL
                    local_xq[e][block / 4U] = pending_q;
                    local_xs[e][block / 4U] = pending_s;
                }
            }
        }
    }
}

template <int N>
void quantize_x(const ap_uint<512> * x, unsigned blocks, xq_word (*local_xq)[XWORDS], xs_word (*local_xs)[XWORDS],
                bool q8_1 = false) {
    hls::stream<xmsg> unused;
    quantize_x_impl<N, false>(x, blocks, local_xq, local_xs, q8_1, unused);
}

// Q8_K-quantize x like llama.cpp's quantize_row_q8_K_ref for the K-quant formats: per 256 values the value with
// the largest magnitude (the first one on ties) gives iscale = -127 / max, q = min(127, nearest_int(iscale * x)),
// round half to even through the FP32 addition of 1.5 * 2^23, and d = 1 / iscale (all zero for a zero block).
// local_xq as for Q8_0 (128 activations per word, element e of the word at bits 8e..8e+7); local_xs per word:
// bits 31:0 the super-block's d (FP32), bits 32+12h..43+12h the sum of the 16 quants 16h..16h+15 of the word
// (the stored int8 values, as in ggml).
template <int N, bool CHAIN, bool SRC_STREAM = false>
void quantize_x_k_impl(const ap_uint<512> * x, unsigned superblocks, xq_word (*local_xq)[XWORDS],
                       xs_word (*local_xs)[XWORDS], hls::stream<xmsg> & link, hls::stream<ap_uint<512>> * xin = nullptr) {
    QXK_SUPER: for (unsigned sb = 0; sb < superblocks; ++sb) {
        ap_uint<512> raw[16];
#pragma HLS ARRAY_PARTITION variable=raw complete
        uint32_t amax = 0;
        float vmax = 0.0f;
        QXK_MAX: for (unsigned w = 0; w < 16; ++w) {
#pragma HLS PIPELINE II=1
            const ap_uint<512> word = SRC_STREAM ? xin->read() : x[sb * 16U + w];
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
        ap_uint<96> pending_sum = 0;
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
                part += int(ap_int<8>(q));
            }
            // Shift registers: after eight words the super-block half's first value sits at the lowest position.
            pending_q = (pending_q >> 128) | (xq_word(packed) << 896);
            pending_sum = (pending_sum >> 12) | (ap_uint<96>(ap_uint<12>(part)) << 84);
            if ((w & 7U) == 7U) {
                const unsigned half = w / 8U;
                xs_word xs = xs_word(float_to_bits(d));
                xs.range(127, 32) = pending_sum;
                if (CHAIN) {
                    link.write(xmsg{pending_q, xs});
                } else {
                    for (int e = 0; e < N; ++e) {
#pragma HLS UNROLL
                        local_xq[e][sb * 2U + half] = pending_q;
                        local_xs[e][sb * 2U + half] = xs;
                    }
                }
            }
        }
    }
}

template <int N>
void quantize_x_k(const ap_uint<512> * x, unsigned superblocks, xq_word (*local_xq)[XWORDS],
                  xs_word (*local_xs)[XWORDS]) {
    hls::stream<xmsg> unused;
    quantize_x_k_impl<N, false>(x, superblocks, local_xq, local_xs, unused);
}

// Both quantizers in one process (gemvqc16 runs it inside its pass): Q8_K for the K-quant formats (units =
// superblocks), else Q8_0 / Q8_1 (units = blocks).
void quantize_any_stream(hls::stream<ap_uint<512>> & xin, bool k_quant, unsigned units, hls::stream<xmsg> & link,
                         bool q8_1) {
    if (k_quant) {
        quantize_x_k_impl<1, true, true>(nullptr, units, nullptr, nullptr, link, &xin);
    } else {
        quantize_x_impl<1, true, true>(nullptr, units, nullptr, nullptr, q8_1, link, &xin);
    }
}

// Source of the quantizer inside the gemvqc16 pass, on-chip only (act belongs to the store process there): the
// deferred norm's row normalized on the fly from nx (the row) and nw (the weight), (x * scale) * w per lane (* w
// only if weighted), the arithmetic of the RMSNORM opcode, so the quantizer sees exactly the values that opcode
// would have written; or the raw words of x, copied to xbuf before the pass; or (from_gate) the words of `gate` as
// they come (the GDN step's output in the same pass).
void x_source_onchip(const ap_uint<512> * nx, const ap_uint<512> * nw, const ap_uint<512> * xbuf, float scale,
                     bool weighted, bool from_norm, bool from_gate, unsigned words, hls::stream<ap_uint<512>> & gate,
                     hls::stream<ap_uint<512>> & out) {
    X_SOURCE_ONCHIP: for (unsigned i = 0; i < words; ++i) {
#pragma HLS PIPELINE II=1
        const ap_uint<512> xw = nx[i], gw = nw[i], raw = xbuf[i];
        ap_uint<512> v = 0;
        for (unsigned l = 0; l < 16; ++l) {
#pragma HLS UNROLL
            const float s = word_lane_float(xw, l) * scale;
            v.range(l * 32 + 31, l * 32) = float_to_bits(weighted ? s * word_lane_float(gw, l) : s);
        }
        ap_uint<512> g = 0;
        if (from_gate) g = gate.read();
        out.write(from_gate ? g : (from_norm ? v : raw));
    }
}

// Broadcast chain: the 16 engines of gemvqc16 each keep a copy of the quantized x. Writing all copies from the
// quantizer is a 1152-bit net with fan-out 16 that crosses the SLR boundary, and its routing delay (~9 ns)
// limited the kernel clock. Here the words travel from engine to engine instead: stage e writes its own copy
// and forwards the word through a registered stream, so every net is local and fans out to two sinks.
// Each stage also hands every word to its engine through a shallow FIFO (local); the engine copies x into its own
// buffer before it computes (compute_group in gemvqc16.cpp). A column-blocked GEMV sends x block by block, and the
// FIFO holds the next block while the engine still computes the current one. (A stream of blocks would need two
// 1024-bit-wide buffers per engine, ~17 RAMB36 more each; with one buffer HLS 2025.2 deadlocks.)
void xcopy_forward(hls::stream<xmsg> & in, hls::stream<xmsg> & out, hls::stream<xmsg> & local, unsigned words) {
    XCOPY_FORWARD: for (unsigned i = 0; i < words; ++i) {
#pragma HLS PIPELINE II=1
        const xmsg m = in.read();
        local.write(m);
        out.write(m);
    }
}
void xcopy_last(hls::stream<xmsg> & in, hls::stream<xmsg> & local, unsigned words) {
    XCOPY_LAST: for (unsigned i = 0; i < words; ++i) {
#pragma HLS PIPELINE II=1
        local.write(in.read());
    }
}
#define XCOPY_STAGE(n, words) xcopy_forward(link[n], link[n + 1], xloc[n], words)
#define XCOPY_CHAIN(words)                                                                                           \
    XCOPY_STAGE(0, words);   XCOPY_STAGE(1, words);   XCOPY_STAGE(2, words);                                         \
    XCOPY_STAGE(3, words);   XCOPY_STAGE(4, words);   XCOPY_STAGE(5, words);                                         \
    XCOPY_STAGE(6, words);   XCOPY_STAGE(7, words);   XCOPY_STAGE(8, words);                                         \
    XCOPY_STAGE(9, words);   XCOPY_STAGE(10, words);  XCOPY_STAGE(11, words);                                        \
    XCOPY_STAGE(12, words);  XCOPY_STAGE(13, words);  XCOPY_STAGE(14, words);                                        \
    xcopy_last(link[15], xloc[15], words)
// The same chain for 20 engines (gemvqc16 with GEMVQC16_ENGINES 20).
#define XCOPY_CHAIN20(words)                                                                                         \
    XCOPY_STAGE(0, words);   XCOPY_STAGE(1, words);   XCOPY_STAGE(2, words);                                         \
    XCOPY_STAGE(3, words);   XCOPY_STAGE(4, words);   XCOPY_STAGE(5, words);                                         \
    XCOPY_STAGE(6, words);   XCOPY_STAGE(7, words);   XCOPY_STAGE(8, words);                                         \
    XCOPY_STAGE(9, words);   XCOPY_STAGE(10, words);  XCOPY_STAGE(11, words);                                        \
    XCOPY_STAGE(12, words);  XCOPY_STAGE(13, words);  XCOPY_STAGE(14, words);                                        \
    XCOPY_STAGE(15, words);  XCOPY_STAGE(16, words);  XCOPY_STAGE(17, words);                                        \
    XCOPY_STAGE(18, words);                                                                                          \
    xcopy_last(link[19], xloc[19], words)
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
    if (format == GEMVQ_Q6_K) return (blocks_per_row / 8U + 2U) / 3U + 3U * (blocks_per_row / 8U);  // packed
    const unsigned full = blocks_per_row / 32U, tail = blocks_per_row % 32U;
    if (format == GEMVQ_Q8_0) return full * 17U + (tail ? 1U + tail / 2U : 0U);
    const unsigned headers = header_words(format);
    return full * (headers + 8U) + (tail ? headers + tail / 4U : 0U);
}

#if GEMVQ_INT_SCALE
// A forced pipeline register (the pattern of hls::reg): returns its argument with latency 1. HLS chains logic into
// one clock by its own delay estimate, which the routed engine paths exceeded by far (multiplier, shifter and
// accumulator in one clock, 18-32 logic levels); these registers split the scale path into fixed stages.
template <typename T>
T stage_reg(T d) {
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY min=1 max=1
#pragma HLS INLINE off
    return d;
}

// Exact integer scale arithmetic. A scale is split into sign, significand m and exponent e (value = m * 2^e); the
// product of two scales is then an integer product, and the block sum times the scale product is shifted into the
// fixed-point row accumulator (LSB 2^-ACC_FRAC_BITS). No rounding happens except dropping bits below the LSB.
struct scale_parts {
    bool neg;
    ap_uint<24> m;
    int e;
};

// FP16 bits: value = m * 2^e with an 11-bit significand (subnormals keep their raw fraction and e = -24).
scale_parts half_scale_parts(ap_uint<16> bits) {
#pragma HLS INLINE
    scale_parts r;
    const ap_uint<5> ex = bits.range(14, 10);
    const ap_uint<10> f = bits.range(9, 0);
    r.neg = bits[15];
    r.m = ex == 0 ? ap_uint<24>(f) : ap_uint<24>(ap_uint<11>((ap_uint<1>(1), f)));
    r.e = ex == 0 ? -24 : int(ex) - 25;
    return r;
}

// FP32 bits: value = m * 2^e with a 24-bit significand.
scale_parts float_scale_parts(ap_uint<32> bits) {
#pragma HLS INLINE
    scale_parts r;
    const ap_uint<8> ex = bits.range(30, 23);
    r.neg = bits[31];
    r.m = ex == 0 ? ap_uint<24>(0) : ap_uint<24>((ap_uint<1>(1), bits.range(22, 0)));
    r.e = int(ex) - 150;
    return r;
}

// p * 2^exp2 as a fixed-point accumulator value. exp2 is clamped to [LO, LO + RANGE]; the shift is one left shift
// by exp2 - LO followed by a constant right shift, so a single barrel shifter covers negative and positive exponents.
template <int PW, int LO, int RANGE>
acc_t shifted_fixed(ap_int<PW> p, int exp2) {
#pragma HLS INLINE
    static_assert(ACC_FRAC_BITS + LO <= 0, "LO must be at most -ACC_FRAC_BITS");
    int t = exp2 - LO;
    t = t < 0 ? 0 : (t > RANGE ? RANGE : t);
    const ap_int<PW + RANGE> v = ap_int<PW + RANGE>(p) << t;
    return acc_t(v >> (-(ACC_FRAC_BITS + LO)));
}

// The two scale multipliers of a lane. Q4_0 / Q8_0 / Q4_1: dot * weight_scale * activation_scale (both FP16), as
// the product of the significands and its product with the block sum; term is the lane's value. K-quants: the same
// multipliers give the lane's two scale products, sc_a * half0 + sc_b * half1 (s_sum). The format groups are never
// active together.
struct lane_scaled {
    acc_t term;
    int s_sum;
};
lane_scaled scale_lane_fixed(bool kq, int dot, ap_uint<16> weight_scale_bits, ap_uint<16> x_scale_bits, bool active,
                             ap_int<8> sc_a, ap_int<8> sc_b, ap_int<20> half0, ap_int<20> half1) {
#pragma HLS INLINE
    const scale_parts w = half_scale_parts(weight_scale_bits);
    const scale_parts x = half_scale_parts(x_scale_bits);
    const ap_int<12> a1 = kq ? ap_int<12>(sc_a) : ap_int<12>(ap_uint<11>(w.m));
    const ap_int<20> b1 = kq ? half0 : ap_int<20>(ap_uint<11>(x.m));
    const ap_int<32> first = a1 * b1;  // significand product (22 bits), or sc_a * half0
#pragma HLS BIND_OP variable=first op=mul impl=dsp
    const ap_int<22> d = active ? ap_int<22>(w.neg ? -dot : dot) : ap_int<22>(0);
    const ap_int<22> a2 = kq ? ap_int<22>(sc_b) : d;
    const ap_int<23> b2 = kq ? ap_int<23>(half1) : ap_int<23>(ap_uint<22>(first));
    const ap_int<45> p = stage_reg(ap_int<45>(a2 * b2));
    lane_scaled r;
    r.term = stage_reg(shifted_fixed<45, -58, 70>(p, w.e + x.e));
    r.s_sum = int(first) + int(ap_int<28>(p));
    return r;
}

// K-quant word: s_sum * d - m_sum * dmin with d = d_half * dx, dmin = dmin_half * dx (dx the FP32 scale of the Q8_K block).
acc_t scale_k_fixed(int s_sum, int m_sum, ap_uint<16> d_bits, ap_uint<16> dmin_bits, ap_uint<32> dx_bits) {
#pragma HLS INLINE
    const scale_parts dh = half_scale_parts(d_bits);
    const scale_parts mh = half_scale_parts(dmin_bits);
    const scale_parts dx = float_scale_parts(dx_bits);
    const ap_uint<35> md = stage_reg(ap_uint<35>(ap_uint<11>(dh.m) * ap_uint<24>(dx.m)));
    const ap_uint<35> mm = stage_reg(ap_uint<35>(ap_uint<11>(mh.m) * ap_uint<24>(dx.m)));
    const bool neg_s = dh.neg != dx.neg, neg_m = mh.neg != dx.neg;
    // Widths: |S| <= 16 * 128 * (16 * 128 * 32) = 2^27 (Q6_K), |M| <= 8 * 63 * (32 * 128) < 2^21 (Q5_K).
    const ap_int<29> s_val(s_sum);
    const ap_int<23> m_val(m_sum);
    const ap_int<29> si = neg_s ? ap_int<29>(-s_val) : s_val;
    const ap_int<23> mi = neg_m ? ap_int<23>(-m_val) : m_val;
    const ap_int<69> ps = stage_reg(ap_int<69>(si * ap_int<36>(md)));
    const ap_int<69> pm = stage_reg(ap_int<69>(mi * ap_int<36>(mm)));
    return stage_reg(shifted_fixed<69, -81, 78>(ps, dh.e + dx.e)) - stage_reg(shifted_fixed<69, -81, 78>(pm, mh.e + dx.e));
}
#endif

// One weight word per cycle. Lane k handles block (block_in_row + k): Q8_0 uses lanes 0-1 with 32 int8 weights
// each, Q4_0/Q4_1 lanes 0-3 with 16 bytes each (element i < 16 = low nibble of byte i, i >= 16 = high nibble of
// byte i - 16; Q4_0 minus 8). Every group of 32 blocks starts with its header words (Q4_1: d word and m word),
// one cycle each. Q4_1 adds m * s of the Q8_1 activations per block.
// K-quants (Q5_K, Q6_K): a super-block (8 blocks) starts with its header words, then two quant words in the
// Q4_0 nibble order. Q5_K: weight = nibble + 16 * high bit, scale and min per 32 weights. Q6_K: weight =
// (nibble | 2 high bits << 4) - 32, int8 scale per 16 weights. The super-block's integer sums S = sum sc * (q . q8)
// and M = sum m * (sum q8) give one FP32 term d * dx * S - dmin * dx * M after its second quant word.
//
// Products: 64 multipliers of 18 x 18 bits (one DSP each). The 4-bit formats have 128 weights per word, and one
// multiplier gives the sum of two of their products: with the weights as unsigned values u (Q4_0 the nibble, Q5_K
// nibble + 16 h, Q6_K the 6 bits) and activations xa, xb,
//     (xa * 2^10 + ua) * (xb * 2^10 + ub) = xa xb * 2^20 + (xa ub + xb ua) * 2^10 + ua ub,
// so bits 24:10 hold xa ub + xb ua after two corrections. The 5 low bits of xa xb reach the field from above: the
// multiplier subtracts them at bit 20. ua ub reaches it from below only for Q6_K (ua ub >= 1024): c = ua ub >> 10
// is at most 3. The multiplier also subtracts the 2 low bits of the sum at bit 10 (simple logic of the operands'
// low bits): c is then alone in bits 11:10 and drops out, and the 2 low bits are added after the sum. The signed
// weights are u - 8 (Q4_0) and u - 32 (Q6_K): the engine subtracts 8 or 32 times the sum of the activations, which
// the quantizer delivers (xs). All of it is exact integer arithmetic.
// Q8_0 has 64 weights per word: a multiplier gets (xa * 2^10) * (w + 128) with one int8 weight w, the product in
// bits 25:10, and the engine subtracts 128 times the sum of the block's activations.
// A row may also be computed in column blocks (gemvqc16's column-blocked GEMV): one call per block, blocks_per_row
// then counts the block's columns. The row sums of a block go to racc (indexed by the row of the call) and the next
// block adds to them (not first); only the last block writes row_results. The fixed-point sum does not depend on the
// order, so the results are the bits of the whole row in one call.
constexpr unsigned MAX_BLOCKED_ROWS = 512;  // rows per engine of a column-blocked GEMV
void compute_words(hls::stream<ap_uint<512>> & words, const xq_word * local_xq, const xs_word * local_xs,
                   unsigned format, unsigned word_count, unsigned blocks_per_row,
                   hls::stream<float> & row_results, acc_t * racc = nullptr, bool first = true, bool last = true) {
    const bool q8 = format == GEMVQ_Q8_0, q4 = format == GEMVQ_Q4_0, q41 = GEMVQ_HAS_Q4_1 && format == GEMVQ_Q4_1;
    const bool k5 = format == GEMVQ_Q5_K, k6 = GEMVQ_HAS_Q6_K && format == GEMVQ_Q6_K, kq = k5 || k6;
    const unsigned bpw = blocks_per_word(format);
    const unsigned headers = header_words(format);
    ap_uint<512> scales = 0;   // first header word: scale word, Q4_1 d word or K-quant meta word
    ap_uint<512> scales2 = 0;  // second header word: Q4_1 m word or Q6_K high-bit word
    // Packed Q6_K: one meta word per group of up to three super-blocks (meta j in bits 160j..160j+159); k6_slot is
    // the super-block's place in its group. A group starts with the meta word and the high-bit word, the other
    // super-blocks with the high-bit word only. The meta word stays in scales (Q6_K reads only its bits 159:0) and
    // moves down by one meta at the high-bit word of the next super-block: a wire shift instead of a slot select.
    unsigned k6_slot = 0;
    unsigned header_left = headers;
    bool k_second = false;     // K-quants: the next quant word is the super-block's second one
    int k_s = 0, k_m = 0;      // K-quant integer sums of the first quant word
    unsigned block_in_row = 0;
    unsigned group_left = 0;
    unsigned row = 0;  // row of this call (racc index)
    acc_t accumulator = 0;
    COMPUTE_WORDS: for (unsigned n = 0; n < word_count; ++n) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=racc inter false
        const ap_uint<512> word = words.read();
        if (header_left != 0) {
            // One select per register: scales takes the word (first header word), moves to the next Q6_K meta, or
            // holds; scales2 takes the word (second header word) or holds.
            const bool first_word = header_left == headers;
            const bool next_meta = k6 && !first_word && k6_slot != 0;
            scales = first_word ? word : (next_meta ? ap_uint<512>(scales >> 160) : scales);
            scales2 = first_word ? scales2 : word;
            const unsigned left = blocks_per_row - block_in_row;
            group_left = kq ? 8U : (left < 32U ? left : 32U);
            --header_left;
        } else {
            const xq_word xq = local_xq[block_in_row / 4U];
            const xs_word xs = local_xs[block_in_row / 4U];
            const unsigned x_base = block_in_row & 3U;   // 0 or 2 for Q8_0, 0 otherwise
            const unsigned s_base = block_in_row & 31U;
#if GEMVQ_INT_SCALE
            acc_t lane_term[4];
            float lane_min[4];  // Q4_1: m * s of the lane's block (FP16 x FP16, exact in FP32)
#pragma HLS ARRAY_PARTITION variable=lane_min complete
#else
            float lane_term[4];
#endif
            int lane_s[4], lane_m[4];
            int lane_dot[4], lane_half0[4], lane_half1[4], lane_xsum[4];
#pragma HLS ARRAY_PARTITION variable=lane_dot complete
#pragma HLS ARRAY_PARTITION variable=lane_half0 complete
#pragma HLS ARRAY_PARTITION variable=lane_half1 complete
#pragma HLS ARRAY_PARTITION variable=lane_xsum complete
#pragma HLS ARRAY_PARTITION variable=lane_term complete
#pragma HLS ARRAY_PARTITION variable=lane_s complete
#pragma HLS ARRAY_PARTITION variable=lane_m complete
            for (unsigned k = 0; k < 4; ++k) {
#pragma HLS UNROLL
                // Q8_0 lanes 0/1 read the scales of blocks x_base + k (x_base 0 or 2), all other cases block k.
                const unsigned xb = k < 2 ? ((x_base & 2U) ? k + 2U : k) : k;
                // The 16 multipliers of lane k. 4-bit formats: the 32 weights of the lane (activation block k).
                // Q8_0: block k % 2 of the word with activation block x_base + k % 2. The lane whose own activation
                // block that is (k or k ^ 2) takes the even elements, the other lane the odd ones: multiplier p reads
                // its first activation from its own block or from the other block's odd element, a 2:1 select.
                // All ranges have constant bounds.
                const ap_uint<256> x_packed = xq.range(256 * k + 255, 256 * k);
                const ap_uint<256> x_other = xq.range(256 * (k ^ 2U) + 255, 256 * (k ^ 2U));
                const bool odd8 = q8 && ((x_base & 2U) != 0) != (k >= 2);  // Q8_0: this lane has the odd elements
                // K-quant high bits of this lane: Q5_K one bit per weight of sub-block k or 4 + k (meta word),
                // Q6_K two bits per weight of super-block weights 128w + 32k .. + 31 (high-bit word).
                const ap_uint<32> high5 = k_second ? scales.range(160 + 32 * (4 + k) + 31, 160 + 32 * (4 + k))
                                                   : scales.range(160 + 32 * k + 31, 160 + 32 * k);
                const ap_uint<64> high6 = k_second ? scales2.range(256 + 64 * k + 63, 256 + 64 * k)
                                                   : scales2.range(64 * k + 63, 64 * k);
                ap_uint<6> u[32];  // the lane's weights as unsigned values, wired from nibble and high bits
#pragma HLS ARRAY_PARTITION variable=u complete
                for (unsigned i = 0; i < 32; ++i) {
#pragma HLS UNROLL
                    const ap_uint<8> q4_byte = word.range((k * 16 + (i & 15)) * 8 + 7, (k * 16 + (i & 15)) * 8);
                    const ap_uint<4> nibble = i < 16 ? q4_byte.range(3, 0) : q4_byte.range(7, 4);
                    const bool h1 = k6 && high6[2 * i + 1], h0 = k6 ? bool(high6[2 * i]) : (k5 && high5[i]);
                    u[i] = (ap_uint<1>(h1), ap_uint<1>(h0), nibble);
                }
                // Multiplier p: weights 2p and 2p + 1 of the lane (Q8_0: one weight). Its sum of products is
                // 4 * quad[p] + rest[p].
                ap_int<14> quad[16];
                ap_uint<2> rest[16];
#pragma HLS ARRAY_PARTITION variable=quad complete
#pragma HLS ARRAY_PARTITION variable=rest complete
                for (unsigned p = 0; p < 16; ++p) {
#pragma HLS UNROLL
                    const ap_int<8> xa = odd8 ? ap_int<8>(x_other.range(p * 16 + 15, p * 16 + 8))
                                              : ap_int<8>(x_packed.range(p * 16 + 7, p * 16));
                    const ap_int<8> xb2 = q8 ? ap_int<8>(0) : ap_int<8>(x_packed.range(p * 16 + 15, p * 16 + 8));
                    const ap_uint<6> ua = q8 ? ap_uint<6>(0) : u[2 * p + 1];
                    // Q8_0: the int8 weight plus 128 (element 2p or 2p + 1 of block k % 2).
                    const unsigned b8 = 32 * (k % 2) + 2 * p;
                    const ap_uint<8> w8 = odd8 ? word.range((b8 + 1) * 8 + 7, (b8 + 1) * 8) : word.range(b8 * 8 + 7, b8 * 8);
                    const ap_uint<8> ub = q8 ? ap_uint<8>(w8 ^ ap_uint<8>(0x80)) : ap_uint<8>(u[2 * p]);
                    const ap_int<18> fa = (xa, ap_uint<4>(0), ua);
                    const ap_int<18> fb = (xb2, ap_uint<2>(0), ub);
                    // The multiplier subtracts the 5 low bits of xa * xb at bit 20 and m = (xa ub + xb ua) mod 4 at
                    // bit 10. Bits 24:12 are then (xa ub + xb ua - m) / 4, and c stays in bits 11:10.
                    const ap_uint<5> low = ap_uint<5>(xa.range(4, 0)) * ap_uint<5>(xb2.range(4, 0));
#pragma HLS BIND_OP variable=low op=mul impl=fabric
                    const bool a0 = xa[0] && ub[0], b0 = xb2[0] && ua[0];
                    const bool m0 = a0 != b0;
                    const bool m1 = ((xa[1] && ub[0]) != (xa[0] && ub[1])) !=
                                    (((xb2[1] && ua[0]) != (xb2[0] && ua[1])) != (a0 && b0));
                    rest[p] = (ap_uint<1>(m1), ap_uint<1>(m0));
                    const ap_uint<25> fix = (low, ap_uint<8>(0), rest[p], ap_uint<10>(0));
                    const ap_int<37> product = fa * fb - ap_int<37>(fix);
                    quad[p] = (ap_uint<1>(q8 ? product[25] : product[24]), ap_uint<13>(product.range(24, 12)));
                }
                int half_sum[2];
#pragma HLS ARRAY_PARTITION variable=half_sum complete
                for (unsigned h = 0; h < 2; ++h) {
#pragma HLS UNROLL
                    const unsigned b = 8 * h;
                    const ap_int<17> quads = ((quad[b] + quad[b + 1]) + (quad[b + 2] + quad[b + 3])) +
                                             ((quad[b + 4] + quad[b + 5]) + (quad[b + 6] + quad[b + 7]));
                    const ap_uint<5> rests = ((rest[b] + rest[b + 1]) + (rest[b + 2] + rest[b + 3])) +
                                             ((rest[b + 4] + rest[b + 5]) + (rest[b + 6] + rest[b + 7]));
                    half_sum[h] = int(quads) * 4 + int(rests);
                }
                // Sums of the activations: Q8_0 quantizer one per block (bits 31:16 of the lane), Q8_K one per 16.
                const ap_int<13> x_sum = xs.range(k * 32 + 28, k * 32 + 16);
                const ap_int<12> x_sum_a = xs.range(32 + 24 * k + 11, 32 + 24 * k);
                const ap_int<12> x_sum_b = xs.range(32 + 24 * k + 23, 32 + 24 * k + 12);
                const int offset_a = k6 ? int(x_sum_a) * 32 : (q4 ? int(x_sum) * 8 : 0);  // Q8_0: below
                const int offset_b = k6 ? int(x_sum_b) * 32 : 0;
                const int half0 = half_sum[0] - offset_a, half1 = half_sum[1] - offset_b;  // weights 0..15, 16..31
                lane_dot[k] = half0 + half1;
                // Q8_0: block k is the multipliers of lanes k and k + 2, added below.
                lane_half0[k] = half0;
                lane_half1[k] = half1;
                lane_xsum[k] = int(x_sum_a) + int(x_sum_b);
            }
            for (unsigned k = 0; k < 4; ++k) {
#pragma HLS UNROLL
                const unsigned xb = k < 2 ? ((x_base & 2U) ? k + 2U : k) : k;
                // Widths: 32 * 128 * 128 for a Q8_0 block, 32 * 128 * 15 for a lane of the 4-bit formats.
                const ap_int<18> dot4 = lane_dot[k];
                const ap_int<13> x_sum8 = xs.range(xb * 32 + 28, xb * 32 + 16);
                const ap_int<21> dot8 = k < 2 ? lane_dot[k] + lane_dot[k + 2] - int(x_sum8) * 128 : 0;
                const int dot = k < 2 && q8 ? int(dot8) : int(dot4);
                const int half0 = lane_half0[k], half1 = lane_half1[k];
                const unsigned si = (s_base + k) & 31U;
                const bool active = k < bpw;
#if GEMVQ_INT_SCALE
                lane_min[k] = q41 && active ? half_bits_to_float(scales2.range(si * 16 + 15, si * 16)) *
                                                  half_bits_to_float(xs.range(xb * 32 + 31, xb * 32 + 16))
                                            : 0.0f;
#else
                const float weight_scale = half_bits_to_float(scales.range(si * 16 + 15, si * 16));
                // xs holds the FP16 activation scale. Q4_1 (ggml: (d4 * d8) * sumi + m4 * s8): xs holds d8 and s8 as
                // FP16, the m word the weight minima. One multiply path serves all three formats.
                const float x_scale = half_bits_to_float(xs.range(xb * 32 + 15, xb * 32));
                const float min_term = q41 ? half_bits_to_float(scales2.range(si * 16 + 15, si * 16)) *
                                                 half_bits_to_float(xs.range(xb * 32 + 31, xb * 32 + 16))
                                           : 0.0f;
                const float product = (weight_scale * x_scale) * float(dot);
                lane_term[k] = active ? (q41 ? product + min_term : product) : 0.0f;
#endif
                // K-quant scales (meta bytes, read as int8): Q5_K one per lane (sub-block k or 4 + k, byte 4 + sub)
                // and its min (byte 12 + sub); Q6_K one per 16 weights (bytes 4 + 8w + 2k and the next one).
                // k is a loop constant after unrolling, so every range has constant bounds and each select is a 2:1
                // mux on k_second instead of a 512-bit variable shifter.
                const ap_int<8> sc5 = k_second ? ap_int<8>(scales.range(32 + 8 * (4 + k) + 7, 32 + 8 * (4 + k)))
                                               : ap_int<8>(scales.range(32 + 8 * k + 7, 32 + 8 * k));
                const ap_int<8> sc6a = k_second ? ap_int<8>(scales.range(32 + 8 * (8 + 2 * k) + 7, 32 + 8 * (8 + 2 * k)))
                                                : ap_int<8>(scales.range(32 + 8 * (2 * k) + 7, 32 + 8 * (2 * k)));
                const ap_int<8> sc6b = k_second ? ap_int<8>(scales.range(40 + 8 * (8 + 2 * k) + 7, 40 + 8 * (8 + 2 * k)))
                                                : ap_int<8>(scales.range(40 + 8 * (2 * k) + 7, 40 + 8 * (2 * k)));
                const ap_int<8> sc_a = k6 ? sc6a : sc5;
                const ap_int<8> sc_b = k6 ? sc6b : sc5;
                const ap_uint<6> mn5 = k_second ? ap_uint<6>(scales.range(96 + 8 * (4 + k) + 5, 96 + 8 * (4 + k)))
                                                : ap_uint<6>(scales.range(96 + 8 * k + 5, 96 + 8 * k));
                // Two scale products per lane (Q5_K: the same scale twice) and the Q5_K min product, in DSPs.
#if GEMVQ_INT_SCALE
                const lane_scaled scaled = scale_lane_fixed(kq, dot, scales.range(si * 16 + 15, si * 16),
                                                            xs.range(xb * 32 + 15, xb * 32), active, sc_a, sc_b,
                                                            ap_int<20>(half0), ap_int<20>(half1));
                lane_term[k] = scaled.term;
                lane_s[k] = scaled.s_sum;
#else
                const int scaled_a = int(sc_a * ap_int<20>(half0));
#pragma HLS BIND_OP variable=scaled_a op=mul impl=dsp
                const int scaled_b = int(sc_b * ap_int<20>(half1));
#pragma HLS BIND_OP variable=scaled_b op=mul impl=dsp
                lane_s[k] = scaled_a + scaled_b;
#endif
                const int min_product = int(mn5 * ap_int<13>(lane_xsum[k]));
#pragma HLS BIND_OP variable=min_product op=mul impl=dsp
                lane_m[k] = k5 ? min_product : 0;
            }
            // The lanes of one word are summed in FP32 and converted once: a single variable shift per word
            // instead of four (~2.5k LUT less per engine). The row sum stays order-independent fixed point.
#if GEMVQ_INT_SCALE
            acc_t word_term = (lane_term[0] + lane_term[1]) + (lane_term[2] + lane_term[3]);
            // Q4_1 (ggml: (d4 * d8) * sumi + m4 * s8 per block): the four min terms summed in FP32, one conversion.
            if (q41) word_term += float_to_fixed((lane_min[0] + lane_min[1]) + (lane_min[2] + lane_min[3]));
            if (kq) {
                const int s_sum = (k_second ? k_s : 0) + (lane_s[0] + lane_s[1]) + (lane_s[2] + lane_s[3]);
                const int m_sum = (k_second ? k_m : 0) + (lane_m[0] + lane_m[1]) + (lane_m[2] + lane_m[3]);
                word_term = k_second ? scale_k_fixed(s_sum, m_sum, scales.range(15, 0), scales.range(31, 16), xs.range(31, 0))
                                     : acc_t(0);
                k_s = s_sum;
                k_m = m_sum;
                k_second = !k_second;
            }
            // Only this add is in the recurrence; everything before it is registered.
            const acc_t start = first ? acc_t(0) : racc[row];
            const acc_t next = (block_in_row == 0 ? start : accumulator) + stage_reg(word_term);
#else
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
            const acc_t base = block_in_row == 0 ? (first ? acc_t(0) : racc[row]) : accumulator;
            const acc_t next = base + float_to_fixed(word_term);
#endif
            accumulator = next;
            block_in_row += bpw;
            group_left -= bpw;
            if (group_left == 0) {
                // Packed Q6_K: the next super-block of the group needs only its high-bit word.
                const bool group_end = k6_slot == 2U || block_in_row == blocks_per_row;
                k6_slot = group_end ? 0U : k6_slot + 1U;
                header_left = k6 && !group_end ? 1U : headers;
            }
            if (block_in_row == blocks_per_row) {
                if (last) {
                    row_results.write(fixed_to_float(next));
                } else {
                    racc[row] = next;
                }
                ++row;
                block_in_row = 0;
            }
        }
    }
}

} // namespace
} // namespace gemvqw_engine
#endif
