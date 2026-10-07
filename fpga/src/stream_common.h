// SPDX-License-Identifier: Apache-2.0
// Shared building blocks of the gemvqc16 kernel: FP16 to FP32 conversion, the fixed-point row accumulator and the
// sequential weight burst reader.
#pragma once
#include <ap_int.h>
#include <ap_fixed.h>
#ifdef __SYNTHESIS__
#include <hls_math.h>
#endif
#include <hls_stream.h>

// Synthesis-safe FP16-to-FP32 conversion
inline float half_bits_to_float(ap_uint<16> bits) {
#pragma HLS INLINE
    ap_uint<1> sign = bits[15];
    ap_uint<5> exp  = bits.range(14, 10);
    ap_uint<10> mant = bits.range(9, 0);

    ap_uint<32> res_bits;
    res_bits[31] = sign;

    if (exp == 0) {
        if (mant == 0) {
            res_bits.range(30, 0) = 0;
        } else {
            // Use an 11-bit temporary so the implicit leading bit is retained.
            // A data-dependent while loop here would stop HLS from pipelining
            // any loop that calls this function, so locate the leading one
            // with a fixed, fully unrolled priority scan instead.
            int leading_one = 0;
            for (int i = 0; i < 10; ++i) {
#pragma HLS UNROLL
                if (mant[i]) leading_one = i;
            }
            const int shift = 10 - leading_one;
            const ap_uint<11> normalized_mant = ap_uint<11>(mant) << shift;
            res_bits.range(30, 23) = 127 - 14 - shift;
            // Widen before shifting: an ap_uint<10> shift expression can
            // otherwise truncate the subnormal mantissa before assignment.
            const ap_uint<23> mantissa_bits = ap_uint<23>(normalized_mant.range(9, 0)) << 13;
            res_bits.range(22, 0) = mantissa_bits;
        }
    } else if (exp == 0x1F) {
        res_bits.range(30, 23) = 0xFF;
        res_bits.range(22, 0) = (mant != 0) ? (ap_uint<23>)0x400000 : (ap_uint<23>)0;
    } else {
        res_bits.range(30, 23) = (exp - 15) + 127;
        res_bits.range(22, 0) = (ap_uint<23>)mant << 13;
    }

    union {
        unsigned int i;
        float f;
    } conv;
    conv.i = res_bits.to_uint();
    return conv.f;
}

// Row sums are accumulated in fixed point (LSB 2^-40) so the per-block FP32
// products need no loop-carried FP addition and the result does not depend on
// the summation order. 80 bits cover |sum| < 2^38 with margin.
typedef ap_int<80> acc_t;
constexpr int ACC_FRAC_BITS = 40;

static acc_t float_to_fixed(float value) {
#pragma HLS INLINE
    union {
        unsigned int i;
        float f;
    } conv;
    conv.f = value;
    const ap_uint<32> bits = conv.i;
    const bool negative = bits[31];
    const int exponent = bits.range(30, 23).to_int();
    const ap_uint<24> mantissa = ap_uint<24>(bits.range(22, 0)) | (ap_uint<24>(1) << 23);
    // value = mantissa * 2^(exponent - 150); fixed LSB 2^-40.
    const int shift = exponent - 150 + ACC_FRAC_BITS;
    ap_uint<79> magnitude = 0;
    if (exponent == 0) {
        magnitude = 0;
    } else if (exponent == 255 || shift > 54) {
        magnitude = ap_uint<79>(1) << 78;
    } else if (shift >= 0) {
        magnitude = ap_uint<79>(mantissa) << shift;
    } else if (shift > -24) {
        magnitude = ap_uint<79>(mantissa) >> (-shift);
    }
    const acc_t signed_magnitude = acc_t(magnitude);
    return negative ? acc_t(-signed_magnitude) : signed_magnitude;
}

static float fixed_to_float(acc_t value) {
#pragma HLS INLINE
    const bool negative = value < 0;
    const ap_uint<80> magnitude = negative ? ap_uint<80>(-value) : ap_uint<80>(value);
    if (magnitude == 0) {
        return 0.0f;
    }
    const int lz = magnitude.countLeadingZeros();
    const ap_uint<80> normalized = magnitude << lz;
    ap_uint<25> mantissa = normalized.range(79, 56);
    const bool guard = normalized[55];
    const bool sticky = normalized.range(54, 0) != 0;
    int biased = (79 - lz) - ACC_FRAC_BITS + 127;
    if (guard && (sticky || mantissa[0])) {
        mantissa += 1;
    }
    if (mantissa[24]) {
        mantissa >>= 1;
        ++biased;
    }
    ap_uint<32> bits = 0;
    bits[31] = negative;
    if (biased >= 255) {
        bits.range(30, 23) = 0xFF;
    } else {
        bits.range(30, 23) = biased;
        bits.range(22, 0) = mantissa.range(22, 0);
    }
    union {
        unsigned int i;
        float f;
    } conv;
    conv.i = bits.to_uint();
    return conv.f;
}


// Pure sequential burst reader: no conditions, so the AXI adapter can issue
// long outstanding bursts and hide the HBM latency.
static void read_words(
    const ap_uint<512> *matrix_w,
    unsigned long long matrix_offset_words,
    unsigned int word_count,
    hls::stream<ap_uint<512>> &words) {
    const ap_uint<512> *source = matrix_w + matrix_offset_words;
    READ_WORDS: for (unsigned int i = 0; i < word_count; ++i) {
#pragma HLS PIPELINE II=1
        words.write(source[i]);
    }
}
