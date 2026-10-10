// SPDX-License-Identifier: Apache-2.0 AND MIT
// Portions of the reference arithmetic and of the format handling follow ggml / llama.cpp (MIT License,
// Copyright (c) 2023-2026 The ggml authors), see THIRD-PARTY-NOTICES.md.
// Host-side word-aligned weight layout for the gemvqw kernels (plain C++, no HLS types).
//
// Q4_0/Q8_0: every row is split into groups of up to 32 GGUF blocks. A group is one 64-byte word holding the
// blocks' FP16 scales (scale k at bytes 2k..2k+1, unused slots zero), followed by the blocks' quant bytes in original
// order: Q8_0 32 bytes per block (2 blocks per word), Q4_0 16 bytes per block (4 blocks per word, ggml nibble order).
// The bits are the original GGUF bits; only scales and quants are separated. cols must be a multiple of 128,
// so the last group of a row has a multiple of 4 blocks and fills whole words.
//
// Q4_1 (ggml type 3): like Q4_0, but each group has two header words, the blocks' FP16 d (bytes 2k..2k+1) and
// their FP16 m; the quant words follow in Q4_0 order (no extra space).
//
// Q6_K (ggml type 14, cols a multiple of 256): per super-block a meta word (d at bytes 0-1, bytes 2-3 zero, the 16
// int8 scales of the 16-weight sub-blocks at bytes 4..19), a high-bit word (the upper 2 bits of weight e at bits
// 2e..2e+1) and two quant words with the lower 4 bits in the Q4_0 nibble order (weight 128w + 32k + e of the
// super-block in lane k of quant word w). 256 bytes per super-block instead of 210.
// Packed Q6_K (q6k_packed, the gemvqc16 layout since feature bit 13): the super-blocks of a row in groups of up to
// three; a group starts with one word holding the metas of its super-blocks (meta j at bytes 20j..20j+19, same
// format), then per super-block the high-bit word and the two quant words. 10 words per three super-blocks
// (640 bytes, GGUF: 630) instead of 12.
//
// Q5_K (ggml type 13, cols a multiple of 256): per 256-weight super-block one meta word and two quant words.
// Meta word: d (bytes 0-1), dmin (2-3), the 8 sub-block scales (4-11) and mins (12-19) decoded from their 6-bit
// packing, and the high bits as one 32-bit mask per sub-block (bytes 20+4s..23+4s, bit e = high bit of element e).
// Quant word w holds sub-blocks 4w..4w+3 in the Q4_0 nibble order: sub-block 4w+k at bytes 16k..16k+15, element
// e < 16 in the low nibble of byte e, e >= 16 in the high nibble of byte e-16. Only the packing changes; all scale
// and quant values are the original ones (192 bytes per super-block instead of 176).
#ifndef FPGA_LLAMA_GGUF_GEMVQW_LAYOUT_H
#define FPGA_LLAMA_GGUF_GEMVQW_LAYOUT_H
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gemvqw_layout {

// Q4_0 / Q4_1 / Q8_0: bytes of one 32-weight block.
inline unsigned block_bytes(unsigned format) { return format == 2U ? 18U : (format == 3U ? 20U : 34U); }
inline unsigned blocks_per_word(unsigned format) { return format == 8U ? 2U : 4U; }

// Bytes of one GGUF row.
inline size_t row_bytes(unsigned cols, unsigned format) {
    if (format == 13U) return size_t(cols / 256U) * 176U;
    if (format == 14U) return size_t(cols / 256U) * 210U;
    return size_t(cols / 32U) * block_bytes(format);
}

inline unsigned words_per_row(unsigned cols, unsigned format, bool q6k_packed = true) {
    if (format == 13U) return 3U * (cols / 256U);
    if (format == 14U) return q6k_packed ? (cols / 256U + 2U) / 3U + 3U * (cols / 256U) : 4U * (cols / 256U);
    const unsigned blocks = cols / 32U, bpw = blocks_per_word(format), headers = format == 3U ? 2U : 1U;
    const unsigned full = blocks / 32U, tail = blocks % 32U;
    return full * (headers + 32U / bpw) + (tail ? headers + tail / bpw : 0U);
}

inline void append_row_q6_k(const uint8_t * row, unsigned cols, std::vector<uint8_t> & out, bool packed) {
    size_t meta_word = 0;
    for (unsigned sb = 0; sb < cols / 256U; ++sb) {
        const uint8_t * block = row + size_t(sb) * 210U;
        const uint8_t * ql = block, * qh = block + 128, * scales = block + 192;
        if (packed && sb % 3U == 0) {  // the group's meta word
            meta_word = out.size();
            out.resize(meta_word + 64U, 0);
        }
        const size_t base = out.size() - (packed ? 64U : 0U);  // "meta word" of the old layout, shifted below
        out.resize(out.size() + (packed ? 192U : 256U), 0);
        uint8_t * meta = packed ? &out[meta_word + 20U * (sb % 3U)] : &out[base];
        std::memcpy(meta, block + 208, 2);
        std::memcpy(meta + 4, scales, 16);
        uint8_t * high = &out[base + 64U];
        for (unsigned e = 0; e < 256; ++e) {  // ggml dequantize_row_q6_K: half n, quarter c, position l
            const unsigned n = e / 128U, c = (e % 128U) / 32U, l = e % 32U;
            const unsigned bits = (qh[32U * n + l] >> (2U * c)) & 3U;
            high[e / 4U] |= uint8_t(bits << (2U * (e % 4U)));
            const uint8_t byte = ql[64U * n + 32U * (c & 1U) + l];
            const uint8_t nibble = c < 2 ? uint8_t(byte & 15U) : uint8_t(byte >> 4);
            uint8_t * lane = &out[base + 128U + 64U * n + 16U * c];  // high word + 1 + n
            if (l < 16) lane[l] |= nibble;
            else lane[l - 16] |= uint8_t(nibble << 4);
        }
    }
}

inline void append_row_q5_k(const uint8_t * row, unsigned cols, std::vector<uint8_t> & out) {
    for (unsigned sb = 0; sb < cols / 256U; ++sb) {
        const uint8_t * block = row + size_t(sb) * 176U;
        const uint8_t * scales = block + 4, * qh = block + 16, * qs = block + 48;
        const size_t base = out.size();
        out.resize(base + 192U, 0);
        uint8_t * meta = &out[base];
        std::memcpy(meta, block, 4);
        for (unsigned j = 0; j < 8; ++j) {  // ggml get_scale_min_k4
            meta[4 + j] = j < 4 ? uint8_t(scales[j] & 63) : uint8_t((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
            meta[12 + j] = j < 4 ? uint8_t(scales[j + 4] & 63) : uint8_t((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
            for (unsigned e = 0; e < 32; ++e) meta[20 + 4 * j + e / 8] |= uint8_t(((qh[e] >> j) & 1U) << (e % 8));
        }
        for (unsigned s = 0; s < 8; ++s) {
            uint8_t * lane = &out[base + 64U * (1U + s / 4U) + 16U * (s % 4U)];
            for (unsigned e = 0; e < 32; ++e) {
                const uint8_t byte = qs[32U * (s / 2U) + e];
                const uint8_t nibble = (s & 1U) ? uint8_t(byte >> 4) : uint8_t(byte & 15U);
                if (e < 16) lane[e] |= nibble;
                else lane[e - 16] |= uint8_t(nibble << 4);
            }
        }
    }
}

// Appends one GGUF row in word layout to out (64-byte words).
inline void append_row(const uint8_t * row, unsigned cols, unsigned format, std::vector<uint8_t> & out,
                       bool q6k_packed = true) {
    if (format == 13U) {
        append_row_q5_k(row, cols, out);
        return;
    }
    if (format == 14U) {
        append_row_q6_k(row, cols, out, q6k_packed);
        return;
    }
    const unsigned blocks = cols / 32U, bb = block_bytes(format), head = format == 3U ? 4U : 2U, qb = bb - head;
    const unsigned header_bytes = format == 3U ? 128U : 64U;  // Q4_1: d word and m word
    for (unsigned first = 0; first < blocks; first += 32U) {
        const unsigned count = blocks - first < 32U ? blocks - first : 32U;
        const size_t scale_word = out.size();
        out.resize(out.size() + header_bytes + size_t(count) * qb, 0);
        for (unsigned k = 0; k < count; ++k) {
            const uint8_t * block = row + size_t(first + k) * bb;
            std::memcpy(&out[scale_word + 2U * k], block, 2);
            if (format == 3U) std::memcpy(&out[scale_word + 64U + 2U * k], block + 2, 2);
            std::memcpy(&out[scale_word + header_bytes + size_t(k) * qb], block + head, qb);
        }
    }
}

// Splits a row-major GGUF matrix over `engines` arenas: row r goes to engine r % engines.
inline std::vector<std::vector<uint8_t>> repack(const uint8_t * matrix, unsigned rows, unsigned cols, unsigned format,
                                                unsigned engines, bool q6k_packed = true) {
    std::vector<std::vector<uint8_t>> arenas(engines);
    const size_t bytes = row_bytes(cols, format);
    for (unsigned r = 0; r < rows; ++r) append_row(matrix + r * bytes, cols, format, arenas[r % engines], q6k_packed);
    return arenas;
}

} // namespace gemvqw_layout
#endif
