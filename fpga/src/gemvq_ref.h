// SPDX-License-Identifier: Apache-2.0 AND MIT
// Portions of the reference arithmetic and of the format handling follow ggml / llama.cpp (MIT License,
// Copyright (c) 2023-2026 The ggml authors), see THIRD-PARTY-NOTICES.md.
// CPU reference for the gemvq kernels (plain C++): ggml's generic quantize_row_q8_0 + vec_dot_q{4,8}_0_q8_0 with
// FP32 accumulation per block, on the original GGUF bytes. Shared by the kernel model and the host code.
#ifndef FPGA_LLAMA_GGUF_GEMVQ_REF_H
#define FPGA_LLAMA_GGUF_GEMVQ_REF_H
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gemvq_ref {

inline float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1fU, mant = h & 0x3ffU, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -1;
            do { mant <<= 1; ++e; } while (!(mant & 0x400U));
            bits = sign | uint32_t(127 - 15 - e) << 23 | (mant & 0x3ffU) << 13;
        }
    } else if (exp == 31) bits = sign | 0x7f800000U | mant << 13;
    else bits = sign | (exp - 15 + 127) << 23 | mant << 13;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t float_to_half(float f) {  // round to nearest even, normal/subnormal range only
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint16_t sign = (bits >> 16) & 0x8000U;
    int exp = int((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffffU;
    if (exp >= 31) return sign | 0x7c00U;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x800000U;
        const int shift = 14 - exp;
        uint32_t r = mant >> shift, rem = mant & ((1U << shift) - 1), half = 1U << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return sign | r;
    }
    uint32_t r = mant >> 13, rem = mant & 0x1fffU;
    uint32_t out = uint32_t(exp) << 10 | r;
    if (rem > 0x1000U || (rem == 0x1000U && (r & 1))) ++out;
    return sign | out;
}

// ggml nearest_int: round half to even through the FP32 addition of 1.5 * 2^23.
inline int nearest_int(float value) {
    const float biased = value + 12582912.f;
    int bits;
    std::memcpy(&bits, &biased, sizeof(int));
    return (bits & 0x007fffff) - 0x00400000;
}

// ggml quantize_row_q8_K_ref: per 256 values the one with the largest magnitude (first on ties) gives
// iscale = -127 / max, q = min(127, nearest_int(iscale * x)), d = 1 / iscale (d = 0 for an all-zero block).
inline void quantize_q8_k(const float * x, unsigned n, std::vector<int8_t> & q, std::vector<float> & d) {
    q.assign(n, 0);
    d.assign(n / 256, 0.0f);
    for (unsigned b = 0; b < n / 256; ++b) {
        float max = 0, amax = 0;
        for (unsigned j = 0; j < 256; ++j) {
            const float ax = std::fabs(x[b * 256 + j]);
            if (ax > amax) {
                amax = ax;
                max = x[b * 256 + j];
            }
        }
        if (!amax) continue;
        const float iscale = -127.f / max;
        for (unsigned j = 0; j < 256; ++j) q[b * 256 + j] = int8_t(std::min(127, nearest_int(iscale * x[b * 256 + j])));
        d[b] = 1 / iscale;
    }
}

// Sub-block scale and min of a Q5_K/Q4_K super-block (ggml get_scale_min_k4).
inline void scale_min_k4(unsigned j, const uint8_t * q, int & sc, int & m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// One Q5_K row (nsb super-blocks of 176 bytes) against Q8_K activations, as ggml_vec_dot_q5_K_q8_K_generic.
inline float dot_q5_k(const uint8_t * row, const int8_t * q8, const float * d8, unsigned nsb, double & magnitude) {
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float sumf = 0;
    for (unsigned sb = 0; sb < nsb; ++sb) {
        const uint8_t * block = row + size_t(sb) * 176;
        const uint8_t * scales = block + 4, * qh = block + 16, * qs = block + 48;
        const int8_t * y = q8 + size_t(sb) * 256;
        int a[256];
        for (unsigned s = 0; s < 8; ++s) {
            for (unsigned e = 0; e < 32; ++e) {
                const uint8_t byte = qs[32 * (s / 2) + e];
                a[32 * s + e] = ((s & 1) ? (byte >> 4) : (byte & 15)) + (((qh[e] >> s) & 1) ? 16 : 0);
            }
        }
        int sumi = 0;
        for (unsigned j = 0; j < 16; ++j) {
            int bsum = 0, sc, m;
            for (unsigned i = 0; i < 16; ++i) bsum += y[16 * j + i];
            scale_min_k4(j / 2, scales, sc, m);
            sumi += bsum * m;
        }
        int aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (unsigned j = 0; j < 8; ++j) {
            int sc, m;
            scale_min_k4(j, scales, sc, m);
            for (unsigned i = 0; i < 32; ++i) aux32[i % 8] += sc * (y[32 * j + i] * a[32 * j + i]);
        }
        const float d = half_to_float(uint16_t(block[0] | block[1] << 8)) * d8[sb];
        double part = 0;
        for (unsigned l = 0; l < 8; ++l) {
            sums[l] += d * float(aux32[l]);
            part += double(aux32[l]);
        }
        const float dmin = half_to_float(uint16_t(block[2] | block[3] << 8)) * d8[sb];
        sumf -= dmin * float(sumi);
        magnitude += std::fabs(double(d) * part) + std::fabs(double(dmin) * sumi);
    }
    for (unsigned l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

// magnitude[r] = sum of |block terms| of row r, used for the tolerance of the fixed-point kernel sums.
inline std::vector<float> reference_q5_k(const uint8_t * w, const float * x, unsigned rows, unsigned cols,
                                         std::vector<double> & magnitude) {
    std::vector<int8_t> q;
    std::vector<float> d;
    quantize_q8_k(x, cols, q, d);
    std::vector<float> y(rows);
    magnitude.assign(rows, 0.0);
    for (unsigned r = 0; r < rows; ++r) y[r] = dot_q5_k(w + size_t(r) * (cols / 256) * 176, q.data(), d.data(), cols / 256, magnitude[r]);
    return y;
}

// One Q6_K row against Q8_K activations, as ggml_vec_dot_q6_K_q8_K_generic.
inline float dot_q6_k(const uint8_t * row, const int8_t * q8, const float * d8, unsigned nsb, double & magnitude) {
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (unsigned sb = 0; sb < nsb; ++sb) {
        const uint8_t * block = row + size_t(sb) * 210;
        const uint8_t * ql = block, * qh = block + 128;
        const int8_t * scales = reinterpret_cast<const int8_t *>(block + 192);
        const int8_t * y = q8 + size_t(sb) * 256;
        int a[256];
        for (unsigned n = 0; n < 2; ++n) {
            for (unsigned l = 0; l < 32; ++l) {
                const uint8_t h = qh[32 * n + l];
                a[128 * n + l] = int8_t((ql[64 * n + l] & 0xF) | (((h >> 0) & 3) << 4)) - 32;
                a[128 * n + l + 32] = int8_t((ql[64 * n + l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32;
                a[128 * n + l + 64] = int8_t((ql[64 * n + l] >> 4) | (((h >> 4) & 3) << 4)) - 32;
                a[128 * n + l + 96] = int8_t((ql[64 * n + l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
            }
        }
        int aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (unsigned j = 0; j < 16; ++j) {
            for (unsigned i = 0; i < 16; ++i) aux32[i % 8] += scales[j] * (y[16 * j + i] * a[16 * j + i]);
        }
        const float d = half_to_float(uint16_t(block[208] | block[209] << 8)) * d8[sb];
        double part = 0;
        for (unsigned l = 0; l < 8; ++l) {
            sums[l] += d * float(aux32[l]);
            part += double(aux32[l]);
        }
        magnitude += std::fabs(double(d) * part);
    }
    float sumf = 0;
    for (unsigned l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

inline std::vector<float> reference_q6_k(const uint8_t * w, const float * x, unsigned rows, unsigned cols,
                                         std::vector<double> & magnitude) {
    std::vector<int8_t> q;
    std::vector<float> d;
    quantize_q8_k(x, cols, q, d);
    std::vector<float> y(rows);
    magnitude.assign(rows, 0.0);
    for (unsigned r = 0; r < rows; ++r) y[r] = dot_q6_k(w + size_t(r) * (cols / 256) * 210, q.data(), d.data(), cols / 256, magnitude[r]);
    return y;
}

// Q4_1 against Q8_1 activations: quantize_row_q8_1_ref (d = amax / 127 as FP16, q = roundf(x / d), s = FP16(sum q * d))
// and ggml_vec_dot_q4_1_q8_1_generic (sum += (d4 * d8) * sumi + m4 * s8).
inline std::vector<float> reference_q4_1(const uint8_t * w, const float * x, unsigned rows, unsigned cols,
                                         std::vector<double> & magnitude) {
    const unsigned nb = cols / 32;
    std::vector<int8_t> q(cols);
    std::vector<float> d8(nb), s8(nb);
    for (unsigned b = 0; b < nb; ++b) {
        float amax = 0;
        for (unsigned j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127, id = d ? 1.0f / d : 0.0f;
        int sum = 0;
        for (unsigned j = 0; j < 32; ++j) {
            q[b * 32 + j] = int8_t(std::roundf(x[b * 32 + j] * id));
            sum += q[b * 32 + j];
        }
        d8[b] = half_to_float(float_to_half(d));
        s8[b] = half_to_float(float_to_half(float(sum) * d));
    }
    std::vector<float> y(rows);
    magnitude.assign(rows, 0.0);
    for (unsigned r = 0; r < rows; ++r) {
        float sum = 0;
        for (unsigned b = 0; b < nb; ++b) {
            const uint8_t * blk = &w[(size_t(r) * nb + b) * 20];
            int sumi = 0;
            for (unsigned j = 0; j < 16; ++j) sumi += (blk[4 + j] & 0x0F) * q[b * 32 + j] + (blk[4 + j] >> 4) * q[b * 32 + j + 16];
            const float dw = half_to_float(uint16_t(blk[0] | blk[1] << 8)), mw = half_to_float(uint16_t(blk[2] | blk[3] << 8));
            const float term = (dw * d8[b]) * float(sumi) + mw * s8[b];
            sum += term;
            magnitude[r] += std::fabs((dw * d8[b]) * float(sumi)) + std::fabs(mw * s8[b]);
        }
        y[r] = sum;
    }
    return y;
}

inline std::vector<float> reference(const uint8_t * w, const float * x, unsigned rows, unsigned cols, unsigned format,
                                    std::vector<double> & magnitude) {
    if (format == 13U) return reference_q5_k(w, x, rows, cols, magnitude);
    if (format == 14U) return reference_q6_k(w, x, rows, cols, magnitude);
    if (format == 3U) return reference_q4_1(w, x, rows, cols, magnitude);
    const unsigned nb = cols / 32, bb = format == 2U ? 18 : 34;
    std::vector<int8_t> q(cols);
    std::vector<float> d(nb);
    for (unsigned b = 0; b < nb; ++b) {
        float amax = 0;
        for (unsigned j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        const float s = amax / 127.0f, id = s ? 1.0f / s : 0.0f;
        d[b] = half_to_float(float_to_half(s));
        for (unsigned j = 0; j < 32; ++j) q[b * 32 + j] = int8_t(std::roundf(x[b * 32 + j] * id));
    }
    std::vector<float> y(rows);
    magnitude.assign(rows, 0.0);
    for (unsigned r = 0; r < rows; ++r) {
        float sum = 0;
        for (unsigned b = 0; b < nb; ++b) {
            const uint8_t * blk = &w[(size_t(r) * nb + b) * bb];
            int sumi = 0;
            for (unsigned j = 0; j < 32; ++j) {
                const int wv = format == 2U ? int(j < 16 ? (blk[2 + j] & 15) : (blk[2 + j - 16] >> 4)) - 8
                                                    : int(int8_t(blk[2 + j]));
                sumi += wv * q[b * 32 + j];
            }
            const float term = float(sumi) * (half_to_float(uint16_t(blk[0] | blk[1] << 8)) * d[b]);
            sum += term;
            magnitude[r] += std::fabs(term);
        }
        y[r] = sum;
    }
    return y;
}

// Kernel rows are fixed-point sums of the same block terms, so they may differ from the FP32 reference sum by
// rounding of the terms' sum; the bound scales with the row's term magnitudes.
inline bool within_tolerance(float got, float ref, double magnitude) {
    const double err = std::fabs(double(got) - double(ref));
    return err <= 1e-6 + 1e-5 * std::fabs(double(ref)) + 1e-6 * magnitude;
}

// Fused SwiGLU of two GEMV rows (g, u with their references and term magnitudes): the GEMV bounds above carried
// through y = silu(g) * u (|silu'| <= 1.1), plus the element-wise rounding of silu_mul.
inline bool within_swiglu(float got, float g_ref, double g_mag, float u_ref, double u_mag) {
    const double tol_g = 1e-6 + 1e-5 * std::fabs(double(g_ref)) + 1e-6 * g_mag;
    const double tol_u = 1e-6 + 1e-5 * std::fabs(double(u_ref)) + 1e-6 * u_mag;
    const double silu = double(g_ref) / (1.0 + std::exp(-double(g_ref)));
    const double ref = silu * double(u_ref);
    const double err = std::fabs(double(got) - ref);
    return err <= 1.1 * tol_g * (std::fabs(double(u_ref)) + tol_u) + std::fabs(silu) * tol_u + 1e-6 +
                      1e-5 * std::fabs(ref);
}

// Element-wise references following ggml-cpu: rms_norm sums x^2 in double, scale = 1 / sqrtf(mean + eps), then
// ggml_mul with the weight; silu(x) = x / (1 + expf(-x)).
inline std::vector<float> rmsnorm(const float * x, const float * w, unsigned n, float eps) {
    double sum = 0.0;
    for (unsigned i = 0; i < n; ++i) sum += double(x[i]) * double(x[i]);
    const float scale = 1.0f / std::sqrt(float(sum / n) + eps);
    std::vector<float> y(n);
    for (unsigned i = 0; i < n; ++i) y[i] = (x[i] * scale) * w[i];
    return y;
}

inline std::vector<float> add(const float * a, const float * b, unsigned n) {
    std::vector<float> y(n);
    for (unsigned i = 0; i < n; ++i) y[i] = a[i] + b[i];
    return y;
}

inline std::vector<float> silu_mul(const float * g, const float * u, unsigned n) {
    std::vector<float> y(n);
    for (unsigned i = 0; i < n; ++i) y[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
    return y;
}

// RMSNORM opcode with rows and flags (bit0 L2: ggml_l2_norm, scale = 1 / max(sqrt(sum), eps); bit1: no weight).
inline std::vector<float> rownorm(const float * x, const float * w, unsigned n, unsigned rows, float eps,
                                  unsigned flags) {
    std::vector<float> y(size_t(n) * rows);
    for (unsigned r = 0; r < rows; ++r) {
        const float * xr = x + size_t(r) * n;
        double sum = 0.0;
        for (unsigned i = 0; i < n; ++i) sum += double(xr[i] * xr[i]);
        const float scale = (flags & 1U) ? 1.0f / std::fmax(std::sqrt(float(sum)), eps)
                                         : 1.0f / std::sqrt(float(sum / n) + eps);
        for (unsigned i = 0; i < n; ++i) {
            y[size_t(r) * n + i] = (flags & 2U) ? xr[i] * scale : (xr[i] * scale) * w[i];
        }
    }
    return y;
}

inline std::vector<float> mul(const float * a, const float * b, unsigned n) {
    std::vector<float> y(n);
    for (unsigned i = 0; i < n; ++i) y[i] = a[i] * b[i];
    return y;
}

// kind 1 SiLU, 2 sigmoid, 3 softplus, as ggml-cpu's unary ops.
inline std::vector<float> unary(const float * x, unsigned n, unsigned kind) {
    std::vector<float> y(n);
    for (unsigned i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = kind == 1 ? v / (1.0f + std::exp(-v))
             : kind == 2 ? 1.0f / (1.0f + std::exp(-v))
                         : (v > 20.0f ? v : std::log(1.0f + std::exp(v)));
    }
    return y;
}

// ggml_ssm_conv for one token (d_conv 4) on the window [state (3 per channel), x], optional SiLU; returns y and
// the new state (last three window entries per channel).
inline std::vector<float> conv(const float * state, const float * x, const float * w, unsigned ch, bool silu,
                               std::vector<float> & new_state) {
    std::vector<float> y(ch);
    new_state.resize(size_t(ch) * 3);
    for (unsigned c = 0; c < ch; ++c) {
        const float in[4] = {state[c * 3], state[c * 3 + 1], state[c * 3 + 2], x[c]};
        float sum = 0.0f;
        for (unsigned t = 0; t < 4; ++t) sum += in[t] * w[c * 4 + t];
        y[c] = silu ? sum / (1.0f + std::exp(-sum)) : sum;
        for (unsigned t = 0; t < 3; ++t) new_state[c * 3 + t] = in[t + 1];
    }
    return y;
}

// ggml GATED_DELTA_NET, one token, scalar gate, S = 128: state[h][j][i] (row j = column j of S); q, k: [hk][128],
// v: [hv][128]; key head of value head h is h % hk. Updates state in place and returns y [hv][128].
// y_mag / s_mag: sums of |terms| for a rounding-aware tolerance.
inline std::vector<float> gdn(std::vector<float> & state, const float * q, const float * k, const float * v,
                              const float * g, const float * beta, unsigned hv, unsigned hk, float scale,
                              std::vector<double> & y_mag, std::vector<double> & s_mag) {
    const unsigned S = 128;
    std::vector<float> y(size_t(hv) * S);
    y_mag.assign(y.size(), 0.0);
    s_mag.assign(state.size(), 0.0);
    for (unsigned h = 0; h < hv; ++h) {
        const float * qh = q + (h % hk) * S, * kh = k + (h % hk) * S;
        const float decay = std::exp(g[h]);
        for (unsigned j = 0; j < S; ++j) {
            float * row = &state[(size_t(h) * S + j) * S];
            double dmag = 0.0;
            float dot = 0.0f;
            for (unsigned i = 0; i < S; ++i) {
                row[i] *= decay;
                dot += row[i] * kh[i];
                dmag += std::fabs(double(row[i]) * kh[i]);
            }
            const float delta = (v[h * S + j] - dot) * beta[h];
            const double delta_mag = (std::fabs(double(v[h * S + j])) + dmag) * std::fabs(double(beta[h]));
            float out = 0.0f;
            double omag = 0.0;
            for (unsigned i = 0; i < S; ++i) {
                s_mag[(size_t(h) * S + j) * S + i] = std::fabs(double(row[i])) + std::fabs(double(kh[i])) * delta_mag;
                row[i] += kh[i] * delta;
                out += row[i] * qh[i];
                omag += (std::fabs(double(row[i])) + std::fabs(double(kh[i])) * delta_mag) * std::fabs(double(qh[i]));
            }
            y[h * S + j] = out * scale;
            y_mag[h * S + j] = omag * scale;
        }
    }
    return y;
}

// Element-wise results differ from the reference only by FP rounding (sum order, exp implementation).
inline bool within_elementwise(float got, float ref) {
    return std::fabs(double(got) - double(ref)) <= 1e-6 + 2e-5 * std::fabs(double(ref));
}

} // namespace gemvq_ref
#endif
