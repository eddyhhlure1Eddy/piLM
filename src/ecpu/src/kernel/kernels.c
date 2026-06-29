#include "../include/kernel/api.h"
#include "../include/device/api.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

static inline float bf16_to_f32(uint16_t x) {
    uint32_t bits = ((uint32_t)x) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint16_t f32_to_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (uint16_t)(bits >> 16);
}

static int w8a16_preconvert_enabled(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    const char *v = getenv("ECPU_W8A16_PRECONVERT");
    cached = (v && strcmp(v, "1") == 0) ? 1 : 0;
    return cached;
}

static int w8a16_m_flat_enabled(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    const char *v = getenv("ECPU_W8A16_M_FLAT");
    cached = (v && strcmp(v, "1") == 0) ? 1 : 0;
    return cached;
}

static inline int8_t unpack_i4_signed(uint8_t packed, int high) {
    uint8_t nibble = high ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0x0Fu);
    return (int8_t)(nibble >= 8 ? (int)nibble - 16 : (int)nibble);
}

/* Forward declarations for helpers defined later in this file; the W8+q8 and
 * W4+q8 kernels below both use them. */
static float quantize_bf16_row_to_i8(const uint16_t *a, int8_t *out, size_t K);
#if defined(__AVX2__)
static inline int32_t hsum256_epi32(__m256i v);
#endif

#if defined(__AVX2__)
static inline float hsum256_ps(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline float dot_f32_f32_avx2(const float *a, const float *b, size_t K) {
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 av = _mm256_loadu_ps(a + k);
        __m256 bv = _mm256_loadu_ps(b + k);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    float out = hsum256_ps(acc);
    for (; k < K; k++) out += a[k] * b[k];
    return out;
}

static inline float dot_f32_i8_avx2(const float *a, const int8_t *w, size_t K) {
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 av = _mm256_loadu_ps(a + k);
        __m128i wb = _mm_loadl_epi64((const __m128i *)(w + k));
        __m256i wi = _mm256_cvtepi8_epi32(wb);
        __m256 wf = _mm256_cvtepi32_ps(wi);
        acc = _mm256_fmadd_ps(av, wf, acc);
    }
    float out = hsum256_ps(acc);
    for (; k < K; k++) out += a[k] * (float)w[k];
    return out;
}

static inline float dot_bf16_i8_avx2(const uint16_t *a, const int8_t *w, size_t K) {
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m128i ab = _mm_loadu_si128((const __m128i *)(a + k));
        __m256i ai = _mm256_cvtepu16_epi32(ab);
        ai = _mm256_slli_epi32(ai, 16);
        __m256 af = _mm256_castsi256_ps(ai);
        __m128i wb = _mm_loadl_epi64((const __m128i *)(w + k));
        __m256i wi = _mm256_cvtepi8_epi32(wb);
        __m256 wf = _mm256_cvtepi32_ps(wi);
        acc = _mm256_fmadd_ps(af, wf, acc);
    }
    float out = hsum256_ps(acc);
    for (; k < K; k++) out += bf16_to_f32(a[k]) * (float)w[k];
    return out;
}

static inline float dot_bf16_i4_packed_avx2(const uint16_t *a, const uint8_t *w, size_t K) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m128i seven = _mm_set1_epi8(7);
    const __m128i sixteen = _mm_set1_epi8(16);
    size_t k = 0;
    for (; k + 16 <= K; k += 16) {
        __m128i packed = _mm_loadl_epi64((const __m128i *)(w + k / 2));
        __m128i lo = _mm_and_si128(packed, mask4);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask4);
        __m128i q = _mm_unpacklo_epi8(lo, hi);
        __m128i neg = _mm_cmpgt_epi8(q, seven);
        q = _mm_sub_epi8(q, _mm_and_si128(neg, sixteen));

        __m128i ab0 = _mm_loadu_si128((const __m128i *)(a + k));
        __m256i ai0 = _mm256_cvtepu16_epi32(ab0);
        ai0 = _mm256_slli_epi32(ai0, 16);
        __m256 af0 = _mm256_castsi256_ps(ai0);
        __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
        acc0 = _mm256_fmadd_ps(af0, wf0, acc0);

        __m128i q_hi = _mm_srli_si128(q, 8);
        __m128i ab1 = _mm_loadu_si128((const __m128i *)(a + k + 8));
        __m256i ai1 = _mm256_cvtepu16_epi32(ab1);
        ai1 = _mm256_slli_epi32(ai1, 16);
        __m256 af1 = _mm256_castsi256_ps(ai1);
        __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
        acc1 = _mm256_fmadd_ps(af1, wf1, acc1);
    }
    float out = hsum256_ps(_mm256_add_ps(acc0, acc1));
    for (; k + 1 < K; k += 2) {
        uint8_t packed = w[k / 2];
        int8_t w0 = unpack_i4_signed(packed, 0);
        int8_t w1 = unpack_i4_signed(packed, 1);
        out += bf16_to_f32(a[k + 0]) * (float)w0;
        out += bf16_to_f32(a[k + 1]) * (float)w1;
    }
    if (k < K) {
        int8_t w0 = unpack_i4_signed(w[k / 2], 0);
        out += bf16_to_f32(a[k]) * (float)w0;
    }
    return out;
}

static inline __m128i unpack_i4_16_signed_avx2(const uint8_t *w) {
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m128i seven = _mm_set1_epi8(7);
    const __m128i sixteen = _mm_set1_epi8(16);
    __m128i packed = _mm_loadl_epi64((const __m128i *)w);
    __m128i lo = _mm_and_si128(packed, mask4);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask4);
    __m128i q = _mm_unpacklo_epi8(lo, hi);
    __m128i neg = _mm_cmpgt_epi8(q, seven);
    return _mm_sub_epi8(q, _mm_and_si128(neg, sixteen));
}

static inline void dot_bf16_i4_8_packed_avx2(const uint16_t *a,
                                             const uint8_t *w0,
                                             const uint8_t *w1,
                                             const uint8_t *w2,
                                             const uint8_t *w3,
                                             const uint8_t *w4,
                                             const uint8_t *w5,
                                             const uint8_t *w6,
                                             const uint8_t *w7,
                                             size_t K,
                                             float *out) {
    const uint8_t *ws[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    __m256 acc_lo[8];
    __m256 acc_hi[8];
    for (int j = 0; j < 8; j++) {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
    }
    size_t k = 0;
    for (; k + 16 <= K; k += 16) {
        __m128i ab0 = _mm_loadu_si128((const __m128i *)(a + k));
        __m256i ai0 = _mm256_cvtepu16_epi32(ab0);
        ai0 = _mm256_slli_epi32(ai0, 16);
        __m256 af0 = _mm256_castsi256_ps(ai0);

        __m128i ab1 = _mm_loadu_si128((const __m128i *)(a + k + 8));
        __m256i ai1 = _mm256_cvtepu16_epi32(ab1);
        ai1 = _mm256_slli_epi32(ai1, 16);
        __m256 af1 = _mm256_castsi256_ps(ai1);

        for (int j = 0; j < 8; j++) {
            __m128i q = unpack_i4_16_signed_avx2(ws[j] + k / 2);
            __m128i q_hi = _mm_srli_si128(q, 8);
            __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
            __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
            acc_lo[j] = _mm256_fmadd_ps(af0, wf0, acc_lo[j]);
            acc_hi[j] = _mm256_fmadd_ps(af1, wf1, acc_hi[j]);
        }
    }
    for (int j = 0; j < 8; j++) {
        out[j] = hsum256_ps(_mm256_add_ps(acc_lo[j], acc_hi[j]));
    }
    for (; k + 1 < K; k += 2) {
        float a0 = bf16_to_f32(a[k + 0]);
        float a1 = bf16_to_f32(a[k + 1]);
        for (int j = 0; j < 8; j++) {
            uint8_t packed = ws[j][k / 2];
            out[j] += a0 * (float)unpack_i4_signed(packed, 0);
            out[j] += a1 * (float)unpack_i4_signed(packed, 1);
        }
    }
    if (k < K) {
        float a0 = bf16_to_f32(a[k]);
        for (int j = 0; j < 8; j++) {
            out[j] += a0 * (float)unpack_i4_signed(ws[j][k / 2], 0);
        }
    }
}

static inline void dot_bf16_i4_8_blocked16_avx2(const uint16_t *a,
                                                const uint8_t *tile_base,
                                                size_t K,
                                                size_t kblocks,
                                                float *out) {
    __m256 acc_lo[8];
    __m256 acc_hi[8];
    float tail_acc[8];
    for (int j = 0; j < 8; j++) {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
        tail_acc[j] = 0.0f;
    }
    for (size_t kb = 0; kb < kblocks; kb++) {
        const size_t k0 = kb * 16;
        const size_t rem = K > k0 ? K - k0 : 0;
        const uint8_t *tile = tile_base + kb * 64;
        if (rem >= 16) {
            __m128i ab0 = _mm_loadu_si128((const __m128i *)(a + k0));
            __m256i ai0 = _mm256_cvtepu16_epi32(ab0);
            ai0 = _mm256_slli_epi32(ai0, 16);
            __m256 af0 = _mm256_castsi256_ps(ai0);

            __m128i ab1 = _mm_loadu_si128((const __m128i *)(a + k0 + 8));
            __m256i ai1 = _mm256_cvtepu16_epi32(ab1);
            ai1 = _mm256_slli_epi32(ai1, 16);
            __m256 af1 = _mm256_castsi256_ps(ai1);

            for (int j = 0; j < 8; j++) {
                __m128i q = unpack_i4_16_signed_avx2(tile + j * 8);
                __m128i q_hi = _mm_srli_si128(q, 8);
                __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
                __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
                acc_lo[j] = _mm256_fmadd_ps(af0, wf0, acc_lo[j]);
                acc_hi[j] = _mm256_fmadd_ps(af1, wf1, acc_hi[j]);
            }
        } else {
            for (size_t kk = 0; kk < rem; kk++) {
                const float av = bf16_to_f32(a[k0 + kk]);
                for (int j = 0; j < 8; j++) {
                    uint8_t packed = tile[j * 8 + kk / 2];
                    tail_acc[j] += av * (float)unpack_i4_signed(packed, (int)(kk & 1));
                }
            }
        }
    }
    for (int j = 0; j < 8; j++) {
        out[j] = tail_acc[j] + hsum256_ps(_mm256_add_ps(acc_lo[j], acc_hi[j]));
    }
}

static inline void dot_bf16_i8_4_avx2(const uint16_t *a,
                                      const int8_t *w0,
                                      const int8_t *w1,
                                      const int8_t *w2,
                                      const int8_t *w3,
                                      size_t K,
                                      float *out0,
                                      float *out1,
                                      float *out2,
                                      float *out3) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m128i ab = _mm_loadu_si128((const __m128i *)(a + k));
        __m256i ai = _mm256_cvtepu16_epi32(ab);
        ai = _mm256_slli_epi32(ai, 16);
        __m256 af = _mm256_castsi256_ps(ai);

        __m128i wb0 = _mm_loadl_epi64((const __m128i *)(w0 + k));
        __m128i wb1 = _mm_loadl_epi64((const __m128i *)(w1 + k));
        __m128i wb2 = _mm_loadl_epi64((const __m128i *)(w2 + k));
        __m128i wb3 = _mm_loadl_epi64((const __m128i *)(w3 + k));
        __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb0));
        __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb1));
        __m256 wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb2));
        __m256 wf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb3));

        acc0 = _mm256_fmadd_ps(af, wf0, acc0);
        acc1 = _mm256_fmadd_ps(af, wf1, acc1);
        acc2 = _mm256_fmadd_ps(af, wf2, acc2);
        acc3 = _mm256_fmadd_ps(af, wf3, acc3);
    }
    float s0 = hsum256_ps(acc0);
    float s1 = hsum256_ps(acc1);
    float s2 = hsum256_ps(acc2);
    float s3 = hsum256_ps(acc3);
    for (; k < K; k++) {
        float av = bf16_to_f32(a[k]);
        s0 += av * (float)w0[k];
        s1 += av * (float)w1[k];
        s2 += av * (float)w2[k];
        s3 += av * (float)w3[k];
    }
    *out0 = s0;
    *out1 = s1;
    *out2 = s2;
    *out3 = s3;
}

static inline void dot_bf16_i8_8_avx2(const uint16_t *a,
                                      const int8_t *w0,
                                      const int8_t *w1,
                                      const int8_t *w2,
                                      const int8_t *w3,
                                      const int8_t *w4,
                                      const int8_t *w5,
                                      const int8_t *w6,
                                      const int8_t *w7,
                                      size_t K,
                                      float *out) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m128i ab = _mm_loadu_si128((const __m128i *)(a + k));
        __m256i ai = _mm256_cvtepu16_epi32(ab);
        ai = _mm256_slli_epi32(ai, 16);
        __m256 af = _mm256_castsi256_ps(ai);

        __m128i wb0 = _mm_loadl_epi64((const __m128i *)(w0 + k));
        __m128i wb1 = _mm_loadl_epi64((const __m128i *)(w1 + k));
        __m128i wb2 = _mm_loadl_epi64((const __m128i *)(w2 + k));
        __m128i wb3 = _mm_loadl_epi64((const __m128i *)(w3 + k));
        __m128i wb4 = _mm_loadl_epi64((const __m128i *)(w4 + k));
        __m128i wb5 = _mm_loadl_epi64((const __m128i *)(w5 + k));
        __m128i wb6 = _mm_loadl_epi64((const __m128i *)(w6 + k));
        __m128i wb7 = _mm_loadl_epi64((const __m128i *)(w7 + k));

        acc0 = _mm256_fmadd_ps(af, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb0)), acc0);
        acc1 = _mm256_fmadd_ps(af, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb1)), acc1);
        acc2 = _mm256_fmadd_ps(af, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb2)), acc2);
        acc3 = _mm256_fmadd_ps(af, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb3)), acc3);
        acc4 = _mm256_fmadd_ps(af, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb4)), acc4);
        acc5 = _mm256_fmadd_ps(af, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb5)), acc5);
        acc6 = _mm256_fmadd_ps(af, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb6)), acc6);
        acc7 = _mm256_fmadd_ps(af, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wb7)), acc7);
    }
    out[0] = hsum256_ps(acc0);
    out[1] = hsum256_ps(acc1);
    out[2] = hsum256_ps(acc2);
    out[3] = hsum256_ps(acc3);
    out[4] = hsum256_ps(acc4);
    out[5] = hsum256_ps(acc5);
    out[6] = hsum256_ps(acc6);
    out[7] = hsum256_ps(acc7);
    for (; k < K; k++) {
        float av = bf16_to_f32(a[k]);
        out[0] += av * (float)w0[k];
        out[1] += av * (float)w1[k];
        out[2] += av * (float)w2[k];
        out[3] += av * (float)w3[k];
        out[4] += av * (float)w4[k];
        out[5] += av * (float)w5[k];
        out[6] += av * (float)w6[k];
        out[7] += av * (float)w7[k];
    }
}
#endif

int ekernel_gemm(const ekernel_gemm_desc_t *desc,
                 const void *A, const void *B, void *C,
                 ekernel_isa_t isa) {
    if (!desc || !A || !B || !C) return ECPU_ERR_PARAM;
    if (desc->a_prec != ECPU_PRECISION_F32 || desc->b_prec != ECPU_PRECISION_F32
        || desc->out_prec != ECPU_PRECISION_F32) {
        return ECPU_ERR_UNSUPPORTED;
    }
    const float *a = (const float *)A;
    const float *b = (const float *)B;
    float *c = (float *)C;
    const size_t M = desc->M, N = desc->N, K = desc->K;
    const size_t lda = desc->transpose_a ? M : K;
    const size_t ldb = desc->transpose_b ? K : N;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for collapse(2) if (M * N >= 256)
#endif
    for (long long mi = 0; mi < (long long)M; mi++) {
        for (long long ni = 0; ni < (long long)N; ni++) {
            size_t m = (size_t)mi;
            size_t n = (size_t)ni;
            float acc;
#if defined(__AVX2__)
            if (!desc->transpose_a && desc->transpose_b && isa >= EKERNEL_ISA_AVX2) {
                acc = dot_f32_f32_avx2(a + m * lda, b + n * ldb, K);
            } else
#endif
            {
                acc = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    float av = desc->transpose_a ? a[k * lda + m] : a[m * lda + k];
                    float bv = desc->transpose_b ? b[n * ldb + k] : b[k * ldb + n];
                    acc += av * bv;
                }
            }
            c[m * N + n] = acc;
        }
    }
    return ECPU_OK;
}

int ekernel_linear_w8a32(const ekernel_linear_w8a32_desc_t *desc,
                         const float *A,
                         const int8_t *W,
                         const float *scales,
                         float *C,
                         ekernel_isa_t isa) {
    if (!desc || !A || !W || !scales || !C) return ECPU_ERR_PARAM;
    (void)isa;
    const size_t M = desc->M, N = desc->N, K = desc->K;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for collapse(2) if (M * N >= 256)
#endif
    for (long long mi = 0; mi < (long long)M; mi++) {
        for (long long ni = 0; ni < (long long)N; ni++) {
            size_t m = (size_t)mi;
            size_t n = (size_t)ni;
            const float *a = A + m * K;
            const int8_t *w = W + n * K;
#if defined(__AVX2__)
            float acc = dot_f32_i8_avx2(a, w, K);
#else
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += a[k] * (float)w[k];
            }
#endif
            C[m * N + n] = acc * scales[n];
        }
    }
    return ECPU_OK;
}

int ekernel_linear_w8a16_bf16(const ekernel_linear_w8a32_desc_t *desc,
                              const uint16_t *A,
                              const int8_t *W,
                              const float *scales,
                              uint16_t *C,
                              ekernel_isa_t isa) {
    if (!desc || !A || !W || !scales || !C) return ECPU_ERR_PARAM;
    (void)isa;
    const size_t M = desc->M, N = desc->N, K = desc->K;
    if (!w8a16_preconvert_enabled()) {
#if defined(__AVX2__)
        size_t n8 = N / 8;
        if (M == 1) {
            const uint16_t *a = A;
#if defined(ECPU_HAS_OPENMP)
            #pragma omp parallel for if (n8 >= 32)
#endif
            for (long long bi = 0; bi < (long long)n8; bi++) {
                size_t n = (size_t)bi * 8;
                const int8_t *w0 = W + (n + 0) * K;
                const int8_t *w1 = W + (n + 1) * K;
                const int8_t *w2 = W + (n + 2) * K;
                const int8_t *w3 = W + (n + 3) * K;
                const int8_t *w4 = W + (n + 4) * K;
                const int8_t *w5 = W + (n + 5) * K;
                const int8_t *w6 = W + (n + 6) * K;
                const int8_t *w7 = W + (n + 7) * K;
                float acc[8];
                dot_bf16_i8_8_avx2(a, w0, w1, w2, w3, w4, w5, w6, w7, K, acc);
                for (size_t j = 0; j < 8; j++) {
                    C[n + j] = f32_to_bf16(acc[j] * scales[n + j]);
                }
            }
            size_t n = n8 * 8;
            for (; n + 4 <= N; n += 4) {
                const int8_t *w0 = W + (n + 0) * K;
                const int8_t *w1 = W + (n + 1) * K;
                const int8_t *w2 = W + (n + 2) * K;
                const int8_t *w3 = W + (n + 3) * K;
                float acc0, acc1, acc2, acc3;
                dot_bf16_i8_4_avx2(a, w0, w1, w2, w3, K, &acc0, &acc1, &acc2, &acc3);
                C[n + 0] = f32_to_bf16(acc0 * scales[n + 0]);
                C[n + 1] = f32_to_bf16(acc1 * scales[n + 1]);
                C[n + 2] = f32_to_bf16(acc2 * scales[n + 2]);
                C[n + 3] = f32_to_bf16(acc3 * scales[n + 3]);
            }
            for (; n < N; n++) {
                const int8_t *w = W + n * K;
                float acc = dot_bf16_i8_avx2(a, w, K);
                C[n] = f32_to_bf16(acc * scales[n]);
            }
            return ECPU_OK;
        }
        if (w8a16_m_flat_enabled()) {
#if defined(ECPU_HAS_OPENMP)
            #pragma omp parallel for collapse(2) if (M * n8 >= 32)
#endif
            for (long long mi = 0; mi < (long long)M; mi++) {
                for (long long bi = 0; bi < (long long)n8; bi++) {
                    size_t m = (size_t)mi;
                    size_t n = (size_t)bi * 8;
                    const uint16_t *a = A + m * K;
                    const int8_t *w0 = W + (n + 0) * K;
                    const int8_t *w1 = W + (n + 1) * K;
                    const int8_t *w2 = W + (n + 2) * K;
                    const int8_t *w3 = W + (n + 3) * K;
                    const int8_t *w4 = W + (n + 4) * K;
                    const int8_t *w5 = W + (n + 5) * K;
                    const int8_t *w6 = W + (n + 6) * K;
                    const int8_t *w7 = W + (n + 7) * K;
                    float acc[8];
                    dot_bf16_i8_8_avx2(a, w0, w1, w2, w3, w4, w5, w6, w7, K, acc);
                    for (size_t j = 0; j < 8; j++) {
                        C[m * N + n + j] = f32_to_bf16(acc[j] * scales[n + j]);
                    }
                }
            }
            size_t n = n8 * 8;
            if (n < N) {
#if defined(ECPU_HAS_OPENMP)
                #pragma omp parallel for collapse(2) if (M * (N - n) >= 256)
#endif
                for (long long mi = 0; mi < (long long)M; mi++) {
                    for (long long ni = (long long)n; ni < (long long)N; ni++) {
                        size_t m = (size_t)mi;
                        size_t nn = (size_t)ni;
                        const uint16_t *a = A + m * K;
                        const int8_t *w = W + nn * K;
                        float acc = dot_bf16_i8_avx2(a, w, K);
                        C[m * N + nn] = f32_to_bf16(acc * scales[nn]);
                    }
                }
            }
            return ECPU_OK;
        }
        for (size_t m = 0; m < M; m++) {
            const uint16_t *a = A + m * K;
#if defined(ECPU_HAS_OPENMP)
            #pragma omp parallel for if (n8 >= 32)
#endif
            for (long long bi = 0; bi < (long long)n8; bi++) {
                size_t n = (size_t)bi * 8;
                const int8_t *w0 = W + (n + 0) * K;
                const int8_t *w1 = W + (n + 1) * K;
                const int8_t *w2 = W + (n + 2) * K;
                const int8_t *w3 = W + (n + 3) * K;
                const int8_t *w4 = W + (n + 4) * K;
                const int8_t *w5 = W + (n + 5) * K;
                const int8_t *w6 = W + (n + 6) * K;
                const int8_t *w7 = W + (n + 7) * K;
                float acc[8];
                dot_bf16_i8_8_avx2(a, w0, w1, w2, w3, w4, w5, w6, w7, K, acc);
                for (size_t j = 0; j < 8; j++) {
                    C[m * N + n + j] = f32_to_bf16(acc[j] * scales[n + j]);
                }
            }
            size_t n = n8 * 8;
            for (; n + 4 <= N; n += 4) {
                const int8_t *w0 = W + (n + 0) * K;
                const int8_t *w1 = W + (n + 1) * K;
                const int8_t *w2 = W + (n + 2) * K;
                const int8_t *w3 = W + (n + 3) * K;
                float acc0, acc1, acc2, acc3;
                dot_bf16_i8_4_avx2(a, w0, w1, w2, w3, K, &acc0, &acc1, &acc2, &acc3);
                C[m * N + n + 0] = f32_to_bf16(acc0 * scales[n + 0]);
                C[m * N + n + 1] = f32_to_bf16(acc1 * scales[n + 1]);
                C[m * N + n + 2] = f32_to_bf16(acc2 * scales[n + 2]);
                C[m * N + n + 3] = f32_to_bf16(acc3 * scales[n + 3]);
            }
            for (; n < N; n++) {
                const int8_t *w = W + n * K;
                float acc = dot_bf16_i8_avx2(a, w, K);
                C[m * N + n] = f32_to_bf16(acc * scales[n]);
            }
        }
        return ECPU_OK;
#else
#if defined(ECPU_HAS_OPENMP)
        #pragma omp parallel for collapse(2) if (M * N >= 256)
#endif
        for (long long mi = 0; mi < (long long)M; mi++) {
            for (long long ni = 0; ni < (long long)N; ni++) {
                size_t m = (size_t)mi;
                size_t n = (size_t)ni;
                const uint16_t *a = A + m * K;
                const int8_t *w = W + n * K;
#if defined(__AVX2__)
                float acc = dot_bf16_i8_avx2(a, w, K);
#else
                float acc = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    acc += bf16_to_f32(a[k]) * (float)w[k];
                }
#endif
                C[m * N + n] = f32_to_bf16(acc * scales[n]);
            }
        }
        return ECPU_OK;
#endif
    }

    for (size_t m = 0; m < M; m++) {
        const uint16_t *a = A + m * K;
        float *a_f32 = (float *)malloc(K * sizeof(float));
        if (!a_f32) return ECPU_ERR_MEM;
        for (size_t k = 0; k < K; k++) {
            a_f32[k] = bf16_to_f32(a[k]);
        }
#if defined(ECPU_HAS_OPENMP)
        #pragma omp parallel for if (N >= 256)
#endif
        for (long long ni = 0; ni < (long long)N; ni++) {
            size_t n = (size_t)ni;
            const int8_t *w = W + n * K;
#if defined(__AVX2__)
            float acc = dot_f32_i8_avx2(a_f32, w, K);
#else
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += a_f32[k] * (float)w[k];
            }
#endif
            C[m * N + n] = f32_to_bf16(acc * scales[n]);
        }
        free(a_f32);
    }
    return ECPU_OK;
}

int ekernel_linear_w8a16_bf16_argmax(const ekernel_linear_w8a32_desc_t *desc,
                                     const uint16_t *A,
                                     const int8_t *W,
                                     const float *scales,
                                     size_t *out_index,
                                     float *out_value,
                                     ekernel_isa_t isa) {
    if (!desc || !A || !W || !scales || !out_index || !out_value) return ECPU_ERR_PARAM;
    if (desc->M != 1 || desc->N == 0 || desc->K == 0) return ECPU_ERR_UNSUPPORTED;
    (void)isa;
    const size_t N = desc->N, K = desc->K;
    size_t best_idx = 0;
    float best_value = -INFINITY;
#if defined(__AVX2__)
    const size_t n8 = N / 8;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel if (n8 >= 32)
#endif
    {
        size_t local_idx = 0;
        float local_value = -INFINITY;
#if defined(ECPU_HAS_OPENMP)
        #pragma omp for nowait
#endif
        for (long long bi = 0; bi < (long long)n8; bi++) {
            size_t n = (size_t)bi * 8;
            const int8_t *w0 = W + (n + 0) * K;
            const int8_t *w1 = W + (n + 1) * K;
            const int8_t *w2 = W + (n + 2) * K;
            const int8_t *w3 = W + (n + 3) * K;
            const int8_t *w4 = W + (n + 4) * K;
            const int8_t *w5 = W + (n + 5) * K;
            const int8_t *w6 = W + (n + 6) * K;
            const int8_t *w7 = W + (n + 7) * K;
            float acc[8];
            dot_bf16_i8_8_avx2(A, w0, w1, w2, w3, w4, w5, w6, w7, K, acc);
            for (size_t j = 0; j < 8; j++) {
                float v = bf16_to_f32(f32_to_bf16(acc[j] * scales[n + j]));
                if (v > local_value || (v == local_value && n + j < local_idx)) {
                    local_value = v;
                    local_idx = n + j;
                }
            }
        }
#if defined(ECPU_HAS_OPENMP)
        #pragma omp critical
#endif
        {
            if (local_value > best_value || (local_value == best_value && local_idx < best_idx)) {
                best_value = local_value;
                best_idx = local_idx;
            }
        }
    }
    for (size_t n = n8 * 8; n < N; n++) {
        const int8_t *w = W + n * K;
        float v = bf16_to_f32(f32_to_bf16(dot_bf16_i8_avx2(A, w, K) * scales[n]));
        if (v > best_value || (v == best_value && n < best_idx)) {
            best_value = v;
            best_idx = n;
        }
    }
#else
    for (size_t n = 0; n < N; n++) {
        const int8_t *w = W + n * K;
        float acc = 0.0f;
        for (size_t k = 0; k < K; k++) {
            acc += bf16_to_f32(A[k]) * (float)w[k];
        }
        float v = bf16_to_f32(f32_to_bf16(acc * scales[n]));
        if (v > best_value || (v == best_value && n < best_idx)) {
            best_value = v;
            best_idx = n;
        }
    }
#endif
    *out_index = best_idx;
    *out_value = best_value;
    return ECPU_OK;
}

int ekernel_linear_w8a16_bf16_i8b8(const ekernel_linear_w8a32_desc_t *desc,
                                   const uint16_t *A,
                                   const int8_t *W_interleaved,
                                   const float *scales,
                                   uint16_t *C,
                                   ekernel_isa_t isa) {
    if (!desc || !A || !W_interleaved || !scales || !C) return ECPU_ERR_PARAM;
    if (desc->M != 1 || desc->N == 0 || desc->K == 0) return ECPU_ERR_UNSUPPORTED;
    (void)isa;
    const size_t N = desc->N, K = desc->K;
    const size_t blocks = (N + 7) / 8;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for if (blocks >= 32)
#endif
    for (long long bi = 0; bi < (long long)blocks; bi++) {
        size_t b = (size_t)bi;
        size_t n0 = b * 8;
        size_t valid = N - n0;
        if (valid > 8) valid = 8;
        const int8_t *wb = W_interleaved + b * K * 8;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (size_t k = 0; k < K; k++) {
            __m256 av = _mm256_set1_ps(bf16_to_f32(A[k]));
            __m128i w8 = _mm_loadl_epi64((const __m128i *)(wb + k * 8));
            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
            acc = _mm256_fmadd_ps(av, wf, acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (size_t j = 0; j < valid; j++) {
            C[n0 + j] = f32_to_bf16(tmp[j] * scales[n0 + j]);
        }
#else
        float acc[8] = {0};
        for (size_t k = 0; k < K; k++) {
            float av = bf16_to_f32(A[k]);
            for (size_t j = 0; j < valid; j++) {
                acc[j] += av * (float)wb[k * 8 + j];
            }
        }
        for (size_t j = 0; j < valid; j++) {
            C[n0 + j] = f32_to_bf16(acc[j] * scales[n0 + j]);
        }
#endif
    }
    return ECPU_OK;
}

/* ---- Experimental W8A16 + on-the-fly int8 activation (madd_epi16) path ----
 *
 * Same int8 per-output-row-scaled weight layout as ekernel_linear_w8a16_bf16,
 * so it is a drop-in alternative for the same tensors.  Each activation row
 * is quantized to signed int8 with a per-row fp32 scale at call time, and the
 * dot product uses AVX2 _mm256_madd_epi16 on int16 operands (both activation
 * and weight are sign-extended int8->int16, so this is exact for signed*signed
 * with no bias correction needed), accumulating in int32.  This avoids the
 * cvtepi8_epi32 + cvtepi32_ps -> fmadd chain of the stable W8 BF16 kernel.
 *
 * Versus the W4+q8 path: W8 weights are int8 (no nibble unpack) and far more
 * precise than int4, so the int8 activation rounding is expected to be
 * absorbed without perturbing greedy sampling.
 */
#if defined(__AVX2__)
/* 8-output-row int8xint8->int32 dot using madd_epi16.  Loads 16 int8 activation
 * values and 16 int8 values from each of 8 weight rows, widens both to int16,
 * madd_epi16 -> int32, accumulates.  Returns 8 int32 dot products in out[]. */
static inline void dot_i8_i8_8_madd16_avx2(const int8_t *a,
                                           const int8_t *w0,
                                           const int8_t *w1,
                                           const int8_t *w2,
                                           const int8_t *w3,
                                           const int8_t *w4,
                                           const int8_t *w5,
                                           const int8_t *w6,
                                           const int8_t *w7,
                                           size_t K,
                                           int32_t *out) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    __m256i acc4 = _mm256_setzero_si256();
    __m256i acc5 = _mm256_setzero_si256();
    __m256i acc6 = _mm256_setzero_si256();
    __m256i acc7 = _mm256_setzero_si256();
    size_t k = 0;
    for (; k + 16 <= K; k += 16) {
        __m128i a8 = _mm_loadu_si128((const __m128i *)(a + k));
        __m256i a16 = _mm256_cvtepi8_epi16(a8);
        __m256i w0_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w0 + k)));
        __m256i w1_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w1 + k)));
        __m256i w2_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w2 + k)));
        __m256i w3_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w3 + k)));
        __m256i w4_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w4 + k)));
        __m256i w5_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w5 + k)));
        __m256i w6_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w6 + k)));
        __m256i w7_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w7 + k)));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a16, w0_16));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a16, w1_16));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(a16, w2_16));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(a16, w3_16));
        acc4 = _mm256_add_epi32(acc4, _mm256_madd_epi16(a16, w4_16));
        acc5 = _mm256_add_epi32(acc5, _mm256_madd_epi16(a16, w5_16));
        acc6 = _mm256_add_epi32(acc6, _mm256_madd_epi16(a16, w6_16));
        acc7 = _mm256_add_epi32(acc7, _mm256_madd_epi16(a16, w7_16));
    }
    out[0] = hsum256_epi32(acc0);
    out[1] = hsum256_epi32(acc1);
    out[2] = hsum256_epi32(acc2);
    out[3] = hsum256_epi32(acc3);
    out[4] = hsum256_epi32(acc4);
    out[5] = hsum256_epi32(acc5);
    out[6] = hsum256_epi32(acc6);
    out[7] = hsum256_epi32(acc7);
    for (; k < K; k++) {
        int32_t av = (int32_t)a[k];
        out[0] += av * (int32_t)w0[k];
        out[1] += av * (int32_t)w1[k];
        out[2] += av * (int32_t)w2[k];
        out[3] += av * (int32_t)w3[k];
        out[4] += av * (int32_t)w4[k];
        out[5] += av * (int32_t)w5[k];
        out[6] += av * (int32_t)w6[k];
        out[7] += av * (int32_t)w7[k];
    }
}
#endif

int ekernel_linear_w8a16_bf16_q8(const ekernel_linear_w8a32_desc_t *desc,
                                 const uint16_t *A,
                                 const int8_t *W,
                                 const float *scales,
                                 uint16_t *C,
                                 ekernel_isa_t isa) {
    if (!desc || !A || !W || !scales || !C) return ECPU_ERR_PARAM;
    (void)isa;
    const size_t M = desc->M, N = desc->N, K = desc->K;

    /* Pre-quantize each activation row to signed int8 with a per-row scale. */
    float a_scales_stack[64];
    int8_t a_all_stack[16384];
    float *a_scales = (M <= 64) ? a_scales_stack : (float *)malloc(M * sizeof(float));
    int8_t *a_all = (M * K <= 16384) ? a_all_stack : (int8_t *)malloc((size_t)M * K);
    if (!a_scales || !a_all) {
        if (a_scales != a_scales_stack) free(a_scales);
        if (a_all != a_all_stack) free(a_all);
        return ECPU_ERR_MEM;
    }
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for if (M >= 4)
#endif
    for (long long mi = 0; mi < (long long)M; mi++) {
        size_t m = (size_t)mi;
        a_scales[m] = quantize_bf16_row_to_i8(A + m * K, a_all + m * K, K);
    }

    const size_t n8 = N / 8;
#if defined(__AVX2__)
    if (n8 > 0) {
#if defined(ECPU_HAS_OPENMP)
        #pragma omp parallel for collapse(2) if (M * n8 >= 32)
#endif
        for (long long mi = 0; mi < (long long)M; mi++) {
            for (long long bi = 0; bi < (long long)n8; bi++) {
                const size_t m = (size_t)mi;
                const size_t n = (size_t)bi * 8;
                const int8_t *a_int8 = a_all + m * K;
                const float a_scale = a_scales[m];
                int32_t dot[8];
                dot_i8_i8_8_madd16_avx2(
                    a_int8,
                    W + (n + 0) * K, W + (n + 1) * K, W + (n + 2) * K, W + (n + 3) * K,
                    W + (n + 4) * K, W + (n + 5) * K, W + (n + 6) * K, W + (n + 7) * K,
                    K, dot);
                for (int j = 0; j < 8; j++) {
                    float v = scales[n + j] * a_scale * (float)dot[j];
                    C[m * N + n + j] = f32_to_bf16(v);
                }
            }
        }
        const size_t n_tail = n8 * 8;
        if (n_tail < N) {
#if defined(ECPU_HAS_OPENMP)
            #pragma omp parallel for collapse(2) if (M * (N - n_tail) >= 256)
#endif
            for (long long mi = 0; mi < (long long)M; mi++) {
                for (long long ni = (long long)n_tail; ni < (long long)N; ni++) {
                    const size_t m = (size_t)mi;
                    const size_t n = (size_t)ni;
                    const int8_t *a_int8 = a_all + m * K;
                    const float a_scale = a_scales[m];
                    const int8_t *w = W + n * K;
                    __m256i acc = _mm256_setzero_si256();
                    size_t k = 0;
                    for (; k + 16 <= K; k += 16) {
                        __m128i a8 = _mm_loadu_si128((const __m128i *)(a_int8 + k));
                        __m256i a16 = _mm256_cvtepi8_epi16(a8);
                        __m256i w16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + k)));
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, w16));
                    }
                    int32_t dot = hsum256_epi32(acc);
                    for (; k < K; k++) {
                        dot += (int32_t)a_int8[k] * (int32_t)w[k];
                    }
                    C[m * N + n] = f32_to_bf16(scales[n] * a_scale * (float)dot);
                }
            }
        }
    } else {
        /* N < 8: one output row at a time. */
#if defined(ECPU_HAS_OPENMP)
        #pragma omp parallel for collapse(2) if (M * N >= 256)
#endif
        for (long long mi = 0; mi < (long long)M; mi++) {
            for (long long ni = 0; ni < (long long)N; ni++) {
                const size_t m = (size_t)mi;
                const size_t n = (size_t)ni;
                const int8_t *a_int8 = a_all + m * K;
                const float a_scale = a_scales[m];
                const int8_t *w = W + n * K;
                __m256i acc = _mm256_setzero_si256();
                size_t k = 0;
                for (; k + 16 <= K; k += 16) {
                    __m128i a8 = _mm_loadu_si128((const __m128i *)(a_int8 + k));
                    __m256i a16 = _mm256_cvtepi8_epi16(a8);
                    __m256i w16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + k)));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, w16));
                }
                int32_t dot = hsum256_epi32(acc);
                for (; k < K; k++) {
                    dot += (int32_t)a_int8[k] * (int32_t)w[k];
                }
                C[m * N + n] = f32_to_bf16(scales[n] * a_scale * (float)dot);
            }
        }
    }
#else
    /* Scalar fallback. */
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for collapse(2) if (M * N >= 256)
#endif
    for (long long mi = 0; mi < (long long)M; mi++) {
        for (long long ni = 0; ni < (long long)N; ni++) {
            const size_t m = (size_t)mi;
            const size_t n = (size_t)ni;
            const int8_t *a_int8 = a_all + m * K;
            const float a_scale = a_scales[m];
            const int8_t *w = W + n * K;
            int32_t dot = 0;
            for (size_t k = 0; k < K; k++) {
                dot += (int32_t)a_int8[k] * (int32_t)w[k];
            }
            C[m * N + n] = f32_to_bf16(scales[n] * a_scale * (float)dot);
        }
    }
#endif

    if (a_scales != a_scales_stack) free(a_scales);
    if (a_all != a_all_stack) free(a_all);
    return ECPU_OK;
}

int ekernel_linear_w4a16_bf16(const ekernel_linear_w8a32_desc_t *desc,
                              const uint16_t *A,
                              const uint8_t *W,
                              const float *scales,
                              uint16_t *C,
                              ekernel_isa_t isa) {
    if (!desc || !A || !W || !scales || !C) return ECPU_ERR_PARAM;
    (void)isa;
    const size_t M = desc->M, N = desc->N, K = desc->K;
    const size_t packed_k = (K + 1) / 2;
#if defined(__AVX2__)
    const size_t n8 = N / 8;
    if (n8 > 0) {
#if defined(ECPU_HAS_OPENMP)
        #pragma omp parallel for collapse(2) if (M * n8 >= 32)
#endif
        for (long long mi = 0; mi < (long long)M; mi++) {
            for (long long bi = 0; bi < (long long)n8; bi++) {
                const size_t m = (size_t)mi;
                const size_t n = (size_t)bi * 8;
                const uint16_t *a = A + m * K;
                const uint8_t *w0 = W + (n + 0) * packed_k;
                const uint8_t *w1 = W + (n + 1) * packed_k;
                const uint8_t *w2 = W + (n + 2) * packed_k;
                const uint8_t *w3 = W + (n + 3) * packed_k;
                const uint8_t *w4 = W + (n + 4) * packed_k;
                const uint8_t *w5 = W + (n + 5) * packed_k;
                const uint8_t *w6 = W + (n + 6) * packed_k;
                const uint8_t *w7 = W + (n + 7) * packed_k;
                float acc[8];
                dot_bf16_i4_8_packed_avx2(a, w0, w1, w2, w3, w4, w5, w6, w7, K, acc);
                for (size_t j = 0; j < 8; j++) {
                    C[m * N + n + j] = f32_to_bf16(acc[j] * scales[n + j]);
                }
            }
        }
        const size_t n_tail = n8 * 8;
        if (n_tail < N) {
#if defined(ECPU_HAS_OPENMP)
            #pragma omp parallel for collapse(2) if (M * (N - n_tail) >= 256)
#endif
            for (long long mi = 0; mi < (long long)M; mi++) {
                for (long long ni = (long long)n_tail; ni < (long long)N; ni++) {
                    const size_t m = (size_t)mi;
                    const size_t n = (size_t)ni;
                    const uint16_t *a = A + m * K;
                    const uint8_t *w = W + n * packed_k;
                    float acc = dot_bf16_i4_packed_avx2(a, w, K);
                    C[m * N + n] = f32_to_bf16(acc * scales[n]);
                }
            }
        }
        return ECPU_OK;
    }
#endif
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for collapse(2) if (M * N >= 256)
#endif
    for (long long mi = 0; mi < (long long)M; mi++) {
        for (long long ni = 0; ni < (long long)N; ni++) {
            const size_t m = (size_t)mi;
            const size_t n = (size_t)ni;
            const uint16_t *a = A + m * K;
            const uint8_t *w = W + n * packed_k;
#if defined(__AVX2__)
            float acc = dot_bf16_i4_packed_avx2(a, w, K);
#else
            float acc = 0.0f;
            size_t k = 0;
            for (; k + 1 < K; k += 2) {
                uint8_t packed = w[k / 2];
                int8_t w0 = unpack_i4_signed(packed, 0);
                int8_t w1 = unpack_i4_signed(packed, 1);
                acc += bf16_to_f32(a[k + 0]) * (float)w0;
                acc += bf16_to_f32(a[k + 1]) * (float)w1;
            }
            if (k < K) {
                int8_t w0 = unpack_i4_signed(w[k / 2], 0);
                acc += bf16_to_f32(a[k]) * (float)w0;
            }
#endif
            C[m * N + n] = f32_to_bf16(acc * scales[n]);
        }
    }
    return ECPU_OK;
}

static int linear_w4a16g_bf16_impl(const ekernel_linear_w8a32_desc_t *desc,
                                   const uint16_t *A,
                                   const uint8_t *W,
                                   const float *scales,
                                   uint16_t *C,
                                   ekernel_isa_t isa,
                                   size_t group_size) {
    if (!desc || !A || !W || !scales || !C) return ECPU_ERR_PARAM;
    (void)isa;
    const size_t M = desc->M, N = desc->N, K = desc->K;
    const size_t groups = (K + group_size - 1) / group_size;
    const size_t packed_k = (K + 1) / 2;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for collapse(2) if (M * N >= 256)
#endif
    for (long long mi = 0; mi < (long long)M; mi++) {
        for (long long ni = 0; ni < (long long)N; ni++) {
            const size_t m = (size_t)mi;
            const size_t n = (size_t)ni;
            const uint16_t *a = A + m * K;
            const uint8_t *w = W + n * packed_k;
            const float *s = scales + n * groups;
            float acc = 0.0f;
            for (size_t g = 0; g < groups; g++) {
                const size_t k0 = g * group_size;
                size_t gk = K - k0;
                if (gk > group_size) gk = group_size;
#if defined(__AVX2__)
                float block_acc = dot_bf16_i4_packed_avx2(a + k0, w + k0 / 2, gk);
#else
                float block_acc = 0.0f;
                size_t k = 0;
                for (; k + 1 < gk; k += 2) {
                    uint8_t packed = w[(k0 + k) / 2];
                    int8_t w0 = unpack_i4_signed(packed, 0);
                    int8_t w1 = unpack_i4_signed(packed, 1);
                    block_acc += bf16_to_f32(a[k0 + k + 0]) * (float)w0;
                    block_acc += bf16_to_f32(a[k0 + k + 1]) * (float)w1;
                }
                if (k < gk) {
                    int8_t w0 = unpack_i4_signed(w[(k0 + k) / 2], 0);
                    block_acc += bf16_to_f32(a[k0 + k]) * (float)w0;
                }
#endif
                acc += block_acc * s[g];
            }
            C[m * N + n] = f32_to_bf16(acc);
        }
    }
    return ECPU_OK;
}

int ekernel_linear_w4a16g32_bf16(const ekernel_linear_w8a32_desc_t *desc,
                                 const uint16_t *A,
                                 const uint8_t *W,
                                 const float *scales,
                                 uint16_t *C,
                                 ekernel_isa_t isa) {
    return linear_w4a16g_bf16_impl(desc, A, W, scales, C, isa, 32);
}

int ekernel_linear_w4a16g128_bf16(const ekernel_linear_w8a32_desc_t *desc,
                                  const uint16_t *A,
                                  const uint8_t *W,
                                  const float *scales,
                                  uint16_t *C,
                                  ekernel_isa_t isa) {
    return linear_w4a16g_bf16_impl(desc, A, W, scales, C, isa, 128);
}

int ekernel_linear_w4a16_bf16_i4b8(const ekernel_linear_w8a32_desc_t *desc,
                                   const uint16_t *A,
                                   const int8_t *W_interleaved,
                                   const float *scales,
                                   uint16_t *C,
                                   ekernel_isa_t isa) {
    if (!desc || !A || !W_interleaved || !scales || !C) return ECPU_ERR_PARAM;
    if (desc->M != 1 || desc->N == 0 || desc->K == 0) return ECPU_ERR_UNSUPPORTED;
    (void)isa;
    const size_t N = desc->N, K = desc->K;
    const size_t blocks = (N + 7) / 8;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for if (blocks >= 32)
#endif
    for (long long bi = 0; bi < (long long)blocks; bi++) {
        const size_t b = (size_t)bi;
        const size_t n0 = b * 8;
        size_t valid = N - n0;
        if (valid > 8) valid = 8;
        const int8_t *wb = W_interleaved + b * K * 8;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (size_t k = 0; k < K; k++) {
            __m256 av = _mm256_set1_ps(bf16_to_f32(A[k]));
            __m128i w8 = _mm_loadl_epi64((const __m128i *)(wb + k * 8));
            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
            acc = _mm256_fmadd_ps(av, wf, acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (size_t j = 0; j < valid; j++) {
            C[n0 + j] = f32_to_bf16(tmp[j] * scales[n0 + j]);
        }
#else
        float acc[8] = {0};
        for (size_t k = 0; k < K; k++) {
            float av = bf16_to_f32(A[k]);
            for (size_t j = 0; j < valid; j++) {
                acc[j] += av * (float)wb[k * 8 + j];
            }
        }
        for (size_t j = 0; j < valid; j++) {
            C[n0 + j] = f32_to_bf16(acc[j] * scales[n0 + j]);
        }
#endif
    }
    return ECPU_OK;
}

int ekernel_linear_w4a16_bf16_b8(const ekernel_linear_w8a32_desc_t *desc,
                                 const uint16_t *A,
                                 const uint8_t *W_blocked,
                                 const float *scales,
                                 uint16_t *C,
                                 ekernel_isa_t isa) {
    if (!desc || !A || !W_blocked || !scales || !C) return ECPU_ERR_PARAM;
    if (desc->M != 1 || desc->N == 0 || desc->K == 0) return ECPU_ERR_UNSUPPORTED;
    (void)isa;
    const size_t N = desc->N, K = desc->K;
    const size_t blocks = (N + 7) / 8;
    const size_t kblocks = (K + 15) / 16;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for if (blocks >= 32)
#endif
    for (long long bi = 0; bi < (long long)blocks; bi++) {
        const size_t b = (size_t)bi;
        const size_t n0 = b * 8;
        size_t valid = N - n0;
        if (valid > 8) valid = 8;
        const uint8_t *tile_base = W_blocked + b * kblocks * 64;
        float acc[8];
#if defined(__AVX2__)
        dot_bf16_i4_8_blocked16_avx2(A, tile_base, K, kblocks, acc);
#else
        for (size_t j = 0; j < 8; j++) acc[j] = 0.0f;
        for (size_t k = 0; k < K; k++) {
            const size_t kb = k / 16;
            const size_t kk = k % 16;
            const uint8_t *tile = tile_base + kb * 64;
            float av = bf16_to_f32(A[k]);
            for (size_t j = 0; j < 8; j++) {
                uint8_t packed = tile[j * 8 + kk / 2];
                acc[j] += av * (float)unpack_i4_signed(packed, (int)(kk & 1));
            }
        }
#endif
        for (size_t j = 0; j < valid; j++) {
            C[n0 + j] = f32_to_bf16(acc[j] * scales[n0 + j]);
        }
    }
    return ECPU_OK;
}

/* ---- Experimental W4A16 + on-the-fly int8 activation (pmaddwd) path -------
 *
 * Same packed-int4 / per-row-fp32-scale weight layout as ekernel_linear_w4a16_bf16,
 * so it is a drop-in alternative for the same tensors.  The difference is that
 * each activation row is quantized to signed int8 with a per-row fp32 scale at
 * call time, and the dot product is computed with AVX2 _mm256_madd_epi16 on
 * int16 operands, accumulating in int32.  This avoids the per-nibble
 * cvtepi8_epi32 + cvtepi32_ps -> fmadd chain that makes the BF16-activation W4
 * kernel compute-bound on this AVX2 host.  The only added error versus the
 * stable W4 path is the int8 activation rounding; the int4 weights are exact
 * and the int32 accumulation is exact.
 */
static float quantize_bf16_row_to_i8(const uint16_t *a, int8_t *out, size_t K) {
    float a_max = 0.0f;
    for (size_t k = 0; k < K; k++) {
        float v = bf16_to_f32(a[k]);
        float av = v < 0.0f ? -v : v;
        if (av > a_max) a_max = av;
    }
    float a_scale = a_max / 127.0f;
    if (a_scale < 1e-12f) a_scale = 1.0f;
    float inv = 1.0f / a_scale;
    for (size_t k = 0; k < K; k++) {
        float v = bf16_to_f32(a[k]) * inv;
        int q = (int)lroundf(v);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out[k] = (int8_t)q;
    }
    return a_scale;
}

#if defined(__AVX2__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return (int32_t)_mm_cvtsi128_si32(s);
}

/* Unpack 16 signed int4 nibbles (8 packed bytes) and sign-extend to 16 int16. */
static inline __m256i unpack_i4_16_to_i16_avx2(const uint8_t *w) {
    __m128i q8 = unpack_i4_16_signed_avx2(w);
    return _mm256_cvtepi8_epi16(q8);
}
#endif

int ekernel_linear_w4a16_bf16_q8(const ekernel_linear_w8a32_desc_t *desc,
                                 const uint16_t *A,
                                 const uint8_t *W,
                                 const float *scales,
                                 uint16_t *C,
                                 ekernel_isa_t isa) {
    if (!desc || !A || !W || !scales || !C) return ECPU_ERR_PARAM;
    (void)isa;
    const size_t M = desc->M, N = desc->N, K = desc->K;
    const size_t packed_k = (K + 1) / 2;

    /* Pre-quantize each activation row to signed int8 with a per-row scale.
     * The int8 buffer is reused per row inside the m-loop, so it only needs K
     * bytes; a 16 KiB stack buffer covers every real model shape. */
    float a_scales_stack[64];
    int8_t a_all_stack[16384];
    float *a_scales = (M <= 64) ? a_scales_stack : (float *)malloc(M * sizeof(float));
    int8_t *a_all = (M * K <= 16384) ? a_all_stack : (int8_t *)malloc((size_t)M * K);
    if (!a_scales || !a_all) {
        if (a_scales != a_scales_stack) free(a_scales);
        if (a_all != a_all_stack) free(a_all);
        return ECPU_ERR_MEM;
    }
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for if (M >= 4)
#endif
    for (long long mi = 0; mi < (long long)M; mi++) {
        size_t m = (size_t)mi;
        a_scales[m] = quantize_bf16_row_to_i8(A + m * K, a_all + m * K, K);
    }

    const size_t n8 = N / 8;
#if defined(__AVX2__)
    if (n8 > 0) {
#if defined(ECPU_HAS_OPENMP)
        #pragma omp parallel for collapse(2) if (M * n8 >= 32)
#endif
        for (long long mi = 0; mi < (long long)M; mi++) {
            for (long long bi = 0; bi < (long long)n8; bi++) {
                const size_t m = (size_t)mi;
                const size_t n = (size_t)bi * 8;
                const int8_t *a_int8 = a_all + m * K;
                const float a_scale = a_scales[m];
                __m256i acc[8];
                for (int j = 0; j < 8; j++) acc[j] = _mm256_setzero_si256();
                const uint8_t *ws[8] = {
                    W + (n + 0) * packed_k, W + (n + 1) * packed_k,
                    W + (n + 2) * packed_k, W + (n + 3) * packed_k,
                    W + (n + 4) * packed_k, W + (n + 5) * packed_k,
                    W + (n + 6) * packed_k, W + (n + 7) * packed_k,
                };
                size_t k = 0;
                for (; k + 16 <= K; k += 16) {
                    __m128i a8 = _mm_loadu_si128((const __m128i *)(a_int8 + k));
                    __m256i a16 = _mm256_cvtepi8_epi16(a8);
                    for (int j = 0; j < 8; j++) {
                        __m256i w16 = unpack_i4_16_to_i16_avx2(ws[j] + k / 2);
                        __m256i partial = _mm256_madd_epi16(a16, w16);
                        acc[j] = _mm256_add_epi32(acc[j], partial);
                    }
                }
                int32_t dot[8];
                for (int j = 0; j < 8; j++) dot[j] = hsum256_epi32(acc[j]);
                /* K tail (K not a multiple of 16). */
                for (; k < K; k++) {
                    int32_t av = (int32_t)a_int8[k];
                    int high = (int)(k & 1);
                    for (int j = 0; j < 8; j++) {
                        int8_t wv = unpack_i4_signed(ws[j][k / 2], high);
                        dot[j] += av * (int32_t)wv;
                    }
                }
                for (int j = 0; j < 8; j++) {
                    float v = scales[n + j] * a_scale * (float)dot[j];
                    C[m * N + n + j] = f32_to_bf16(v);
                }
            }
        }
        /* N tail (N not a multiple of 8): one output row at a time. */
        const size_t n_tail = n8 * 8;
        if (n_tail < N) {
#if defined(ECPU_HAS_OPENMP)
            #pragma omp parallel for collapse(2) if (M * (N - n_tail) >= 256)
#endif
            for (long long mi = 0; mi < (long long)M; mi++) {
                for (long long ni = (long long)n_tail; ni < (long long)N; ni++) {
                    const size_t m = (size_t)mi;
                    const size_t n = (size_t)ni;
                    const int8_t *a_int8 = a_all + m * K;
                    const float a_scale = a_scales[m];
                    const uint8_t *w = W + n * packed_k;
                    __m256i acc = _mm256_setzero_si256();
                    size_t k = 0;
                    for (; k + 16 <= K; k += 16) {
                        __m128i a8 = _mm_loadu_si128((const __m128i *)(a_int8 + k));
                        __m256i a16 = _mm256_cvtepi8_epi16(a8);
                        __m256i w16 = unpack_i4_16_to_i16_avx2(w + k / 2);
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, w16));
                    }
                    int32_t dot = hsum256_epi32(acc);
                    for (; k < K; k++) {
                        int8_t wv = unpack_i4_signed(w[k / 2], (int)(k & 1));
                        dot += (int32_t)a_int8[k] * (int32_t)wv;
                    }
                    C[m * N + n] = f32_to_bf16(scales[n] * a_scale * (float)dot);
                }
            }
        }
    } else {
        /* N < 8: no 8-row block available. */
#if defined(ECPU_HAS_OPENMP)
        #pragma omp parallel for collapse(2) if (M * N >= 256)
#endif
        for (long long mi = 0; mi < (long long)M; mi++) {
            for (long long ni = 0; ni < (long long)N; ni++) {
                const size_t m = (size_t)mi;
                const size_t n = (size_t)ni;
                const int8_t *a_int8 = a_all + m * K;
                const float a_scale = a_scales[m];
                const uint8_t *w = W + n * packed_k;
                __m256i acc = _mm256_setzero_si256();
                size_t k = 0;
                for (; k + 16 <= K; k += 16) {
                    __m128i a8 = _mm_loadu_si128((const __m128i *)(a_int8 + k));
                    __m256i a16 = _mm256_cvtepi8_epi16(a8);
                    __m256i w16 = unpack_i4_16_to_i16_avx2(w + k / 2);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, w16));
                }
                int32_t dot = hsum256_epi32(acc);
                for (; k < K; k++) {
                    int8_t wv = unpack_i4_signed(w[k / 2], (int)(k & 1));
                    dot += (int32_t)a_int8[k] * (int32_t)wv;
                }
                C[m * N + n] = f32_to_bf16(scales[n] * a_scale * (float)dot);
            }
        }
    }
#else
    /* Scalar fallback for non-AVX2 builds. */
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for collapse(2) if (M * N >= 256)
#endif
    for (long long mi = 0; mi < (long long)M; mi++) {
        for (long long ni = 0; ni < (long long)N; ni++) {
            const size_t m = (size_t)mi;
            const size_t n = (size_t)ni;
            const int8_t *a_int8 = a_all + m * K;
            const float a_scale = a_scales[m];
            const uint8_t *w = W + n * packed_k;
            int32_t dot = 0;
            size_t k = 0;
            for (; k + 1 < K; k += 2) {
                int8_t w0 = unpack_i4_signed(w[k / 2], 0);
                int8_t w1 = unpack_i4_signed(w[k / 2], 1);
                dot += (int32_t)a_int8[k + 0] * (int32_t)w0;
                dot += (int32_t)a_int8[k + 1] * (int32_t)w1;
            }
            if (k < K) {
                int8_t w0 = unpack_i4_signed(w[k / 2], 0);
                dot += (int32_t)a_int8[k] * (int32_t)w0;
            }
            C[m * N + n] = f32_to_bf16(scales[n] * a_scale * (float)dot);
        }
    }
#endif

    if (a_scales != a_scales_stack) free(a_scales);
    if (a_all != a_all_stack) free(a_all);
    return ECPU_OK;
}

int ekernel_rmsnorm(const void *x, void *out, const void *weight,
                    size_t n, float eps, ecpu_precision_t dtype) {
    if (dtype != ECPU_PRECISION_F32) return ECPU_ERR_UNSUPPORTED;
    const float *xf = (const float *)x;
    const float *wf = (const float *)weight;
    float *of = (float *)out;
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += xf[i] * xf[i];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++) of[i] = xf[i] * ss * wf[i];
    return ECPU_OK;
}

int ekernel_rmsnorm_bf16(const uint16_t *x,
                         uint16_t *out,
                         const uint16_t *weight,
                         size_t rows,
                         size_t n,
                         float eps,
                         int add_one) {
    if (!x || !out || !weight || rows == 0 || n == 0) return ECPU_ERR_PARAM;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for if (rows >= 8)
#endif
    for (long long ri = 0; ri < (long long)rows; ri++) {
        size_t r = (size_t)ri;
        const uint16_t *xr = x + r * n;
        uint16_t *orow = out + r * n;
        float ss = 0.0f;
        size_t i = 0;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            __m128i xb = _mm_loadu_si128((const __m128i *)(xr + i));
            __m256i xi = _mm256_cvtepu16_epi32(xb);
            xi = _mm256_slli_epi32(xi, 16);
            __m256 xf = _mm256_castsi256_ps(xi);
            acc = _mm256_fmadd_ps(xf, xf, acc);
        }
        ss = hsum256_ps(acc);
#endif
        for (; i < n; i++) {
            float v = bf16_to_f32(xr[i]);
            ss += v * v;
        }
        float inv = 1.0f / sqrtf(ss / (float)n + eps);
        for (i = 0; i < n; i++) {
            float w = bf16_to_f32(weight[i]);
            if (add_one) w += 1.0f;
            float v = bf16_to_f32(xr[i]) * inv * w;
            orow[i] = f32_to_bf16(v);
        }
    }
    return ECPU_OK;
}

int ekernel_rope(void *x, size_t n, size_t head_dim, size_t pos,
                 float theta, ecpu_precision_t dtype) {
    if (dtype != ECPU_PRECISION_F32) return ECPU_ERR_UNSUPPORTED;
    float *xf = (float *)x;
    const size_t n_heads = n / head_dim;
    for (size_t h = 0; h < n_heads; h++) {
        float *p = xf + h * head_dim;
        for (size_t i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float x0 = p[i], x1 = p[i + 1];
            p[i]     = x0 * c - x1 * s;
            p[i + 1] = x0 * s + x1 * c;
        }
    }
    return ECPU_OK;
}

int ekernel_softmax(void *x, size_t n, float scale, ecpu_precision_t dtype) {
    if (dtype != ECPU_PRECISION_F32) return ECPU_ERR_UNSUPPORTED;
    float *xf = (float *)x;
    float mx = xf[0] * scale;
    for (size_t i = 1; i < n; i++) { float v = xf[i] * scale; if (v > mx) mx = v; }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) { xf[i] = expf(xf[i] * scale - mx); sum += xf[i]; }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) xf[i] *= inv;
    return ECPU_OK;
}

int ekernel_silu(const void *x, void *out, size_t n, ecpu_precision_t dtype) {
    if (dtype != ECPU_PRECISION_F32) return ECPU_ERR_UNSUPPORTED;
    const float *xf = (const float *)x;
    float *of = (float *)out;
    for (size_t i = 0; i < n; i++) {
        float v = xf[i];
        of[i] = v / (1.0f + expf(-v));
    }
    return ECPU_OK;
}

int ekernel_mul(const void *a, const void *b, void *out, size_t n,
                ecpu_precision_t dtype) {
    if (dtype != ECPU_PRECISION_F32) return ECPU_ERR_UNSUPPORTED;
    const float *af = (const float *)a;
    const float *bf = (const float *)b;
    float *of = (float *)out;
    for (size_t i = 0; i < n; i++) of[i] = af[i] * bf[i];
    return ECPU_OK;
}

int ekernel_swiglu_bf16(const uint16_t *gate,
                        const uint16_t *up,
                        uint16_t *out,
                        size_t n) {
    if (!gate || !up || !out) return ECPU_ERR_PARAM;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for if (n >= 65536)
#endif
    for (long long ii = 0; ii < (long long)n; ii++) {
        size_t i = (size_t)ii;
        float g = bf16_to_f32(gate[i]);
        float u = bf16_to_f32(up[i]);
        float silu = g / (1.0f + expf(-g));
        out[i] = f32_to_bf16(silu * u);
    }
    return ECPU_OK;
}

int ekernel_swiglu_w4a16_bf16(const ekernel_w4a16_swiglu_desc_t *desc,
                              const uint16_t *A,
                              const uint8_t *gate_up_W,
                              const float *gate_up_scales,
                              const uint8_t *down_W,
                              const float *down_scales,
                              uint16_t *C,
                              ekernel_isa_t isa) {
    if (!desc || !A || !gate_up_W || !gate_up_scales || !down_W || !down_scales || !C) {
        return ECPU_ERR_PARAM;
    }
    if (desc->M != 1 || desc->hidden_size == 0 || desc->intermediate_size == 0 ||
        desc->gate_up_in_features == 0 || desc->down_in_features != desc->intermediate_size) {
        return ECPU_ERR_UNSUPPORTED;
    }
    (void)isa;
    const size_t H = desc->hidden_size;
    const size_t I = desc->intermediate_size;
    const size_t K = desc->gate_up_in_features;
    const size_t gate_packed_k = (K + 1) / 2;
    const size_t down_packed_k = (I + 1) / 2;

    uint16_t *hidden = (uint16_t *)malloc(I * sizeof(uint16_t));
    if (!hidden) return ECPU_ERR_MEM;

#if defined(__AVX2__)
    const size_t i8 = I / 8;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for if (i8 >= 32)
#endif
    for (long long bi = 0; bi < (long long)i8; bi++) {
        const size_t i = (size_t)bi * 8;
        const uint8_t *g0 = gate_up_W + (i + 0) * gate_packed_k;
        const uint8_t *g1 = gate_up_W + (i + 1) * gate_packed_k;
        const uint8_t *g2 = gate_up_W + (i + 2) * gate_packed_k;
        const uint8_t *g3 = gate_up_W + (i + 3) * gate_packed_k;
        const uint8_t *g4 = gate_up_W + (i + 4) * gate_packed_k;
        const uint8_t *g5 = gate_up_W + (i + 5) * gate_packed_k;
        const uint8_t *g6 = gate_up_W + (i + 6) * gate_packed_k;
        const uint8_t *g7 = gate_up_W + (i + 7) * gate_packed_k;
        const uint8_t *u0 = gate_up_W + (I + i + 0) * gate_packed_k;
        const uint8_t *u1 = gate_up_W + (I + i + 1) * gate_packed_k;
        const uint8_t *u2 = gate_up_W + (I + i + 2) * gate_packed_k;
        const uint8_t *u3 = gate_up_W + (I + i + 3) * gate_packed_k;
        const uint8_t *u4 = gate_up_W + (I + i + 4) * gate_packed_k;
        const uint8_t *u5 = gate_up_W + (I + i + 5) * gate_packed_k;
        const uint8_t *u6 = gate_up_W + (I + i + 6) * gate_packed_k;
        const uint8_t *u7 = gate_up_W + (I + i + 7) * gate_packed_k;
        float gate_acc[8];
        float up_acc[8];
        dot_bf16_i4_8_packed_avx2(A, g0, g1, g2, g3, g4, g5, g6, g7, K, gate_acc);
        dot_bf16_i4_8_packed_avx2(A, u0, u1, u2, u3, u4, u5, u6, u7, K, up_acc);
        for (size_t j = 0; j < 8; j++) {
            float g = bf16_to_f32(f32_to_bf16(gate_acc[j] * gate_up_scales[i + j]));
            float u = bf16_to_f32(f32_to_bf16(up_acc[j] * gate_up_scales[I + i + j]));
            float silu = g / (1.0f + expf(-g));
            hidden[i + j] = f32_to_bf16(silu * u);
        }
    }
    for (size_t i = i8 * 8; i < I; i++) {
        const uint8_t *g = gate_up_W + i * gate_packed_k;
        const uint8_t *urow = gate_up_W + (I + i) * gate_packed_k;
        float gv = bf16_to_f32(f32_to_bf16(dot_bf16_i4_packed_avx2(A, g, K) * gate_up_scales[i]));
        float uv = bf16_to_f32(f32_to_bf16(dot_bf16_i4_packed_avx2(A, urow, K) * gate_up_scales[I + i]));
        float silu = gv / (1.0f + expf(-gv));
        hidden[i] = f32_to_bf16(silu * uv);
    }

    const size_t h8 = H / 8;
#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for if (h8 >= 32)
#endif
    for (long long bi = 0; bi < (long long)h8; bi++) {
        const size_t h = (size_t)bi * 8;
        const uint8_t *w0 = down_W + (h + 0) * down_packed_k;
        const uint8_t *w1 = down_W + (h + 1) * down_packed_k;
        const uint8_t *w2 = down_W + (h + 2) * down_packed_k;
        const uint8_t *w3 = down_W + (h + 3) * down_packed_k;
        const uint8_t *w4 = down_W + (h + 4) * down_packed_k;
        const uint8_t *w5 = down_W + (h + 5) * down_packed_k;
        const uint8_t *w6 = down_W + (h + 6) * down_packed_k;
        const uint8_t *w7 = down_W + (h + 7) * down_packed_k;
        float acc[8];
        dot_bf16_i4_8_packed_avx2(hidden, w0, w1, w2, w3, w4, w5, w6, w7, I, acc);
        for (size_t j = 0; j < 8; j++) {
            C[h + j] = f32_to_bf16(acc[j] * down_scales[h + j]);
        }
    }
    for (size_t h = h8 * 8; h < H; h++) {
        const uint8_t *w = down_W + h * down_packed_k;
        float acc = dot_bf16_i4_packed_avx2(hidden, w, I);
        C[h] = f32_to_bf16(acc * down_scales[h]);
    }
#else
    for (size_t i = 0; i < I; i++) {
        const uint8_t *g = gate_up_W + i * gate_packed_k;
        const uint8_t *urow = gate_up_W + (I + i) * gate_packed_k;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        size_t k = 0;
        for (; k + 1 < K; k += 2) {
            uint8_t gp = g[k / 2];
            uint8_t up = urow[k / 2];
            float a0 = bf16_to_f32(A[k + 0]);
            float a1 = bf16_to_f32(A[k + 1]);
            gate_acc += a0 * (float)unpack_i4_signed(gp, 0);
            gate_acc += a1 * (float)unpack_i4_signed(gp, 1);
            up_acc += a0 * (float)unpack_i4_signed(up, 0);
            up_acc += a1 * (float)unpack_i4_signed(up, 1);
        }
        if (k < K) {
            gate_acc += bf16_to_f32(A[k]) * (float)unpack_i4_signed(g[k / 2], 0);
            up_acc += bf16_to_f32(A[k]) * (float)unpack_i4_signed(urow[k / 2], 0);
        }
        float gv = bf16_to_f32(f32_to_bf16(gate_acc * gate_up_scales[i]));
        float uv = bf16_to_f32(f32_to_bf16(up_acc * gate_up_scales[I + i]));
        float silu = gv / (1.0f + expf(-gv));
        hidden[i] = f32_to_bf16(silu * uv);
    }
    for (size_t h = 0; h < H; h++) {
        const uint8_t *w = down_W + h * down_packed_k;
        float acc = 0.0f;
        size_t k = 0;
        for (; k + 1 < I; k += 2) {
            uint8_t packed = w[k / 2];
            acc += bf16_to_f32(hidden[k + 0]) * (float)unpack_i4_signed(packed, 0);
            acc += bf16_to_f32(hidden[k + 1]) * (float)unpack_i4_signed(packed, 1);
        }
        if (k < I) {
            acc += bf16_to_f32(hidden[k]) * (float)unpack_i4_signed(w[k / 2], 0);
        }
        C[h] = f32_to_bf16(acc * down_scales[h]);
    }
#endif

    free(hidden);
    return ECPU_OK;
}

int ekernel_gated_delta_recurrent_f32(float *state,
                                      const float *q,
                                      const float *k,
                                      const float *v,
                                      const float *beta,
                                      const float *decay,
                                      float *out,
                                      size_t n_heads,
                                      size_t k_dim,
                                      size_t v_dim,
                                      float scale) {
    if (!state || !q || !k || !v || !beta || !decay || !out) return ECPU_ERR_PARAM;
    if (n_heads == 0 || k_dim == 0 || v_dim == 0) return ECPU_ERR_PARAM;

#if defined(ECPU_HAS_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (long long hh = 0; hh < (long long)n_heads; hh++) {
        size_t h = (size_t)hh;
        float *state_h = state + h * k_dim * v_dim;
        const float *q_h = q + h * k_dim;
        const float *k_h = k + h * k_dim;
        const float *v_h = v + h * v_dim;
        float *out_h = out + h * v_dim;
        float b = beta[h];
        float d = decay[h];

        float s_stack[256];
        float delta_stack[256];
        float *s = s_stack;
        float *delta = delta_stack;
        int heap = 0;
        if (v_dim > 256) {
            s = (float *)malloc(v_dim * sizeof(float));
            delta = (float *)malloc(v_dim * sizeof(float));
            heap = 1;
            if (!s || !delta) {
                if (s) free(s);
                if (delta) free(delta);
                continue;
            }
        }

        for (size_t vi = 0; vi < v_dim; vi++) {
            s[vi] = 0.0f;
            out_h[vi] = 0.0f;
        }

        for (size_t ki = 0; ki < k_dim; ki++) {
            float *row = state_h + ki * v_dim;
            float kval = k_h[ki];
            for (size_t vi = 0; vi < v_dim; vi++) {
                float decayed = row[vi] * d;
                row[vi] = decayed;
                s[vi] += decayed * kval;
            }
        }

        for (size_t vi = 0; vi < v_dim; vi++) {
            delta[vi] = (v_h[vi] - s[vi]) * b;
        }

        for (size_t ki = 0; ki < k_dim; ki++) {
            float *row = state_h + ki * v_dim;
            float kval = k_h[ki];
            float qval = q_h[ki];
            for (size_t vi = 0; vi < v_dim; vi++) {
                float updated = row[vi] + kval * delta[vi];
                row[vi] = updated;
                out_h[vi] += updated * qval;
            }
        }

        for (size_t vi = 0; vi < v_dim; vi++) {
            out_h[vi] *= scale;
        }

        if (heap) {
            free(s);
            free(delta);
        }
    }
    return ECPU_OK;
}

int ekernel_attention(const ekernel_attn_desc_t *desc,
                      const void *q, const void *k, const void *v,
                      const size_t *block_table,
                      void *out,
                      ekernel_isa_t isa) {
    if (!desc || !q || !k || !v || !out) return ECPU_ERR_PARAM;
    if (desc->dtype != ECPU_PRECISION_F32) return ECPU_ERR_UNSUPPORTED;
    (void)block_table;
    (void)isa;
    const float *qf = (const float *)q;
    const float *kf = (const float *)k;
    const float *vf = (const float *)v;
    float *of = (float *)out;
    const size_t H = desc->n_heads;
    const size_t Hd = desc->head_dim;
    const size_t QL = desc->q_len;
    const size_t KL = desc->kv_len;
    float *scores = (float *)malloc(QL * KL * sizeof(float));
    if (!scores) return ECPU_ERR_MEM;
    for (size_t h = 0; h < H; h++) {
        const float *qh = qf + h * QL * Hd;
        const float *kh = kf + h * KL * Hd;
        const float *vh = vf + h * KL * Hd;
        float *oh = of + h * QL * Hd;
        for (size_t qi = 0; qi < QL; qi++) {
            const float *qrow = qh + qi * Hd;
            float mx = -1e30f;
            for (size_t ki = 0; ki < KL; ki++) {
                if (desc->causal && ki > qi + (KL - QL)) continue;
                const float *krow = kh + ki * Hd;
                float s = 0.0f;
                for (size_t d = 0; d < Hd; d++) s += qrow[d] * krow[d];
                s *= desc->scale;
                scores[qi * KL + ki] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (size_t ki = 0; ki < KL; ki++) {
                float e = expf(scores[qi * KL + ki] - mx);
                scores[qi * KL + ki] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (size_t d = 0; d < Hd; d++) {
                float acc = 0.0f;
                for (size_t ki = 0; ki < KL; ki++) {
                    acc += scores[qi * KL + ki] * vh[ki * Hd + d];
                }
                oh[qi * Hd + d] = acc * inv;
            }
        }
    }
    free(scores);
    return ECPU_OK;
}
