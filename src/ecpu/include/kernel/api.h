#ifndef EKERNEL_API_H
#define EKERNEL_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "../ecpu_base.h"
#include "../eram.h"

typedef enum {
    EKERNEL_ISA_SCALAR  = 0,
    EKERNEL_ISA_AVX2    = 1,
    EKERNEL_ISA_AVX512  = 2,
    EKERNEL_ISA_AMX     = 3,
    EKERNEL_ISA_NEON    = 4,
    EKERNEL_ISA_SVE     = 5,
} ekernel_isa_t;

ekernel_isa_t ekernel_detect_isa(void);
const char   *ekernel_isa_name(ekernel_isa_t isa);

typedef struct {
    size_t M, N, K;
    ecpu_precision_t a_prec;
    ecpu_precision_t b_prec;
    ecpu_precision_t out_prec;
    int   transpose_a;
    int   transpose_b;
} ekernel_gemm_desc_t;

int ekernel_gemm(const ekernel_gemm_desc_t *desc,
                 const void *A, const void *B,
                 void *C,
                 ekernel_isa_t isa);

typedef struct {
    size_t M, N, K;
} ekernel_linear_w8a32_desc_t;

typedef struct {
    size_t M;
    size_t hidden_size;
    size_t intermediate_size;
    size_t gate_up_in_features;
    size_t down_in_features;
} ekernel_w4a16_swiglu_desc_t;

int ekernel_linear_w8a32(const ekernel_linear_w8a32_desc_t *desc,
                         const float *A,
                         const int8_t *W,
                         const float *scales,
                         float *C,
                         ekernel_isa_t isa);

int ekernel_linear_w8a16_bf16(const ekernel_linear_w8a32_desc_t *desc,
                              const uint16_t *A,
                              const int8_t *W,
                              const float *scales,
                              uint16_t *C,
                              ekernel_isa_t isa);

int ekernel_linear_w8a16_bf16_argmax(const ekernel_linear_w8a32_desc_t *desc,
                                     const uint16_t *A,
                                     const int8_t *W,
                                     const float *scales,
                                     size_t *out_index,
                                     float *out_value,
                                     ekernel_isa_t isa);

int ekernel_linear_w8a16_bf16_i8b8(const ekernel_linear_w8a32_desc_t *desc,
                                   const uint16_t *A,
                                   const int8_t *W_interleaved,
                                   const float *scales,
                                   uint16_t *C,
                                   ekernel_isa_t isa);

/* Experimental W8A16 BF16 linear with on-the-fly int8 activation quantization.
 *
 * Same int8 per-output-row-scaled weight layout as ekernel_linear_w8a16_bf16,
 * so it is a drop-in alternative for the same tensors (qweight int8 [N,K],
 * scales fp32 [N]).  The activation row is quantized to signed int8 with a
 * per-row fp32 scale at call time, and the dot product uses the AVX2
 * _mm256_maddubs_epi16 path (32 int8*int8->int16 macs per instruction).  The
 * only added error versus the stable W8 path is the int8 activation rounding;
 * the int8 weights are exact and the int32 accumulation is exact.  Because W8
 * weights are far more precise than W4 weights, this path is expected to
 * absorb the int8 activation error without perturbing greedy sampling (unlike
 * the W4+q8 path). */
int ekernel_linear_w8a16_bf16_q8(const ekernel_linear_w8a32_desc_t *desc,
                                 const uint16_t *A,
                                 const int8_t *W,
                                 const float *scales,
                                 uint16_t *C,
                                 ekernel_isa_t isa);

int ekernel_linear_w4a16_bf16(const ekernel_linear_w8a32_desc_t *desc,
                              const uint16_t *A,
                              const uint8_t *W,
                              const float *scales,
                              uint16_t *C,
                              ekernel_isa_t isa);

int ekernel_linear_w4a16g32_bf16(const ekernel_linear_w8a32_desc_t *desc,
                                 const uint16_t *A,
                                 const uint8_t *W,
                                 const float *scales,
                                 uint16_t *C,
                                 ekernel_isa_t isa);

int ekernel_linear_w4a16g128_bf16(const ekernel_linear_w8a32_desc_t *desc,
                                  const uint16_t *A,
                                  const uint8_t *W,
                                  const float *scales,
                                  uint16_t *C,
                                  ekernel_isa_t isa);

/* Experimental W4A16 BF16 linear that quantizes the activation row to signed
 * int8 on each call and uses an integer (pmaddwd) dot product.  The packed
 * int4 weight layout and per-output-row fp32 scales are identical to
 * ekernel_linear_w4a16_bf16, so this is a drop-in alternative kernel for the
 * same tensors.  The activation int8 quantization is the only error added on
 * top of the existing W4 weight quantization. */
int ekernel_linear_w4a16_bf16_q8(const ekernel_linear_w8a32_desc_t *desc,
                                 const uint16_t *A,
                                 const uint8_t *W,
                                 const float *scales,
                                 uint16_t *C,
                                 ekernel_isa_t isa);

/* Experimental W4A16 BF16 linear for decode-time benchmarking with weights
 * pre-unpacked to signed int8 and interleaved in 8-output-row blocks:
 * W_interleaved[ceil(N/8), K, 8].  The int4 values and per-row fp32 scales are
 * identical to ekernel_linear_w4a16_bf16; only the storage layout changes so
 * decode can skip per-call nibble unpacking.  Currently supports M=1. */
int ekernel_linear_w4a16_bf16_i4b8(const ekernel_linear_w8a32_desc_t *desc,
                                   const uint16_t *A,
                                   const int8_t *W_interleaved,
                                   const float *scales,
                                   uint16_t *C,
                                   ekernel_isa_t isa);

/* Experimental W4A16 BF16 linear for decode-time benchmarking with packed
 * int4 weights kept at 4-bit density but blocked as:
 * W_blocked[ceil(N/8), ceil(K/16), 8 output rows, 8 packed K bytes].
 * This keeps each 8-row/16-K tile contiguous so the AVX2 kernel can reuse the
 * activation conversion while avoiding strided row loads.  Supports M=1. */
int ekernel_linear_w4a16_bf16_b8(const ekernel_linear_w8a32_desc_t *desc,
                                 const uint16_t *A,
                                 const uint8_t *W_blocked,
                                 const float *scales,
                                 uint16_t *C,
                                 ekernel_isa_t isa);

typedef struct {
    size_t           n_heads;
    size_t           n_kv_heads;
    size_t           head_dim;
    size_t           q_len;
    size_t           kv_len;
    size_t           block_size;
    ecpu_precision_t dtype;
    float            scale;
    float            softcap;
    int              causal;
} ekernel_attn_desc_t;

int ekernel_attention(const ekernel_attn_desc_t *desc,
                      const void *q, const void *k, const void *v,
                      const size_t *block_table,
                      void *out,
                      ekernel_isa_t isa);

int ekernel_rmsnorm(const void *x, void *out, const void *weight,
                    size_t n, float eps, ecpu_precision_t dtype);

int ekernel_rmsnorm_bf16(const uint16_t *x,
                         uint16_t *out,
                         const uint16_t *weight,
                         size_t rows,
                         size_t n,
                         float eps,
                         int add_one);

int ekernel_rope(void *x, size_t n, size_t head_dim, size_t pos,
                 float theta, ecpu_precision_t dtype);

int ekernel_softmax(void *x, size_t n, float scale, ecpu_precision_t dtype);

int ekernel_silu(const void *x, void *out, size_t n, ecpu_precision_t dtype);

int ekernel_mul(const void *a, const void *b, void *out, size_t n,
                ecpu_precision_t dtype);

int ekernel_swiglu_bf16(const uint16_t *gate,
                        const uint16_t *up,
                        uint16_t *out,
                        size_t n);

/* Experimental decode-only W4A16 SwiGLU MLP fusion:
 * gate/up W4 linears -> BF16 SwiGLU activation -> down W4 linear in one C call.
 * Supports M=1 and per-row W4 scales.  It is intentionally opt-in so it can be
 * benchmarked from an isolated DLL without changing the stable W4 service DLL. */
int ekernel_swiglu_w4a16_bf16(const ekernel_w4a16_swiglu_desc_t *desc,
                              const uint16_t *A,
                              const uint8_t *gate_up_W,
                              const float *gate_up_scales,
                              const uint8_t *down_W,
                              const float *down_scales,
                              uint16_t *C,
                              ekernel_isa_t isa);

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
                                      float scale);

#ifdef __cplusplus
}
#endif

#endif
