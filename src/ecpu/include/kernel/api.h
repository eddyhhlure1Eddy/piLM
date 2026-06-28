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

int ekernel_rope(void *x, size_t n, size_t head_dim, size_t pos,
                 float theta, ecpu_precision_t dtype);

int ekernel_softmax(void *x, size_t n, float scale, ecpu_precision_t dtype);

int ekernel_silu(const void *x, void *out, size_t n, ecpu_precision_t dtype);

int ekernel_mul(const void *a, const void *b, void *out, size_t n,
                ecpu_precision_t dtype);

#ifdef __cplusplus
}
#endif

#endif