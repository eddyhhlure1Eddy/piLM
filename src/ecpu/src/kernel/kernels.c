#include "../include/kernel/api.h"
#include "../include/device/api.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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
    (void)isa;
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float av = desc->transpose_a ? a[k * lda + m] : a[m * lda + k];
                float bv = desc->transpose_b ? b[n * ldb + k] : b[k * ldb + n];
                acc += av * bv;
            }
            c[m * N + n] = acc;
        }
    }
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