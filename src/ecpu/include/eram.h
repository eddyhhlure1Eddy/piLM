#ifndef ERAM_H
#define ERAM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ecpu_base.h"

#define ERAM_BLOCK_SIZE 128

typedef struct eram_buffer eram_buffer_t;
typedef struct eram_kv_cache eram_kv_cache_t;

typedef enum {
    ERAM_FLAG_NONE        = 0,
    ERAM_FLAG_MMAP        = 1 << 0,
    ERAM_FLAG_LOCK        = 1 << 1,
    ERAM_FLAG_HUGE_PAGES  = 1 << 2,
    ERAM_FLAG_NUMA_LOCAL  = 1 << 3,
} eram_flags_t;

typedef enum {
    ERAM_QUANT_NONE = 0,
    ERAM_QUANT_Q4_0,
    ERAM_QUANT_Q4_K_M,
    ERAM_QUANT_Q5_K_M,
    ERAM_QUANT_Q8_0,
    ERAM_QUANT_BF16,
    ERAM_QUANT_FP8_E4M3,
    ERAM_QUANT_INT8,
} eram_quant_t;

eram_buffer_t *eram_buffer_alloc(size_t bytes, eram_flags_t flags);
void           eram_buffer_free(eram_buffer_t *buf);
void          *eram_buffer_data(eram_buffer_t *buf);
size_t         eram_buffer_size(eram_buffer_t *buf);

eram_buffer_t *eram_mmap(const char *path, eram_flags_t flags);
int            eram_munmap(eram_buffer_t *buf);
int            eram_advise_random(eram_buffer_t *buf);

eram_kv_cache_t *eram_kv_cache_alloc(size_t n_layers,
                                      size_t n_blocks,
                                      size_t n_kv_heads,
                                      size_t head_dim,
                                      ecpu_precision_t dtype);
void             eram_kv_cache_free(eram_kv_cache_t *cache);
int              eram_kv_cache_write(eram_kv_cache_t *cache,
                                      size_t layer,
                                      const size_t *slot_ids,
                                      size_t n_slots,
                                      const void *keys,
                                      const void *values);
int              eram_kv_cache_read(eram_kv_cache_t *cache,
                                     size_t layer,
                                     const size_t *slot_ids,
                                     size_t n_slots,
                                     void *keys_out,
                                     void *values_out);
size_t           eram_kv_cache_block_size(eram_kv_cache_t *cache);
size_t           eram_kv_cache_n_blocks(eram_kv_cache_t *cache);
size_t           eram_kv_cache_used_bytes(eram_kv_cache_t *cache);

#ifdef __cplusplus
}
#endif

#endif