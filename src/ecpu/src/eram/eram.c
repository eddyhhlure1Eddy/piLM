#include "../include/eram.h"
#include "../include/device/api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
typedef struct {
    HANDLE hFile;
    HANDLE hMap;
} eram_win_mmap_t;
#endif

struct eram_buffer {
    void  *data;
    size_t size;
    eram_flags_t flags;
    int   is_mmap;
#if defined(_WIN32)
    eram_win_mmap_t mmap;
#endif
};

eram_buffer_t *eram_buffer_alloc(size_t bytes, eram_flags_t flags) {
    const edevice_vtable_t *dev = edevice_active();
    if (!dev || !dev->alloc) return NULL;
    void *p = dev->alloc(bytes, 64);
    if (!p) return NULL;
    eram_buffer_t *buf = (eram_buffer_t *)calloc(1, sizeof(*buf));
    buf->data = p;
    buf->size = bytes;
    buf->flags = flags;
    return buf;
}

void eram_buffer_free(eram_buffer_t *buf) {
    if (!buf) return;
    if (buf->is_mmap) {
        eram_munmap(buf);
        return;
    }
    const edevice_vtable_t *dev = edevice_active();
    if (dev && dev->free) dev->free(buf->data);
    free(buf);
}

void  *eram_buffer_data(eram_buffer_t *buf) { return buf ? buf->data : NULL; }
size_t eram_buffer_size(eram_buffer_t *buf) { return buf ? buf->size : 0; }

eram_buffer_t *eram_mmap(const char *path, eram_flags_t flags) {
    (void)flags;
#if defined(_WIN32)
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER fsz;
    GetFileSizeEx(hFile, &fsz);
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); return NULL; }
    void *p = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(hMap); CloseHandle(hFile); return NULL; }
    eram_buffer_t *buf = (eram_buffer_t *)calloc(1, sizeof(*buf));
    buf->data = p;
    buf->size = (size_t)fsz.QuadPart;
    buf->flags = flags | ERAM_FLAG_MMAP;
    buf->is_mmap = 1;
    buf->mmap.hFile = hFile;
    buf->mmap.hMap = hMap;
    return buf;
#else
    (void)path;
    return NULL;
#endif
}

int eram_munmap(eram_buffer_t *buf) {
    if (!buf || !buf->is_mmap) return ECPU_ERR_PARAM;
#if defined(_WIN32)
    UnmapViewOfFile(buf->data);
    CloseHandle(buf->mmap.hMap);
    CloseHandle(buf->mmap.hFile);
    buf->data = NULL;
#endif
    free(buf);
    return ECPU_OK;
}

int eram_advise_random(eram_buffer_t *buf) {
    (void)buf;
    return ECPU_OK;
}

struct eram_kv_cache {
    size_t n_layers;
    size_t n_blocks;
    size_t n_kv_heads;
    size_t head_dim;
    ecpu_precision_t dtype;
    size_t elem_size;
    size_t layer_bytes;
    void **k_blocks;
    void **v_blocks;
    size_t used_blocks;
};

static size_t prec_size(ecpu_precision_t p) {
    switch (p) {
        case ECPU_PRECISION_F32: return 4;
        case ECPU_PRECISION_F16:
        case ECPU_PRECISION_BF16: return 2;
        case ECPU_PRECISION_F8_E4M3:
        case ECPU_PRECISION_F8_E5M2:
        case ECPU_PRECISION_I8: return 1;
        case ECPU_PRECISION_I4: return 1;
        default: return 4;
    }
}

eram_kv_cache_t *eram_kv_cache_alloc(size_t n_layers, size_t n_blocks,
                                      size_t n_kv_heads, size_t head_dim,
                                      ecpu_precision_t dtype) {
    const edevice_vtable_t *dev = edevice_active();
    if (!dev || !dev->alloc) return NULL;
    eram_kv_cache_t *c = (eram_kv_cache_t *)calloc(1, sizeof(*c));
    c->n_layers = n_layers;
    c->n_blocks = n_blocks;
    c->n_kv_heads = n_kv_heads;
    c->head_dim = head_dim;
    c->dtype = dtype;
    c->elem_size = prec_size(dtype);
    c->layer_bytes = (size_t)n_blocks * ERAM_BLOCK_SIZE * n_kv_heads * head_dim * c->elem_size;
    c->k_blocks = (void **)calloc(n_layers, sizeof(void *));
    c->v_blocks = (void **)calloc(n_layers, sizeof(void *));
    for (size_t l = 0; l < n_layers; l++) {
        c->k_blocks[l] = dev->alloc(c->layer_bytes, 64);
        c->v_blocks[l] = dev->alloc(c->layer_bytes, 64);
        if (!c->k_blocks[l] || !c->v_blocks[l]) {
            eram_kv_cache_free(c);
            return NULL;
        }
    }
    return c;
}

void eram_kv_cache_free(eram_kv_cache_t *cache) {
    if (!cache) return;
    const edevice_vtable_t *dev = edevice_active();
    if (cache->k_blocks) {
        for (size_t l = 0; l < cache->n_layers; l++)
            if (cache->k_blocks[l] && dev && dev->free) dev->free(cache->k_blocks[l]);
        free(cache->k_blocks);
    }
    if (cache->v_blocks) {
        for (size_t l = 0; l < cache->n_layers; l++)
            if (cache->v_blocks[l] && dev && dev->free) dev->free(cache->v_blocks[l]);
        free(cache->v_blocks);
    }
    free(cache);
}

int eram_kv_cache_write(eram_kv_cache_t *c, size_t layer,
                         const size_t *slot_ids, size_t n_slots,
                         const void *keys, const void *values) {
    if (!c || layer >= c->n_layers || !slot_ids) return ECPU_ERR_PARAM;
    const size_t block_bytes = (size_t)ERAM_BLOCK_SIZE * c->n_kv_heads * c->head_dim * c->elem_size;
    const char *kp = (const char *)keys;
    const char *vp = (const char *)values;
    char *kdst = (char *)c->k_blocks[layer];
    char *vdst = (char *)c->v_blocks[layer];
    for (size_t i = 0; i < n_slots; i++) {
        if (slot_ids[i] >= c->n_blocks) return ECPU_ERR_PARAM;
        memcpy(kdst + slot_ids[i] * block_bytes, kp + i * block_bytes, block_bytes);
        memcpy(vdst + slot_ids[i] * block_bytes, vp + i * block_bytes, block_bytes);
    }
    return ECPU_OK;
}

int eram_kv_cache_read(eram_kv_cache_t *c, size_t layer,
                        const size_t *slot_ids, size_t n_slots,
                        void *keys_out, void *values_out) {
    if (!c || layer >= c->n_layers || !slot_ids) return ECPU_ERR_PARAM;
    const size_t block_bytes = (size_t)ERAM_BLOCK_SIZE * c->n_kv_heads * c->head_dim * c->elem_size;
    char *kp = (char *)keys_out;
    char *vp = (char *)values_out;
    const char *ksrc = (const char *)c->k_blocks[layer];
    const char *vsrc = (const char *)c->v_blocks[layer];
    for (size_t i = 0; i < n_slots; i++) {
        if (slot_ids[i] >= c->n_blocks) return ECPU_ERR_PARAM;
        memcpy(kp + i * block_bytes, ksrc + slot_ids[i] * block_bytes, block_bytes);
        memcpy(vp + i * block_bytes, vsrc + slot_ids[i] * block_bytes, block_bytes);
    }
    return ECPU_OK;
}

size_t eram_kv_cache_block_size(eram_kv_cache_t *c) { return c ? ERAM_BLOCK_SIZE : 0; }
size_t eram_kv_cache_n_blocks(eram_kv_cache_t *c)   { return c ? c->n_blocks : 0; }
size_t eram_kv_cache_used_bytes(eram_kv_cache_t *c) {
    return c ? c->used_blocks * c->layer_bytes * 2 : 0;
}