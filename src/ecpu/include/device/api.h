#ifndef EDEVICE_API_H
#define EDEVICE_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef enum {
    EDEV_ECPU  = 0,
    EDEV_CUDA  = 1,
    EDEV_COUNT
} edevice_t;

typedef enum {
    EDEV_CAP_NONE        = 0,
    EDEV_CAP_F16         = 1 << 0,
    EDEV_CAP_BF16        = 1 << 1,
    EDEV_CAP_FP8         = 1 << 2,
    EDEV_CAP_INT8_AMX    = 1 << 3,
    EDEV_CAP_AVX512      = 1 << 4,
    EDEV_CAP_AVX2        = 1 << 5,
    EDEV_CAP_NEON        = 1 << 6,
    EDEV_CAP_SVE         = 1 << 7,
    EDEV_CAP_NUMA        = 1 << 8,
    EDEV_CAP_UNIFIED_MEM = 1 << 9,
} edevice_caps_t;

typedef struct edevice_vtable {
    const char *name;
    edevice_t   id;
    uint64_t    caps;

    void  *(*alloc)(size_t bytes, size_t alignment);
    void   (*free)(void *ptr);
    void   (*memcpy_h2d)(void *dst, const void *src, size_t n);
    void   (*memcpy_d2h)(void *dst, const void *src, size_t n);
    void   (*memcpy_d2d)(void *dst, const void *src, size_t n);
    int    (*synchronize)(void);
    size_t (*mem_total)(void);
    size_t (*mem_free)(void);
    int    (*init)(void);
    int    (*shutdown)(void);
} edevice_vtable_t;

const edevice_vtable_t *edevice_get(edevice_t dev);
int   edevice_register(edevice_t dev, const edevice_vtable_t *vtable);
int   edevice_set_active(edevice_t dev);
const edevice_vtable_t *edevice_active(void);
int   edevice_init_all(void);
void  edevice_shutdown_all(void);

#ifdef __cplusplus
}
#endif

#endif