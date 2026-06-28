#ifndef ECPU_H
#define ECPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "ecpu_base.h"
#include "eram.h"
#include "device/api.h"
#include "kernel/api.h"

#define ECPU_VERSION_MAJOR 0
#define ECPU_VERSION_MINOR 1
#define ECPU_VERSION_PATCH 0

#define ECPU_VERSION_STRING "0.1.0"

typedef struct ecpu_context ecpu_context_t;

typedef struct {
    int      n_threads;
    int      numa_policy;
    int      disable_amx;
    int      disable_avx512;
    size_t   eram_budget_bytes;
    size_t   kv_cache_budget_bytes;
    edevice_t device_type;
} ecpu_config_t;

ecpu_config_t ecpu_config_default(void);

int     ecpu_init(const ecpu_config_t *config);
void    ecpu_shutdown(void);
int     ecpu_is_initialized(void);
const char *ecpu_version(void);
const char *ecpu_last_error(void);

#ifdef __cplusplus
}
#endif

#endif