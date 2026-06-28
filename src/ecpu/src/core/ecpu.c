#include "ecpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ecpu_device_register(void);
void cuda_device_register(void);

static void register_all_devices(void) {
    static int done = 0;
    if (done) return;
    ecpu_device_register();
    cuda_device_register();
    done = 1;
}

static struct {
    int initialized;
    ecpu_config_t config;
    char err_buf[512];
} g_ecpu = {0};

static void set_err(const char *msg) {
    strncpy(g_ecpu.err_buf, msg, sizeof(g_ecpu.err_buf) - 1);
    g_ecpu.err_buf[sizeof(g_ecpu.err_buf) - 1] = '\0';
}

ecpu_config_t ecpu_config_default(void) {
    ecpu_config_t c = {0};
    c.n_threads = 0;
    c.numa_policy = 0;
    c.disable_amx = 0;
    c.disable_avx512 = 0;
    c.eram_budget_bytes = 0;
    c.kv_cache_budget_bytes = 4ULL * 1024 * 1024 * 1024;
    c.device_type = EDEV_ECPU;
    return c;
}

int ecpu_init(const ecpu_config_t *config) {
    if (g_ecpu.initialized) {
        set_err("ecpu already initialized");
        return ECPU_ERR_PARAM;
    }
    register_all_devices();
    g_ecpu.config = config ? *config : ecpu_config_default();

    if (edevice_init_all() != 0) {
        set_err("device init failed");
        return ECPU_ERR_DEVICE;
    }
    if (edevice_set_active(g_ecpu.config.device_type) != 0) {
        set_err("set active device failed");
        return ECPU_ERR_DEVICE;
    }

    const edevice_vtable_t *dev = edevice_active();
    if (!dev) {
        set_err("no active device");
        return ECPU_ERR_DEVICE;
    }

    g_ecpu.initialized = 1;
    printf("[ecpu] initialized, device=%s, isa=%s\n",
           dev->name,
           ekernel_isa_name(ekernel_detect_isa()));
    return ECPU_OK;
}

void ecpu_shutdown(void) {
    if (!g_ecpu.initialized) return;
    edevice_shutdown_all();
    memset(&g_ecpu, 0, sizeof(g_ecpu));
}

int ecpu_is_initialized(void) { return g_ecpu.initialized; }

const char *ecpu_version(void) { return ECPU_VERSION_STRING; }

const char *ecpu_last_error(void) {
    return g_ecpu.err_buf[0] ? g_ecpu.err_buf : NULL;
}