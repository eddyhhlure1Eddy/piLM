#include "../include/device/api.h"
#include <stdlib.h>
#include <string.h>

static edevice_vtable_t g_registry[EDEV_COUNT] = {0};
static edevice_t g_active = EDEV_ECPU;
static int g_inited[EDEV_COUNT] = {0};

const edevice_vtable_t *edevice_get(edevice_t dev) {
    if (dev < 0 || dev >= EDEV_COUNT) return NULL;
    return g_registry[dev].name ? &g_registry[dev] : NULL;
}

int edevice_register(edevice_t dev, const edevice_vtable_t *vtable) {
    if (dev < 0 || dev >= EDEV_COUNT || !vtable) return -1;
    g_registry[dev] = *vtable;
    return 0;
}

int edevice_set_active(edevice_t dev) {
    if (dev < 0 || dev >= EDEV_COUNT || !g_registry[dev].name) return -1;
    g_active = dev;
    return 0;
}

const edevice_vtable_t *edevice_active(void) {
    return g_registry[g_active].name ? &g_registry[g_active] : NULL;
}

int edevice_init_all(void) {
    int inited_any = 0;
    for (int i = 0; i < EDEV_COUNT; i++) {
        if (g_registry[i].name && g_registry[i].init && !g_inited[i]) {
            if (g_registry[i].init() == 0) {
                g_inited[i] = 1;
                inited_any = 1;
            }
        }
    }
    return inited_any ? 0 : -1;
}

void edevice_shutdown_all(void) {
    for (int i = 0; i < EDEV_COUNT; i++) {
        if (g_inited[i] && g_registry[i].shutdown) {
            g_registry[i].shutdown();
        }
        g_inited[i] = 0;
    }
    memset(g_registry, 0, sizeof(g_registry));
    g_active = EDEV_ECPU;
}