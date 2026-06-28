#include "../include/device/api.h"
#include <stdio.h>

static const char *CUDA_NAME = "cuda (reserved)";
static int cuda_dev_init(void) {
    printf("[edevice] cuda device is reserved for future, not yet implemented\n");
    return -1;
}
static int cuda_dev_shutdown(void) { return 0; }

static const edevice_vtable_t CUDA_DEVICE_VTABLE = {
    .name          = "cuda (reserved)",
    .id            = EDEV_CUDA,
    .caps          = EDEV_CAP_NONE,
    .alloc         = NULL,
    .free          = NULL,
    .memcpy_h2d    = NULL,
    .memcpy_d2h    = NULL,
    .memcpy_d2d    = NULL,
    .synchronize   = NULL,
    .mem_total     = NULL,
    .mem_free      = NULL,
    .init          = cuda_dev_init,
    .shutdown      = cuda_dev_shutdown,
};

void cuda_device_register(void) {
    edevice_register(EDEV_CUDA, &CUDA_DEVICE_VTABLE);
}