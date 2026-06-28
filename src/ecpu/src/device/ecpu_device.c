#include "../include/device/api.h"
#include "../include/ecpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
static size_t get_total_mem_win(void) {
    MEMORYSTATUSEX st = {sizeof(st)};
    GlobalMemoryStatusEx(&st);
    return (size_t)st.ullTotalPhys;
}
static size_t get_free_mem_win(void) {
    MEMORYSTATUSEX st = {sizeof(st)};
    GlobalMemoryStatusEx(&st);
    return (size_t)st.ullAvailPhys;
}
#else
#include <sys/sysinfo.h>
static size_t get_total_mem_win(void) {
    struct sysinfo si;
    if (sysinfo(&si) == 0) return (size_t)si.totalram * si.mem_unit;
    return 0;
}
static size_t get_free_mem_win(void) {
    struct sysinfo si;
    if (sysinfo(&si) == 0) return (size_t)si.freeram * si.mem_unit;
    return 0;
}
#endif

static void *ecpu_alloc(size_t bytes, size_t alignment) {
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void *p = NULL;
    if (posix_memalign(&p, alignment, bytes) != 0) return NULL;
    return p;
#endif
}

static void ecpu_free(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static void ecpu_memcpy_h2d(void *dst, const void *src, size_t n) {
    memcpy(dst, src, n);
}
static void ecpu_memcpy_d2h(void *dst, const void *src, size_t n) {
    memcpy(dst, src, n);
}
static void ecpu_memcpy_d2d(void *dst, const void *src, size_t n) {
    memcpy(dst, src, n);
}
static int ecpu_sync(void) { return 0; }
static size_t ecpu_mem_total(void) { return get_total_mem_win(); }
static size_t ecpu_mem_free(void) { return get_free_mem_win(); }

static int ecpu_dev_init(void) {
    return 0;
}
static int ecpu_dev_shutdown(void) { return 0; }

static const edevice_vtable_t ECPU_DEVICE_VTABLE = {
    .name          = "ecpu",
    .id            = EDEV_ECPU,
    .caps          = EDEV_CAP_BF16,
    .alloc         = ecpu_alloc,
    .free          = ecpu_free,
    .memcpy_h2d    = ecpu_memcpy_h2d,
    .memcpy_d2h    = ecpu_memcpy_d2h,
    .memcpy_d2d    = ecpu_memcpy_d2d,
    .synchronize   = ecpu_sync,
    .mem_total     = ecpu_mem_total,
    .mem_free      = ecpu_mem_free,
    .init          = ecpu_dev_init,
    .shutdown      = ecpu_dev_shutdown,
};

void ecpu_device_register(void) {
    edevice_register(EDEV_ECPU, &ECPU_DEVICE_VTABLE);
}