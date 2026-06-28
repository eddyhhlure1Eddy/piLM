#include "ecpu.h"
#include <stdio.h>
#include <assert.h>

extern void ecpu_device_register(void);
extern void cuda_device_register(void);

int main(void) {
    ecpu_device_register();
    cuda_device_register();

    ecpu_config_t cfg = ecpu_config_default();
    int rc = ecpu_init(&cfg);
    printf("ecpu_init: %d, version: %s\n", rc, ecpu_version());
    if (rc != ECPU_OK) { printf("err: %s\n", ecpu_last_error()); return 1; }

    const edevice_vtable_t *dev = edevice_active();
    printf("active device: %s, mem_total: %zu MB, mem_free: %zu MB\n",
           dev->name, dev->mem_total() / (1024*1024), dev->mem_free() / (1024*1024));

    ekernel_isa_t isa = ekernel_detect_isa();
    printf("detected ISA: %s\n", ekernel_isa_name(isa));

    void *p = dev->alloc(1024, 64);
    assert(p);
    printf("alloc 1024 bytes OK: %p\n", p);
    dev->free(p);

    eram_kv_cache_t *kvc = eram_kv_cache_alloc(2, 16, 4, 64, ECPU_PRECISION_F32);
    assert(kvc);
    printf("kv_cache: blocks=%zu, block_size=%zu\n",
           eram_kv_cache_n_blocks(kvc), eram_kv_cache_block_size(kvc));
    eram_kv_cache_free(kvc);

    float A[4] = {1,2,3,4};
    float B[4] = {5,6,7,8};
    float C[4] = {0};
    ekernel_gemm_desc_t g = {2,2,2,ECPU_PRECISION_F32,ECPU_PRECISION_F32,ECPU_PRECISION_F32,0,0};
    rc = ekernel_gemm(&g, A, B, C, isa);
    printf("gemm result: [[%.1f,%.1f],[%.1f,%.1f]] (expect [[19,22],[43,50]])\n",
           C[0],C[1],C[2],C[3]);

    ecpu_shutdown();
    printf("ecpu_shutdown OK\n");
    return 0;
}