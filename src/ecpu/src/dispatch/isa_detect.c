#include "../include/kernel/api.h"
#include "../include/device/api.h"
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__clang__) || defined(__GNUC__)
#include <immintrin.h>
#define ECPU_HAS_CPUID 1
#endif
#endif

ekernel_isa_t ekernel_detect_isa(void) {
#if defined(ECPU_HAS_CPUID)
#if defined(__AMX_INT8__)
    return EKERNEL_ISA_AMX;
#elif defined(__AVX512F__)
    return EKERNEL_ISA_AVX512;
#elif defined(__AVX2__)
    return EKERNEL_ISA_AVX2;
#endif
#endif
#if defined(__ARM_NEON)
    return EKERNEL_ISA_NEON;
#endif
#if defined(__ARM_FEATURE_SVE)
    return EKERNEL_ISA_SVE;
#endif
    return EKERNEL_ISA_SCALAR;
}

const char *ekernel_isa_name(ekernel_isa_t isa) {
    switch (isa) {
        case EKERNEL_ISA_SCALAR: return "scalar";
        case EKERNEL_ISA_AVX2:   return "avx2";
        case EKERNEL_ISA_AVX512: return "avx512";
        case EKERNEL_ISA_AMX:    return "amx";
        case EKERNEL_ISA_NEON:   return "neon";
        case EKERNEL_ISA_SVE:    return "sve";
        default: return "unknown";
    }
}