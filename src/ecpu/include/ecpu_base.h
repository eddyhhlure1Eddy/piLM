#ifndef ECPU_BASE_H
#define ECPU_BASE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define ECPU_OK         0
#define ECPU_ERR_PARAM  (-1)
#define ECPU_ERR_MEM    (-2)
#define ECPU_ERR_DEVICE (-3)
#define ECPU_ERR_KERNEL (-4)
#define ECPU_ERR_UNSUPPORTED (-5)

typedef enum {
    ECPU_PRECISION_F32  = 0,
    ECPU_PRECISION_F16  = 1,
    ECPU_PRECISION_BF16 = 2,
    ECPU_PRECISION_F8_E4M3  = 3,
    ECPU_PRECISION_F8_E5M2  = 4,
    ECPU_PRECISION_I8   = 5,
    ECPU_PRECISION_I4   = 6,
} ecpu_precision_t;

#ifdef __cplusplus
}
#endif

#endif