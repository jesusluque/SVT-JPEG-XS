/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef _QUANT_NEON_H_
#define _QUANT_NEON_H_

#include "Definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

void quantization_neon(uint16_t* coeff_16bit, uint32_t size, uint8_t* gclis, uint32_t group_size, uint8_t gtli,
                    QUANT_TYPE quant_type);

#ifdef __cplusplus
}
#endif

#endif /*_QUANT_NEON_H_*/
