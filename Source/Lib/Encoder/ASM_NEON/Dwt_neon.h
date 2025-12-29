/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef _DWT_NEON_H_
#define _DWT_NEON_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void dwt_horizontal_line_neon(int32_t* out_lf, int32_t* out_hf, const int32_t* in, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif /* _DWT_NEON_H_ */
