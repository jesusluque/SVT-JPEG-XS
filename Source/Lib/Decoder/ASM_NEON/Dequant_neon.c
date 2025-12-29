/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include <arm_neon.h>
#include "Dequant.h"
#include "Codestream.h"

static void inv_quant_deadzone_neon(uint16_t* buf, uint32_t size, uint8_t* gclis, uint32_t group_size, uint8_t gtli) {
    if (gtli == 0) return;
    
    uint32_t i = 0;
    if (group_size == 4) {
        uint8x16_t v_gtli = vdupq_n_u8(gtli);
        uint16x8_t v_val_mask = vdupq_n_u16((uint16_t)~BITSTREAM_MASK_SIGN);
        uint16x8_t v_add_val = vdupq_n_u16(1 << (gtli - 1));

        for (; i + 16 <= size; i += 16) {
            uint32_t gcli_idx = i / 4;
            uint8_t g0 = gclis[gcli_idx];
            uint8_t g1 = gclis[gcli_idx+1];
            uint8_t g2 = gclis[gcli_idx+2];
            uint8_t g3 = gclis[gcli_idx+3];
            
            uint8x8_t v_g_lo = vcreate_u8((uint64_t)g0 | ((uint64_t)g0 << 8) | ((uint64_t)g0 << 16) | ((uint64_t)g0 << 24) |
                                          ((uint64_t)g1 << 32) | ((uint64_t)g1 << 40) | ((uint64_t)g1 << 48) | ((uint64_t)g1 << 56));
            uint8x8_t v_g_hi = vcreate_u8((uint64_t)g2 | ((uint64_t)g2 << 8) | ((uint64_t)g2 << 16) | ((uint64_t)g2 << 24) |
                                          ((uint64_t)g3 << 32) | ((uint64_t)g3 << 40) | ((uint64_t)g3 << 48) | ((uint64_t)g3 << 56));
            uint8x16_t v_g = vcombine_u8(v_g_lo, v_g_hi);
            
            uint8x16_t v_mask = vcgtq_u8(v_g, v_gtli);
            
            uint16x8_t v_mask_lo = vmovl_u8(vget_low_u8(v_mask));
            v_mask_lo = vmulq_n_u16(v_mask_lo, 0x0101);
            
            uint16x8_t v_mask_hi = vmovl_u8(vget_high_u8(v_mask));
            v_mask_hi = vmulq_n_u16(v_mask_hi, 0x0101);
            
            uint16x8_t v_coeff_lo = vld1q_u16(buf + i);
            uint16x8_t v_coeff_hi = vld1q_u16(buf + i + 8);
            
            // Check if coeff != 0 (ignoring sign)
            uint16x8_t v_nz_lo = vtstq_u16(v_coeff_lo, v_val_mask);
            uint16x8_t v_nz_hi = vtstq_u16(v_coeff_hi, v_val_mask);
            
            // Combine conditions: (gcli > gtli) && (coeff != 0)
            uint16x8_t v_cond_lo = vandq_u16(v_mask_lo, v_nz_lo);
            uint16x8_t v_cond_hi = vandq_u16(v_mask_hi, v_nz_hi);
            
            // Add value where condition is true
            v_coeff_lo = vorrq_u16(v_coeff_lo, vandq_u16(v_add_val, v_cond_lo));
            v_coeff_hi = vorrq_u16(v_coeff_hi, vandq_u16(v_add_val, v_cond_hi));
            
            vst1q_u16(buf + i, v_coeff_lo);
            vst1q_u16(buf + i + 8, v_coeff_hi);
        }
    }
    
    // Scalar fallback
    for (; i < size; i++) {
        int8_t gcli = gclis[i / group_size];
        if ((gcli > gtli) && (buf[i] & ~BITSTREAM_MASK_SIGN)) {
            buf[i] |= (1 << (gtli - 1));
        }
    }
}

static void inv_quant_uniform_neon(uint16_t* buf, uint32_t size, uint8_t* gclis, uint32_t group_size, uint8_t gtli) {
    if (gtli == 0) return;
    // Scalar fallback
    for (uint32_t coeff_idx = 0; coeff_idx < size; coeff_idx++) {
        int8_t gcli = gclis[coeff_idx / group_size];
        if ((gcli > gtli) && (buf[coeff_idx] & ~BITSTREAM_MASK_SIGN)) {
            uint16_t sign = buf[coeff_idx] & BITSTREAM_MASK_SIGN;
            uint16_t val = (buf[coeff_idx] & ~BITSTREAM_MASK_SIGN);
            uint8_t scale_value = gcli - gtli + 1;
            buf[coeff_idx] = 0;
            for (; val > 0; val >>= scale_value) {
                buf[coeff_idx] += val;
            }
            buf[coeff_idx] |= sign;
        }
    }
}

void dequant_neon(uint16_t* buf, uint32_t size, uint8_t* gclis, uint32_t group_size, uint8_t gtli, QUANT_TYPE dq_type) {
    switch (dq_type) {
    case QUANT_TYPE_UNIFORM:
        inv_quant_uniform_neon(buf, size, gclis, group_size, gtli);
        return;
    case QUANT_TYPE_DEADZONE:
        inv_quant_deadzone_neon(buf, size, gclis, group_size, gtli);
        return;
    default:
        break;
    }
}
