/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include <arm_neon.h>
#include "Quant.h"
#include "Codestream.h"

static void quant_deadzone_neon(uint16_t* buff_16bit, uint32_t size, uint8_t* gclis, uint32_t group_size, uint8_t gtli) {
    uint32_t i = 0;
    if (group_size == 4) {
        uint8x16_t v_gtli = vdupq_n_u8(gtli);
        uint16x8_t v_sign_mask = vdupq_n_u16(BITSTREAM_MASK_SIGN);
        uint16x8_t v_val_mask = vdupq_n_u16((uint16_t)~BITSTREAM_MASK_SIGN);
        int16x8_t v_neg_gtli = vdupq_n_s16(-(int16_t)gtli);
        int16x8_t v_pos_gtli = vdupq_n_s16((int16_t)gtli);

        for (; i + 16 <= size; i += 16) {
            // Load 4 gclis (for 16 coeffs)
            // gclis[0] -> coeffs[0..3]
            // gclis[1] -> coeffs[4..7]
            // gclis[2] -> coeffs[8..11]
            // gclis[3] -> coeffs[12..15]
            
            // We need to load 4 bytes of gclis.
            // Since we might not be aligned or have enough data, be careful.
            // But usually gclis buffer is large enough.
            // Let's load 4 bytes.
            uint32_t gcli_idx = i / 4;
            // We can't easily load 4 bytes and expand to 16 bytes with simple intrinsics without some shuffling.
            // Or we can load 8 bytes (vld1_u8) and use the first 4.
            
            uint8_t g0 = gclis[gcli_idx];
            uint8_t g1 = gclis[gcli_idx+1];
            uint8_t g2 = gclis[gcli_idx+2];
            uint8_t g3 = gclis[gcli_idx+3];
            
            // Construct vector of gclis expanded to match coeffs
            // We process 16 coeffs in two passes of 8.
            // First 8 coeffs use g0 (4 times) and g1 (4 times).
            // Second 8 coeffs use g2 (4 times) and g3 (4 times).
            
            uint8x8_t v_g_lo = vcreate_u8((uint64_t)g0 | ((uint64_t)g0 << 8) | ((uint64_t)g0 << 16) | ((uint64_t)g0 << 24) |
                                          ((uint64_t)g1 << 32) | ((uint64_t)g1 << 40) | ((uint64_t)g1 << 48) | ((uint64_t)g1 << 56));
            
            uint8x8_t v_g_hi = vcreate_u8((uint64_t)g2 | ((uint64_t)g2 << 8) | ((uint64_t)g2 << 16) | ((uint64_t)g2 << 24) |
                                          ((uint64_t)g3 << 32) | ((uint64_t)g3 << 40) | ((uint64_t)g3 << 48) | ((uint64_t)g3 << 56));

            uint8x16_t v_g = vcombine_u8(v_g_lo, v_g_hi);
            
            // Compare gcli > gtli
            uint8x16_t v_mask = vcgtq_u8(v_g, v_gtli);
            
            // Load coeffs
            uint16x8_t v_coeff_lo = vld1q_u16(buff_16bit + i);
            uint16x8_t v_coeff_hi = vld1q_u16(buff_16bit + i + 8);
            
            // Split mask to lo/hi
            uint16x8_t v_mask_lo = vmovl_u8(vget_low_u8(v_mask)); // 0x00 or 0xFF -> 0x0000 or 0x00FF. Wait, we need 0xFFFF.
            // vcgtq_u8 returns 0xFF for true. vmovl_u8 expands to 0x00FF.
            // We want 0xFFFF.
            v_mask_lo = vmulq_n_u16(v_mask_lo, 0x0101); // 0x00FF * 0x0101 = 0xFFFF (approx? No. 255 * 257 = 65535). Yes.
            
            uint16x8_t v_mask_hi = vmovl_u8(vget_high_u8(v_mask));
            v_mask_hi = vmulq_n_u16(v_mask_hi, 0x0101);

            // Process LO
            {
                uint16x8_t v_sign = vandq_u16(v_coeff_lo, v_sign_mask);
                uint16x8_t v_abs = vandq_u16(v_coeff_lo, v_val_mask);
                
                // ((d >> gtli) << gtli)
                // vshlq_u16 with negative value shifts right.
                uint16x8_t v_quant = vshlq_u16(v_abs, v_neg_gtli);
                v_quant = vshlq_u16(v_quant, v_pos_gtli);
                
                // Restore sign if non-zero
                uint16x8_t v_nz_mask = vcgtq_u16(v_quant, vdupq_n_u16(0));
                v_quant = vorrq_u16(v_quant, vandq_u16(v_sign, v_nz_mask));
                
                // Apply gcli > gtli mask
                v_coeff_lo = vandq_u16(v_quant, v_mask_lo);
            }

            // Process HI
            {
                uint16x8_t v_sign = vandq_u16(v_coeff_hi, v_sign_mask);
                uint16x8_t v_abs = vandq_u16(v_coeff_hi, v_val_mask);
                
                uint16x8_t v_quant = vshlq_u16(v_abs, v_neg_gtli);
                v_quant = vshlq_u16(v_quant, v_pos_gtli);
                
                uint16x8_t v_nz_mask = vcgtq_u16(v_quant, vdupq_n_u16(0));
                v_quant = vorrq_u16(v_quant, vandq_u16(v_sign, v_nz_mask));
                
                v_coeff_hi = vandq_u16(v_quant, v_mask_hi);
            }
            
            vst1q_u16(buff_16bit + i, v_coeff_lo);
            vst1q_u16(buff_16bit + i + 8, v_coeff_hi);
        }
    }

    // Scalar fallback
    for (; i < size; i++) {
        uint8_t gcli = gclis[i / group_size];
        if (gcli > gtli) {
            uint16_t sign = buff_16bit[i] & BITSTREAM_MASK_SIGN;
            buff_16bit[i] = ((buff_16bit[i] & ~BITSTREAM_MASK_SIGN) >> gtli) << gtli;
            if (buff_16bit[i]) {
                buff_16bit[i] |= sign;
            }
        }
        else {
            buff_16bit[i] = 0;
        }
    }
}

static void quant_uniform_neon(uint16_t* buff_16bit, uint32_t size, uint8_t* gclis, uint32_t group_size, uint8_t gtli) {
    // Fallback to C for now as it's more complex and deadzone is default
    // Or implement scalar loop here to avoid linking issues if I don't expose the C function
    for (uint32_t coeff_idx = 0; coeff_idx < size; coeff_idx++) {
        uint8_t gcli = gclis[coeff_idx / group_size];
        if (gcli > gtli) {
            uint16_t sign = buff_16bit[coeff_idx] & BITSTREAM_MASK_SIGN;
            uint8_t scale_value = gcli - gtli + 1;
            uint16_t d = buff_16bit[coeff_idx] & ~BITSTREAM_MASK_SIGN;
            d = ((d << scale_value) - d + (1 << gcli)) >> (gcli + 1);
            buff_16bit[coeff_idx] = d << gtli;
            if (buff_16bit[coeff_idx]) {
                buff_16bit[coeff_idx] |= sign;
            }
        }
        else {
            buff_16bit[coeff_idx] = 0;
        }
    }
}

void quantization_neon(uint16_t* coeff_16bit, uint32_t size, uint8_t* gclis, uint32_t group_size, uint8_t gtli,
                    QUANT_TYPE quant_type) {
    switch (quant_type) {
    case QUANT_TYPE_UNIFORM:
        quant_uniform_neon(coeff_16bit, size, gclis, group_size, gtli);
        return;
    case QUANT_TYPE_DEADZONE:
        quant_deadzone_neon(coeff_16bit, size, gclis, group_size, gtli);
        return;
    default:
        break;
    }
    return;
}
