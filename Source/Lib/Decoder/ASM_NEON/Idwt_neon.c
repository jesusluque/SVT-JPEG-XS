/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "Idwt_neon.h"
#include <arm_neon.h>
#include <assert.h>

static void idwt_horizontal_line_lf16_hf16_c_fallback(const int16_t* in_lf, const int16_t* in_hf, int32_t* out, uint32_t len, uint8_t shift) {
    assert((len >= 2) && "[idwt_c()] ERROR: Length is too small!");

    out[0] = ((int32_t)in_lf[0] << shift) - ((((int32_t)in_hf[0] << shift) + 1) >> 1);

    for (uint32_t i = 1; i < len - 2; i += 2) {
        out[2] = ((int32_t)in_lf[1] << shift) - ((((int32_t)in_hf[0] << shift) + ((int32_t)in_hf[1] << shift) + 2) >> 2);
        out[1] = ((int32_t)in_hf[0] << shift) + ((out[0] + out[2]) >> 1);
        in_lf++;
        in_hf++;
        out += 2;
    }
    if (len & 1) {
        out[2] = ((int32_t)in_lf[1] << shift) - ((((int32_t)in_hf[0] << shift) + 1) >> 1);
        out[1] = ((int32_t)in_hf[0] << shift) + ((out[0] + out[2]) >> 1);
    }
    else { //!(len & 1)
        out[1] = ((int32_t)in_hf[0] << shift) + out[0];
    }
}

void idwt_horizontal_line_lf16_hf16_neon(const int16_t* in_lf, const int16_t* in_hf, int32_t* out, uint32_t len, uint8_t shift) {
    if (len < 16) {
        idwt_horizontal_line_lf16_hf16_c_fallback(in_lf, in_hf, out, len, shift);
        return;
    }

    // Initial boundary
    int32_t E_prev = ((int32_t)in_lf[0] << shift) - ((((int32_t)in_hf[0] << shift) + 1) >> 1);
    out[0] = E_prev;

    // Pointers
    const int16_t* lf_ptr = in_lf + 1;
    const int16_t* hf_ptr = in_hf;
    int32_t* out_ptr = out; 

    uint32_t i = 1;
    int32x4_t v_shift = vdupq_n_s32(shift);
    int32x4_t v_two = vdupq_n_s32(2);

    for (; i + 16 <= len - 2; i += 16) {
        // Load 8 LF samples (L[1]...L[8])
        int16x8_t v_lf_16 = vld1q_s16(lf_ptr);
        int32x4_t v_lf_lo = vshlq_s32(vmovl_s16(vget_low_s16(v_lf_16)), v_shift);
        int32x4_t v_lf_hi = vshlq_s32(vmovl_s16(vget_high_s16(v_lf_16)), v_shift);

        // Load 9 HF samples (H[0]...H[8])
        int16x8_t v_hf_curr_16 = vld1q_s16(hf_ptr); // H[0]..H[7]
        int16_t h_next = hf_ptr[8]; // H[8]
        
        int32x4_t v_hf_curr_lo = vshlq_s32(vmovl_s16(vget_low_s16(v_hf_curr_16)), v_shift);
        int32x4_t v_hf_curr_hi = vshlq_s32(vmovl_s16(vget_high_s16(v_hf_curr_16)), v_shift);

        // Construct H_next vector
        int16x8_t v_hf_next_16 = vextq_s16(v_hf_curr_16, v_hf_curr_16, 1); // H[1]..H[7], H[0]
        v_hf_next_16 = vsetq_lane_s16(h_next, v_hf_next_16, 7);

        int32x4_t v_hf_next_lo = vshlq_s32(vmovl_s16(vget_low_s16(v_hf_next_16)), v_shift);
        int32x4_t v_hf_next_hi = vshlq_s32(vmovl_s16(vget_high_s16(v_hf_next_16)), v_shift);

        // Calculate E[1..4] (lo) and E[5..8] (hi)
        int32x4_t v_sum_lo = vaddq_s32(v_hf_curr_lo, v_hf_next_lo);
        v_sum_lo = vaddq_s32(v_sum_lo, v_two);
        v_sum_lo = vshrq_n_s32(v_sum_lo, 2);
        int32x4_t v_E_lo = vsubq_s32(v_lf_lo, v_sum_lo);

        int32x4_t v_sum_hi = vaddq_s32(v_hf_curr_hi, v_hf_next_hi);
        v_sum_hi = vaddq_s32(v_sum_hi, v_two);
        v_sum_hi = vshrq_n_s32(v_sum_hi, 2);
        int32x4_t v_E_hi = vsubq_s32(v_lf_hi, v_sum_hi);

        // Calculate O[0..3] (lo) and O[4..7] (hi)
        int32x4_t v_prev_scalar = vdupq_n_s32(E_prev);
        int32x4_t v_E_prev_lo = vextq_s32(v_prev_scalar, v_E_lo, 3); // [E_prev, E1, E2, E3]

        int32x4_t v_O_sum_lo = vaddq_s32(v_E_prev_lo, v_E_lo);
        v_O_sum_lo = vshrq_n_s32(v_O_sum_lo, 1);
        int32x4_t v_O_lo = vaddq_s32(v_hf_curr_lo, v_O_sum_lo);

        int32x4_t v_E_prev_hi = vextq_s32(v_E_lo, v_E_hi, 3); // [E4, E5, E6, E7]

        int32x4_t v_O_sum_hi = vaddq_s32(v_E_prev_hi, v_E_hi);
        v_O_sum_hi = vshrq_n_s32(v_O_sum_hi, 1);
        int32x4_t v_O_hi = vaddq_s32(v_hf_curr_hi, v_O_sum_hi);

        // Store results
        int32x4x2_t v_res_lo = vzipq_s32(v_O_lo, v_E_lo);
        vst1q_s32(out_ptr + 1, v_res_lo.val[0]); 
        vst1q_s32(out_ptr + 5, v_res_lo.val[1]); 

        int32x4x2_t v_res_hi = vzipq_s32(v_O_hi, v_E_hi);
        vst1q_s32(out_ptr + 9, v_res_hi.val[0]);
        vst1q_s32(out_ptr + 13, v_res_hi.val[1]);

        // Update pointers and state
        lf_ptr += 8;
        hf_ptr += 8;
        out_ptr += 16;
        E_prev = vgetq_lane_s32(v_E_hi, 3); // E8
    }

    // Remaining loop
    for (; i < len - 2; i += 2) {
        out_ptr[0] = E_prev;
        
        int32_t L_next = (int32_t)lf_ptr[0] << shift; 
        int32_t H_curr = (int32_t)hf_ptr[0] << shift;
        int32_t H_next = (int32_t)hf_ptr[1] << shift;
        
        int32_t E_next = L_next - ((H_curr + H_next + 2) >> 2);
        int32_t O_curr = H_curr + ((E_prev + E_next) >> 1);
        
        out_ptr[2] = E_next;
        out_ptr[1] = O_curr;
        
        lf_ptr++;
        hf_ptr++;
        out_ptr += 2;
        E_prev = E_next;
    }
    
    // Final boundary
    out_ptr[0] = E_prev; 
    if (len & 1) {
        int32_t L_next = (int32_t)lf_ptr[0] << shift;
        int32_t H_curr = (int32_t)hf_ptr[0] << shift;
        
        int32_t E_next = L_next - ((H_curr + 1) >> 1);
        int32_t O_curr = H_curr + ((E_prev + E_next) >> 1);
        
        out_ptr[2] = E_next;
        out_ptr[1] = O_curr;
    }
    else { //!(len & 1)
        int32_t H_curr = (int32_t)hf_ptr[0] << shift;
        out_ptr[1] = H_curr + E_prev;
    }
}

static void idwt_horizontal_line_lf32_hf16_c_fallback(const int32_t* in_lf, const int16_t* in_hf, int32_t* out, uint32_t len, uint8_t shift) {
    assert((len >= 2) && "[idwt_c()] ERROR: Length is too small!");

    out[0] = in_lf[0] - ((((int32_t)in_hf[0] << shift) + 1) >> 1);

    for (uint32_t i = 1; i < len - 2; i += 2) {
        out[2] = in_lf[1] - ((((int32_t)in_hf[0] << shift) + ((int32_t)in_hf[1] << shift) + 2) >> 2);
        out[1] = ((int32_t)in_hf[0] << shift) + ((out[0] + out[2]) >> 1);
        in_lf++;
        in_hf++;
        out += 2;
    }
    if (len & 1) {
        out[2] = in_lf[1] - ((((int32_t)in_hf[0] << shift) + 1) >> 1);
        out[1] = ((int32_t)in_hf[0] << shift) + ((out[0] + out[2]) >> 1);
    }
    else { //!(len & 1)
        out[1] = ((int32_t)in_hf[0] << shift) + out[0];
    }
}

void idwt_horizontal_line_lf32_hf16_neon(const int32_t* in_lf, const int16_t* in_hf, int32_t* out, uint32_t len, uint8_t shift) {
    if (len < 16) {
        idwt_horizontal_line_lf32_hf16_c_fallback(in_lf, in_hf, out, len, shift);
        return;
    }

    // Initial boundary
    int32_t E_prev = in_lf[0] - ((((int32_t)in_hf[0] << shift) + 1) >> 1);
    out[0] = E_prev;

    // Pointers
    const int32_t* lf_ptr = in_lf + 1;
    const int16_t* hf_ptr = in_hf;
    int32_t* out_ptr = out; 

    uint32_t i = 1;
    int32x4_t v_shift = vdupq_n_s32(shift);
    int32x4_t v_two = vdupq_n_s32(2);

    for (; i + 16 <= len - 2; i += 16) {
        // Load 8 LF samples (L[1]...L[8])
        int32x4_t v_lf_lo = vld1q_s32(lf_ptr);
        int32x4_t v_lf_hi = vld1q_s32(lf_ptr + 4);

        // Load 9 HF samples (H[0]...H[8])
        int16x8_t v_hf_curr_16 = vld1q_s16(hf_ptr); // H[0]..H[7]
        int16_t h_next = hf_ptr[8]; // H[8]
        
        int32x4_t v_hf_curr_lo = vshlq_s32(vmovl_s16(vget_low_s16(v_hf_curr_16)), v_shift);
        int32x4_t v_hf_curr_hi = vshlq_s32(vmovl_s16(vget_high_s16(v_hf_curr_16)), v_shift);

        // Construct H_next vector
        int16x8_t v_hf_next_16 = vextq_s16(v_hf_curr_16, v_hf_curr_16, 1); // H[1]..H[7], H[0]
        v_hf_next_16 = vsetq_lane_s16(h_next, v_hf_next_16, 7);

        int32x4_t v_hf_next_lo = vshlq_s32(vmovl_s16(vget_low_s16(v_hf_next_16)), v_shift);
        int32x4_t v_hf_next_hi = vshlq_s32(vmovl_s16(vget_high_s16(v_hf_next_16)), v_shift);

        // Calculate E[1..4] (lo) and E[5..8] (hi)
        int32x4_t v_sum_lo = vaddq_s32(v_hf_curr_lo, v_hf_next_lo);
        v_sum_lo = vaddq_s32(v_sum_lo, v_two);
        v_sum_lo = vshrq_n_s32(v_sum_lo, 2);
        int32x4_t v_E_lo = vsubq_s32(v_lf_lo, v_sum_lo);

        int32x4_t v_sum_hi = vaddq_s32(v_hf_curr_hi, v_hf_next_hi);
        v_sum_hi = vaddq_s32(v_sum_hi, v_two);
        v_sum_hi = vshrq_n_s32(v_sum_hi, 2);
        int32x4_t v_E_hi = vsubq_s32(v_lf_hi, v_sum_hi);

        // Calculate O[0..3] (lo) and O[4..7] (hi)
        int32x4_t v_prev_scalar = vdupq_n_s32(E_prev);
        int32x4_t v_E_prev_lo = vextq_s32(v_prev_scalar, v_E_lo, 3); // [E_prev, E1, E2, E3]

        int32x4_t v_O_sum_lo = vaddq_s32(v_E_prev_lo, v_E_lo);
        v_O_sum_lo = vshrq_n_s32(v_O_sum_lo, 1);
        int32x4_t v_O_lo = vaddq_s32(v_hf_curr_lo, v_O_sum_lo);

        int32x4_t v_E_prev_hi = vextq_s32(v_E_lo, v_E_hi, 3); // [E4, E5, E6, E7]

        int32x4_t v_O_sum_hi = vaddq_s32(v_E_prev_hi, v_E_hi);
        v_O_sum_hi = vshrq_n_s32(v_O_sum_hi, 1);
        int32x4_t v_O_hi = vaddq_s32(v_hf_curr_hi, v_O_sum_hi);

        // Store results
        int32x4x2_t v_res_lo = vzipq_s32(v_O_lo, v_E_lo);
        vst1q_s32(out_ptr + 1, v_res_lo.val[0]); 
        vst1q_s32(out_ptr + 5, v_res_lo.val[1]); 

        int32x4x2_t v_res_hi = vzipq_s32(v_O_hi, v_E_hi);
        vst1q_s32(out_ptr + 9, v_res_hi.val[0]);
        vst1q_s32(out_ptr + 13, v_res_hi.val[1]);

        // Update pointers
        lf_ptr += 8;
        hf_ptr += 8;
        out_ptr += 16;
        
        // Update E_prev for next iteration
        E_prev = vgetq_lane_s32(v_E_hi, 3);
    }

    while (i < len - 2) {
        int32_t val_lf = lf_ptr[0];
        int32_t val_hf0 = (int32_t)hf_ptr[0] << shift;
        int32_t val_hf1 = (int32_t)hf_ptr[1] << shift;
        
        out_ptr[2] = val_lf - ((val_hf0 + val_hf1 + 2) >> 2);
        out_ptr[1] = val_hf0 + ((out_ptr[0] + out_ptr[2]) >> 1);
        
        lf_ptr++;
        hf_ptr++;
        out_ptr += 2;
        i += 2;
    }
    
    if (len & 1) {
        int32_t val_lf = lf_ptr[0];
        int32_t val_hf0 = (int32_t)hf_ptr[0] << shift;
        
        out_ptr[2] = val_lf - ((val_hf0 + 1) >> 1);
        out_ptr[1] = val_hf0 + ((out_ptr[0] + out_ptr[2]) >> 1);
    }
    else {
        int32_t val_hf0 = (int32_t)hf_ptr[0] << shift;
        out_ptr[1] = val_hf0 + out_ptr[0];
    }
}

static void idwt_vertical_line_c_fallback(const int32_t* in_lf, const int32_t* in_hf0, const int32_t* in_hf1, int32_t* out[4], uint32_t len,
                          int32_t first_precinct, int32_t last_precinct, int32_t height) {
    assert((len >= 2) && "[idwt_c()] ERROR: Length is too small!");
    //Corner case: height is equal to 2
    if (height == 2) {
        int32_t* out_2 = out[2];
        int32_t* out_3 = out[3];
        for (uint32_t i = 0; i < len; i++) {
            out_2[0] = in_lf[0] - ((in_hf1[0] + 1) >> 1);
            out_3[0] = in_hf1[0] + (out_2[0]);
            out_2++;
            out_3++;
            in_lf++;
            in_hf1++;
        }
        return;
    }
    //Corner case: first precinct in component
    if (first_precinct) {
        int32_t* out_2 = out[2];
        for (uint32_t i = 0; i < len; i++) {
            out_2[0] = in_lf[0] - ((in_hf1[0] + 1) >> 1);
            out_2++;
            in_lf++;
            in_hf1++;
        }
        return;
    }
    //Corner case: last precinct in component, height odd
    if (last_precinct && (height & 1)) {
        int32_t* out_0 = out[0];
        int32_t* out_1 = out[1];
        int32_t* out_2 = out[2];
        for (uint32_t i = 0; i < len; i++) {
            out_2[0] = in_lf[0] - ((in_hf0[0] + 1) >> 1);
            out_1[0] = in_hf0[0] + ((out_0[0] + out_2[0]) >> 1);
            out_0++;
            out_1++;
            out_2++;
            in_lf++;
            in_hf0++;
        }
        return;
    }
    //Corner case: last precinct in component, height even
    if (last_precinct && (!(height & 1))) {
        int32_t* out_0 = out[0];
        int32_t* out_1 = out[1];
        int32_t* out_2 = out[2];
        int32_t* out_3 = out[3];
        for (uint32_t i = 0; i < len; i++) {
            out_2[0] = in_lf[0] - ((in_hf0[0] + in_hf1[0] + 2) >> 2);
            out_1[0] = in_hf0[0] + ((out_0[0] + out_2[0]) >> 1);
            out_3[0] = in_hf1[0] + (out_2[0]);
            out_0++;
            out_1++;
            out_2++;
            out_3++;
            in_lf++;
            in_hf0++;
            in_hf1++;
        }
        return;
    }
    int32_t* out_0 = out[0];
    int32_t* out_1 = out[1];
    int32_t* out_2 = out[2];
    for (uint32_t i = 0; i < len; i++) {
        out_2[0] = in_lf[0] - ((in_hf0[0] + in_hf1[0] + 2) >> 2);
        out_1[0] = in_hf0[0] + ((out_0[0] + out_2[0]) >> 1);
        out_0++;
        out_1++;
        out_2++;
        in_lf++;
        in_hf0++;
        in_hf1++;
    }
}

void idwt_vertical_line_neon(const int32_t* in_lf, const int32_t* in_hf0, const int32_t* in_hf1, int32_t* out[4], uint32_t len,
                          int32_t first_precinct, int32_t last_precinct, int32_t height) {
    if (first_precinct || last_precinct || height == 2 || len < 16) {
        idwt_vertical_line_c_fallback(in_lf, in_hf0, in_hf1, out, len, first_precinct, last_precinct, height);
        return;
    }

    int32_t* out_0 = out[0];
    int32_t* out_1 = out[1];
    int32_t* out_2 = out[2];
    
    uint32_t i = 0;
    int32x4_t v_two = vdupq_n_s32(2);
    
    for (; i + 4 <= len; i += 4) {
        int32x4_t v_lf = vld1q_s32(in_lf + i);
        int32x4_t v_hf0 = vld1q_s32(in_hf0 + i);
        int32x4_t v_hf1 = vld1q_s32(in_hf1 + i);
        int32x4_t v_out0 = vld1q_s32(out_0 + i);
        
        // out_2[0] = in_lf[0] - ((in_hf0[0] + in_hf1[0] + 2) >> 2);
        int32x4_t v_sum_hf = vaddq_s32(v_hf0, v_hf1);
        v_sum_hf = vaddq_s32(v_sum_hf, v_two);
        v_sum_hf = vshrq_n_s32(v_sum_hf, 2);
        int32x4_t v_out2 = vsubq_s32(v_lf, v_sum_hf);
        
        // out_1[0] = in_hf0[0] + ((out_0[0] + out_2[0]) >> 1);
        int32x4_t v_sum_out = vaddq_s32(v_out0, v_out2);
        v_sum_out = vshrq_n_s32(v_sum_out, 1);
        int32x4_t v_out1 = vaddq_s32(v_hf0, v_sum_out);
        
        vst1q_s32(out_2 + i, v_out2);
        vst1q_s32(out_1 + i, v_out1);
    }
    
    // Tail
    for (; i < len; i++) {
        out_2[i] = in_lf[i] - ((in_hf0[i] + in_hf1[i] + 2) >> 2);
        out_1[i] = in_hf0[i] + ((out_0[i] + out_2[i]) >> 1);
    }
}

static void idwt_vertical_line_recalc_c_fallback(const int32_t* in_lf, const int32_t* in_hf0, const int32_t* in_hf1, int32_t* out[4],
                                 uint32_t len, uint32_t precinct_line_idx) {
    assert((len >= 2) && "[idwt_c()] ERROR: Length is too small!");

    int32_t* out_0 = out[0];
    if (precinct_line_idx > 1) {
        for (uint32_t i = 0; i < len; i++) {
            out_0[0] = in_lf[0] - ((in_hf0[0] + in_hf1[0] + 2) >> 2);
            out_0++;
            in_lf++;
            in_hf0++;
            in_hf1++;
        }
    }
    else {
        for (uint32_t i = 0; i < len; i++) {
            out_0[0] = in_lf[0] - ((in_hf1[0] + 1) >> 1);
            out_0++;
            in_lf++;
            in_hf1++;
        }
    }
}

void idwt_vertical_line_recalc_neon(const int32_t* in_lf, const int32_t* in_hf0, const int32_t* in_hf1, int32_t* out[4],
                                 uint32_t len, uint32_t precinct_line_idx) {
    if (len < 16) {
        idwt_vertical_line_recalc_c_fallback(in_lf, in_hf0, in_hf1, out, len, precinct_line_idx);
        return;
    }

    int32_t* out_0 = out[0];
    uint32_t i = 0;
    
    if (precinct_line_idx > 1) {
        int32x4_t v_two = vdupq_n_s32(2);
        for (; i + 4 <= len; i += 4) {
            int32x4_t v_lf = vld1q_s32(in_lf + i);
            int32x4_t v_hf0 = vld1q_s32(in_hf0 + i);
            int32x4_t v_hf1 = vld1q_s32(in_hf1 + i);
            
            int32x4_t v_sum = vaddq_s32(v_hf0, v_hf1);
            v_sum = vaddq_s32(v_sum, v_two);
            v_sum = vshrq_n_s32(v_sum, 2);
            int32x4_t v_res = vsubq_s32(v_lf, v_sum);
            
            vst1q_s32(out_0 + i, v_res);
        }
        for (; i < len; i++) {
            out_0[i] = in_lf[i] - ((in_hf0[i] + in_hf1[i] + 2) >> 2);
        }
    } else {
        for (; i + 4 <= len; i += 4) {
            int32x4_t v_lf = vld1q_s32(in_lf + i);
            int32x4_t v_hf1 = vld1q_s32(in_hf1 + i);
            
            int32x4_t v_sum = vaddq_s32(v_hf1, vdupq_n_s32(1));
            v_sum = vshrq_n_s32(v_sum, 1);
            int32x4_t v_res = vsubq_s32(v_lf, v_sum);
            
            vst1q_s32(out_0 + i, v_res);
        }
        for (; i < len; i++) {
            out_0[i] = in_lf[i] - ((in_hf1[i] + 1) >> 1);
        }
    }
}
