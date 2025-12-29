/*
* Copyright(c) 2024 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "Dwt_neon.h"
#include <arm_neon.h>
#include <assert.h>

void dwt_horizontal_line_neon(int32_t* out_lf, int32_t* out_hf, const int32_t* in, uint32_t len) {
    assert((len >= 2) && "[dwt_horizontal_line_neon()] ERROR: Length is too small!");

    if (len == 2) {
        out_hf[0] = in[1] - in[0];
        out_lf[0] = in[0] + ((out_hf[0] + 1) >> 1);
        return;
    }

    // First element (scalar)
    out_hf[0] = in[1] - ((in[0] + in[2]) >> 1);
    out_lf[0] = in[0] + ((out_hf[0] + 1) >> 1);

    const uint32_t count = ((len - 1) / 2);
    uint32_t id = 1;

    // Vector loop
    // We process 4 elements at a time.
    // We need at least 4 elements remaining.
    if (count >= 5) { // 1 (already done) + 4 = 5
        int32x4_t v_two = vdupq_n_s32(2);
        
        for (; id + 4 <= count; id += 4) {
            // Load 8 input samples (4 even, 4 odd) starting from in[id*2]
            // in[id*2] ... in[id*2 + 7]
            // We also need in[id*2 + 8] (next even) for the HF calculation of the last element in the vector.
            
            // Using vld2q to deinterleave
            int32x4x2_t v_in = vld2q_s32(&in[id * 2]);
            int32x4_t v_even = v_in.val[0]; // in[2*id], in[2*id+2], ...
            int32x4_t v_odd = v_in.val[1];  // in[2*id+1], in[2*id+3], ...

            // We need the next even element for the HF calculation: (even + next_even) >> 1
            // The next evens are shifted version of v_even combined with the first even of the next chunk.
            // Actually, simpler: load the next even element.
            int32_t next_even_scalar = in[(id + 4) * 2];
            
            // Create v_even_next: [in[2*id+2], in[2*id+4], in[2*id+6], in[2*id+8]]
            // We can construct this by extracting from v_even and appending next_even_scalar.
            int32x4_t v_even_next = vextq_s32(v_even, v_even, 1);
            v_even_next = vsetq_lane_s32(next_even_scalar, v_even_next, 3);

            // Calculate HF
            // out_hf[id] = in[id * 2 + 1] - ((in[id * 2] + in[id * 2 + 2]) >> 1);
            // v_hf = v_odd - ((v_even + v_even_next) >> 1)
            int32x4_t v_sum_even = vaddq_s32(v_even, v_even_next);
            int32x4_t v_avg_even = vshrq_n_s32(v_sum_even, 1);
            int32x4_t v_hf = vsubq_s32(v_odd, v_avg_even);

            // Store HF
            vst1q_s32(&out_hf[id], v_hf);

            // Calculate LF
            // out_lf[id] = in[id * 2] + ((out_hf[id - 1] + out_hf[id] + 2) >> 2);
            // We need out_hf[id-1].
            // For the first element in vector, it's out_hf[id-1] (scalar or from prev vector).
            // For others, it's the previous element in v_hf.
            
            int32_t prev_hf_scalar = out_hf[id - 1];
            int32x4_t v_hf_prev = vextq_s32(v_hf, v_hf, 3); // Rotate right by 1 (element 3 moves to 0)
            // But we need to insert the scalar at lane 0 and shift others up?
            // No, vextq_s32(a, b, n) extracts from concatenated b:a.
            // We want [prev, hf0, hf1, hf2].
            // If we have v_hf = [hf0, hf1, hf2, hf3].
            // We want to shift right and insert prev.
            // We can use a temporary vector for prev.
            // Or just construct it.
            
            // Let's use vextq with a vector containing the previous scalar.
            int32x4_t v_prev_vec = vdupq_n_s32(prev_hf_scalar);
            // We want [prev, hf0, hf1, hf2].
            // vextq_s32(low, high, n) -> extracts starting at n.
            // If we do vextq_s32(v_prev_vec, v_hf, 3)?
            // [prev, prev, prev, prev] [hf0, hf1, hf2, hf3]
            // index 0 1 2 3 4 5 6 7
            // start at 3: prev, hf0, hf1, hf2. Yes!
            // Wait, vextq_s32(a, b, n) extracts from b:a (b is high, a is low).
            // Elements are [a0, a1, a2, a3, b0, b1, b2, b3].
            // We want [prev, hf0, hf1, hf2].
            // If a = [prev, x, x, x], b = [hf0, hf1, hf2, hf3].
            // We want elements at indices 3, 4, 5, 6? No.
            // We want [prev, hf0, hf1, hf2].
            // That corresponds to taking the last element of 'a' and first 3 of 'b'.
            // So if a = [x, x, x, prev], b = [hf0, hf1, hf2, hf3].
            // Then vextq_s32(a, b, 3) gives [prev, hf0, hf1, hf2].
            
            v_prev_vec = vsetq_lane_s32(prev_hf_scalar, v_prev_vec, 3);
            v_hf_prev = vextq_s32(v_prev_vec, v_hf, 3);

            // v_lf = v_even + ((v_hf_prev + v_hf + 2) >> 2)
            int32x4_t v_sum_hf = vaddq_s32(v_hf_prev, v_hf);
            v_sum_hf = vaddq_s32(v_sum_hf, v_two);
            int32x4_t v_term = vshrq_n_s32(v_sum_hf, 2);
            int32x4_t v_lf = vaddq_s32(v_even, v_term);

            // Store LF
            vst1q_s32(&out_lf[id], v_lf);
        }
    }

    // Scalar cleanup loop
    for (; id < count; id++) {
        out_hf[id] = in[id * 2 + 1] - ((in[id * 2] + in[id * 2 + 2]) >> 1);
        out_lf[id] = in[id * 2] + ((out_hf[id - 1] + out_hf[id] + 2) >> 2);
    }

    // Final boundary handling
    if (!(len & 1)) {
        out_hf[len / 2 - 1] = in[len - 1] - in[len - 2];
        out_lf[len / 2 - 1] = in[len - 2] + ((out_hf[len / 2 - 2] + out_hf[len / 2 - 1] + 2) >> 2);
    }
    else { //if (len & 1){
        out_lf[len / 2] = in[len - 1] + ((out_hf[len / 2 - 1] + 1) >> 1);
    }
}
