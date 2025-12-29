#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <assert.h>

// Mock or include necessary headers
#include "Idwt.h"
#include "Idwt_neon.h"

#define MAX_LEN 1024
#define TEST_ITERATIONS 100

void print_array(const char* name, int32_t* arr, int len) {
    printf("%s: ", name);
    for (int i = 0; i < len; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    srand(time(NULL));

    int16_t in_lf[MAX_LEN];
    int16_t in_hf[MAX_LEN];
    int32_t out_c[MAX_LEN * 2];
    int32_t out_neon[MAX_LEN * 2];

    printf("Running IDWT NEON vs C comparison tests...\n");

    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        // Random length (must be even for this function usually, but let's test various)
        // The function signature implies len is the width of the output line?
        // Or width of input?
        // Looking at C code: loop goes to len-2.
        // It seems len is the number of output pixels?
        // Let's try various lengths.
        uint32_t len = (rand() % (MAX_LEN / 2)) * 2; 
        if (len < 16) len = 16; // Ensure enough for NEON path to trigger

        uint8_t shift = rand() % 4; // Small shift

        // Fill inputs
        for (int i = 0; i < MAX_LEN; i++) {
            in_lf[i] = (rand() % 2000) - 1000;
            in_hf[i] = (rand() % 2000) - 1000;
        }

        // Clear outputs
        memset(out_c, 0, sizeof(out_c));
        memset(out_neon, 0, sizeof(out_neon));

        // Run C
        idwt_horizontal_line_lf16_hf16_c(in_lf, in_hf, out_c, len, shift);

        // Run NEON
        idwt_horizontal_line_lf16_hf16_neon(in_lf, in_hf, out_neon, len, shift);

        // Compare
        int fail = 0;
        for (uint32_t i = 0; i < len; i++) {
            if (out_c[i] != out_neon[i]) {
                fail = 1;
                printf("Mismatch at iter %d, index %d. C: %d, NEON: %d\n", iter, i, out_c[i], out_neon[i]);
                break;
            }
        }

        if (fail) {
            printf("Inputs (first 10):\n");
            printf("LF: "); for(int k=0; k<10; k++) printf("%d ", in_lf[k]); printf("\n");
            printf("HF: "); for(int k=0; k<10; k++) printf("%d ", in_hf[k]); printf("\n");
            printf("Shift: %d, Len: %d\n", shift, len);
            printf("TEST FAILED\n");
            return 1;
        }
    }

    printf("All %d iterations PASSED.\n", TEST_ITERATIONS);
    return 0;
}
