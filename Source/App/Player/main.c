#include <stdio.h>
#include <stdlib.h>
#include <SDL.h>
#include "SvtJpegxsDec.h"

// Simple error handling macro
#define CHECK_SDL(x) if (!(x)) { fprintf(stderr, "SDL Error: %s\n", SDL_GetError()); return -1; }

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_jxs_file>\n", argv[0]);
        return -1;
    }

    const char* input_file_name = argv[1];
    FILE* input_file = fopen(input_file_name, "rb");
    if (!input_file) {
        printf("Could not open file: %s\n", input_file_name);
        return -1;
    }

    // 1. Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // 2. Initialize Decoder
    svt_jpeg_xs_decoder_api_t dec_api = {0};
    dec_api.threads_num = 4; // Adjust based on CPU
    dec_api.use_cpu_flags = CPU_FLAGS_ALL;
    
    // Read first chunk to probe frame size
    size_t probe_size = 4096;
    uint8_t* probe_buffer = malloc(probe_size);
    if (!probe_buffer) return -1;

    size_t read_bytes = fread(probe_buffer, 1, probe_size, input_file);
    fseek(input_file, 0, SEEK_SET); // Reset file

    uint32_t frame_size = 0;
    svt_jpeg_xs_image_config_t image_config;
    
    // Get frame size and image config
    // Use fast_search=1 to get size from header if possible
    SvtJxsErrorType_t err = svt_jpeg_xs_decoder_get_single_frame_size(
        probe_buffer, read_bytes, &image_config, &frame_size, 1);

    free(probe_buffer);

    if (err != SvtJxsErrorNone) {
        printf("Error parsing bitstream header: %d\n", err);
        return -1;
    }

    printf("Video: %dx%d, %d components, %d bit depth\n", image_config.width, image_config.height, image_config.components_num, image_config.bit_depth);

    // Initialize Decoder Instance
    // We need a buffer for the bitstream.
    uint8_t* bitstream_buffer = malloc(frame_size);
    if (!bitstream_buffer) return -1;

    size_t r = fread(bitstream_buffer, 1, frame_size, input_file);
    if (r != frame_size) {
        printf("Could not read first frame\n");
        return -1;
    }
    
    err = svt_jpeg_xs_decoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, 
                                   &dec_api, bitstream_buffer, frame_size, &image_config);
    if (err != SvtJxsErrorNone) {
        printf("Decoder init failed: %d\n", err);
        return -1;
    }

    // 3. Create SDL Window & Renderer
    SDL_Window* window = SDL_CreateWindow("SVT-JPEG-XS Player",
                                          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          image_config.width, image_config.height,
                                          SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    CHECK_SDL(window);

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    CHECK_SDL(renderer);

    // Create Texture
    Uint32 sdl_format = SDL_PIXELFORMAT_IYUV; // Default to Planar YUV
    if (image_config.format == COLOUR_FORMAT_PACKED_YUV444_OR_RGB) {
        sdl_format = SDL_PIXELFORMAT_RGB24;
    } else if (image_config.format == COLOUR_FORMAT_PLANAR_YUV422) {
        // SDL doesn't have a direct Planar YUV422 format usually, might need conversion or use YUY2 (packed)
        // For now, let's assume IYUV (420) or RGB. 
        // If it is 422, we might see artifacts if we treat it as 420, but let's stick to basic setup.
        printf("Warning: Format %d might need specific SDL handling. Trying IYUV.\n", image_config.format);
    }

    SDL_Texture* texture = SDL_CreateTexture(renderer, sdl_format,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             image_config.width, image_config.height);
    CHECK_SDL(texture);

    // 4. Allocate Image Buffer for Decoder Output
    svt_jpeg_xs_image_buffer_t image_buffer = {0};
    uint32_t pixel_size = image_config.bit_depth <= 8 ? 1 : 2;
    for (uint8_t i = 0; i < image_config.components_num; ++i) {
        image_buffer.stride[i] = image_config.components[i].width;
        image_buffer.alloc_size[i] = image_buffer.stride[i] * image_config.components[i].height * pixel_size;
        image_buffer.data_yuv[i] = malloc(image_buffer.alloc_size[i]);
        if (!image_buffer.data_yuv[i]) {
            printf("Allocation failed\n");
            return -1;
        }
    }

    // 5. Main Loop
    int quit = 0;
    SDL_Event e;
    
    // We already read the first frame into bitstream_buffer
    int first_frame = 1;

    // FPS Calculation
    Uint32 last_time = SDL_GetTicks();
    Uint32 frame_count = 0;
    float current_fps = 0.0f;
    int show_fps = 0;
    char title_buffer[128];

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) quit = 1;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_i) {
                    show_fps = !show_fps;
                    if (!show_fps) {
                        SDL_SetWindowTitle(window, "SVT-JPEG-XS Player");
                    }
                }
            }
        }

        // FPS Update
        frame_count++;
        Uint32 current_time = SDL_GetTicks();
        if (current_time > last_time + 1000) {
            current_fps = frame_count / ((current_time - last_time) / 1000.0f);
            last_time = current_time;
            frame_count = 0;
            
            if (show_fps) {
                snprintf(title_buffer, sizeof(title_buffer), "SVT-JPEG-XS Player - FPS: %.2f", current_fps);
                SDL_SetWindowTitle(window, title_buffer);
            }
        }

        if (!first_frame) {
            // Read next frame
            size_t r = fread(bitstream_buffer, 1, frame_size, input_file);
            if (r != frame_size) {
                // End of file, loop
                fseek(input_file, 0, SEEK_SET);
                r = fread(bitstream_buffer, 1, frame_size, input_file);
                if (r != frame_size) break; // Should not happen unless empty file
            }
        }
        first_frame = 0;

        // Decode
        svt_jpeg_xs_frame_t dec_input = {0};
        dec_input.bitstream.buffer = bitstream_buffer;
        dec_input.bitstream.used_size = frame_size;
        dec_input.image = image_buffer; // Your allocated buffer

        err = svt_jpeg_xs_decoder_send_frame(&dec_api, &dec_input, 1);
        if (err) {
            printf("Send frame error: %d\n", err);
            break;
        }

        svt_jpeg_xs_frame_t dec_output;
        err = svt_jpeg_xs_decoder_get_frame(&dec_api, &dec_output, 1);
        if (err) {
            printf("Get frame error: %d\n", err);
            break;
        }

        // Update Texture
        if (sdl_format == SDL_PIXELFORMAT_IYUV) {
            // SDL_UpdateYUVTexture expects Y, U, V planes.
            // Assuming component 0=Y, 1=U, 2=V
            SDL_UpdateYUVTexture(texture, NULL,
                                 dec_output.image.data_yuv[0], dec_output.image.stride[0],
                                 dec_output.image.data_yuv[1], dec_output.image.stride[1],
                                 dec_output.image.data_yuv[2], dec_output.image.stride[2]);
        } else if (sdl_format == SDL_PIXELFORMAT_RGB24) {
            // RGB
            SDL_UpdateTexture(texture, NULL, dec_output.image.data_yuv[0], dec_output.image.stride[0]);
        }

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        
        SDL_Delay(33); // ~30 FPS
    }

    // Cleanup
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    svt_jpeg_xs_decoder_close(&dec_api);
    free(bitstream_buffer);
    for (uint8_t i = 0; i < image_config.components_num; ++i) {
        free(image_buffer.data_yuv[i]);
    }
    fclose(input_file);
    
    return 0;
}
