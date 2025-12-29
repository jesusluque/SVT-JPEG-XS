#!/bin/bash

# Configuration
WIDTH=1920
HEIGHT=1080
FRAMES=30
INPUT_YUV="smpte_bars_${WIDTH}x${HEIGHT}.yuv"
BITSTREAM="output.jxs"
DECODED_YUV="decoded_${WIDTH}x${HEIGHT}.yuv"
ENC_APP="./Bin/Release/SvtJpegxsEncApp"
DEC_APP="./Bin/Release/SvtJpegxsDecApp"

# Check if apps exist
if [ ! -f "$ENC_APP" ]; then
    echo "Error: Encoder app not found at $ENC_APP"
    exit 1
fi
if [ ! -f "$DEC_APP" ]; then
    echo "Error: Decoder app not found at $DEC_APP"
    exit 1
fi

# 1. Generate SMPTE bars using FFmpeg
echo "----------------------------------------"
echo "1. Generating SMPTE bars YUV..."
echo "----------------------------------------"
ffmpeg -y -f lavfi -i smptebars=size=${WIDTH}x${HEIGHT}:rate=30 -pix_fmt yuv420p -vframes ${FRAMES} ${INPUT_YUV}

if [ ! -f "$INPUT_YUV" ]; then
    echo "Error: Failed to generate input YUV."
    exit 1
fi

# 2. Encode
echo "----------------------------------------"
echo "2. Encoding to JPEG XS..."
echo "----------------------------------------"
# Note: Adjust parameters as needed. Defaulting to 8-bit.
$ENC_APP -i ${INPUT_YUV} -w ${WIDTH} -h ${HEIGHT} --input-depth 8 --colour-format yuv420 --bpp 3.0 -b ${BITSTREAM}

if [ ! -f "$BITSTREAM" ]; then
    echo "Error: Encoding failed."
    exit 1
fi

# 3. Decode
echo "----------------------------------------"
echo "3. Decoding back to YUV..."
echo "----------------------------------------"
$DEC_APP -i ${BITSTREAM} -o ${DECODED_YUV}

if [ ! -f "$DECODED_YUV" ]; then
    echo "Error: Decoding failed."
    exit 1
fi

# 4. Compare (PSNR)
echo "----------------------------------------"
echo "4. Verifying Quality (PSNR)..."
echo "----------------------------------------"
ffmpeg -s ${WIDTH}x${HEIGHT} -pix_fmt yuv420p -i ${INPUT_YUV} -s ${WIDTH}x${HEIGHT} -pix_fmt yuv420p -i ${DECODED_YUV} -lavfi psnr -f null -

echo "----------------------------------------"
echo "Test Pipeline Complete."
echo "Generated: $INPUT_YUV"
echo "Encoded:   $BITSTREAM"
echo "Decoded:   $DECODED_YUV"
