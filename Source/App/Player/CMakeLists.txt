cmake_minimum_required(VERSION 3.10)

project(SvtJpegxsPlayer C)

# Find SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Add Executable
add_executable(SvtJpegxsPlayer main.c)

# Link Libraries
# Link against the SVT-JPEG-XS Decoder Library and SDL2
target_link_libraries(SvtJpegxsPlayer PRIVATE SvtJpegxsLib ${SDL2_LIBRARIES})

# Include Directories for SVT-JPEG-XS API
target_include_directories(SvtJpegxsPlayer PRIVATE
    ../../API
    ${PROJECT_SOURCE_DIR}/Source/API
)

install(TARGETS SvtJpegxsPlayer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
