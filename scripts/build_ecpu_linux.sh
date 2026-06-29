#!/usr/bin/env sh
set -eu

BUILD_DIR="src/ecpu/build_linux"
NATIVE_ARCH="ON"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --portable|--raspi)
            NATIVE_ARCH="OFF"
            shift
            ;;
        --native)
            NATIVE_ARCH="ON"
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

cmake -S src/ecpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DECPU_NATIVE_ARCH="$NATIVE_ARCH"
cmake --build "$BUILD_DIR" -j "$JOBS"

printf '%s\n' "Built: $BUILD_DIR/libecpu.so"
printf '%s\n' "Use with: export PILM_ECPU_LIB=$BUILD_DIR/libecpu.so"
