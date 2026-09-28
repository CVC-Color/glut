#!/bin/bash
# Build a hald-image dataset from a directory of .cube 3D LUTs.
#
# Pipeline:
#   1. generate_hald.py --step STEP
#        -> hald_step<STEP>.png          (identity hald)
#        -> hald_step<STEP>_exclude.png  (complementary samples; only when STEP != 1)
#   2. apply_cube.py renders every .cube file in --lut_dir onto each hald image.
#        - STEP == 1: no _exclude image exists (the step=1 hald already covers
#          every RGB value), so there is nothing to hold out -> a single folder:
#            hald_step1.png -> dataset/hald_images/<lut_dir basename>_fullsize/
#        - STEP != 1: train/test split using the two hald images:
#            hald_step<STEP>.png         -> dataset/hald_images/<lut_dir basename>_train/
#            hald_step<STEP>_exclude.png -> dataset/hald_images/<lut_dir basename>_test/
#      (apply_cube.py already copies its --hald_image into --out_dir as
#      Original_Image.png; the explicit `cp` below is a harmless safety net
#      in case that ever changes.)
#   3. Delete the generated hald_step*.png file(s) from the repo root.
#
# Must be run from the repository root (output paths are relative to it).
#
# Usage:
#   ./data/prepare_hald_dataset.sh [--lut_dir DIR] [--step N]
#
# Examples:
#   ./data/prepare_hald_dataset.sh --lut_dir ./dataset/cube_files/7luts --step 2
#   # -> dataset/hald_images/7luts_train/  dataset/hald_images/7luts_test/
#
#   ./data/prepare_hald_dataset.sh --lut_dir ./dataset/cube_files/7luts --step 1
#   # -> dataset/hald_images/7luts_fullsize/

set -euo pipefail

# Resolve the other data/ scripts relative to this file, so the pipeline works
# no matter what the caller's CWD is (README examples invoke it from the repo
# root, but this makes that assumption non-load-bearing).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LUT_DIR="./dataset/cube_files/7luts"
STEP=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lut_dir)
            LUT_DIR="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,27p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ ! -d "$LUT_DIR" ]; then
    echo "ERROR: LUT directory not found: $LUT_DIR" >&2
    exit 1
fi
if ! ls "$LUT_DIR"/*.cube >/dev/null 2>&1; then
    echo "ERROR: no .cube files found in $LUT_DIR" >&2
    exit 1
fi

LUT_NAME="$(basename "$LUT_DIR")"
HALD_MAIN="hald_step${STEP}.png"
HALD_EXCLUDE="hald_step${STEP}_exclude.png"

echo "=== 1. Generating identity hald images (step=${STEP}) ==="
python "$SCRIPT_DIR/generate_hald.py" --step "${STEP}"

if [ ! -f "$HALD_MAIN" ]; then
    echo "ERROR: expected ${HALD_MAIN} after generate_hald.py, but it was not found." >&2
    exit 1
fi

if [ "$STEP" -eq 1 ]; then
    # step=1 already covers every RGB value -> no held-out complement, one folder.
    FULL_DIR="dataset/hald_images/${LUT_NAME}_fullsize"

    echo ""
    echo "=== 2. Applying LUTs from ${LUT_DIR} ==="
    echo "--- full: ${FULL_DIR}  (from ${HALD_MAIN}) ---"
    python "$SCRIPT_DIR/apply_cube.py" \
        --lut_dir "$LUT_DIR" \
        --hald_image "$HALD_MAIN" \
        --out_dir "$FULL_DIR"

    echo ""
    echo "=== 3. Copying the hald image in as Original_Image.png ==="
    cp "$HALD_MAIN" "${FULL_DIR}/Original_Image.png"

    echo ""
    echo "=== 4. Cleaning up generated hald image ==="
    rm -f "$HALD_MAIN"

    echo ""
    echo "Done."
    echo "  full: ${FULL_DIR}"
else
    if [ ! -f "$HALD_EXCLUDE" ]; then
        echo "ERROR: expected ${HALD_EXCLUDE} after generate_hald.py, but it was not found." >&2
        exit 1
    fi

    TRAIN_DIR="dataset/hald_images/${LUT_NAME}_train"
    TEST_DIR="dataset/hald_images/${LUT_NAME}_test"

    echo ""
    echo "=== 2. Applying LUTs from ${LUT_DIR} ==="
    echo "--- train: ${TRAIN_DIR}  (from ${HALD_MAIN}) ---"
    python "$SCRIPT_DIR/apply_cube.py" \
        --lut_dir "$LUT_DIR" \
        --hald_image "$HALD_MAIN" \
        --out_dir "$TRAIN_DIR"

    echo "--- test:  ${TEST_DIR}  (from ${HALD_EXCLUDE}) ---"
    python "$SCRIPT_DIR/apply_cube.py" \
        --lut_dir "$LUT_DIR" \
        --hald_image "$HALD_EXCLUDE" \
        --out_dir "$TEST_DIR"

    echo ""
    echo "=== 3. Copying the hald images in as Original_Image.png ==="
    cp "$HALD_MAIN" "${TRAIN_DIR}/Original_Image.png"
    cp "$HALD_EXCLUDE" "${TEST_DIR}/Original_Image.png"

    echo ""
    echo "=== 4. Cleaning up generated hald images ==="
    rm -f "$HALD_MAIN" "$HALD_EXCLUDE"

    echo ""
    echo "Done."
    echo "  train: ${TRAIN_DIR}"
    echo "  test:  ${TEST_DIR}"
fi
