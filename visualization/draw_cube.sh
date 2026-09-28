#!/bin/bash
# Batch 3D LUT (.cube) visualization script.
# Must be run from the repository root, e.g.: ./visualization/draw_cube.sh

# model directory
MODEL_DIR="./dataset/cube_files/7luts"

# check that the directories exist
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: directory $MODEL_DIR not found"
    exit 1
fi

# output directory (defaults to <model_dir>/visualizations)
OUTPUT_BASE_DIR="./results/vis_cube/7luts"
mkdir -p "$OUTPUT_BASE_DIR"

# GPU id (set when several GPUs are present)
GPU_ID="${3:-0}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Python script path
SCRIPT_PATH="visualization/plot_cube.py"

# check that the Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script $SCRIPT_PATH not found"
    exit 1
fi

# counters
total=0
success=0
failed=0

echo "========================================="
echo "Starting batch .cube visualization"
echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Using GPU: $GPU_ID"
echo "========================================="

# iterate over every .pth file
for model_file in "$MODEL_DIR"/*.cube; do
    # guard against an empty glob
    if [ ! -f "$model_file" ]; then
        echo "WARNING: no .cube files in $MODEL_DIR"
        break
    fi
    
    # base name without extension
    filename=$(basename -- "$model_file")
    filename_without_ext="${filename%.*}"
    
    # output path
    output_path="$OUTPUT_BASE_DIR/${filename_without_ext}_vis.png"
    
    
    total=$((total + 1))
    
    echo "----------------------------------------"
    echo "Processing file [$total]: $filename"
    echo "  Output: $output_path"
    
    # run the Python script
    python "$SCRIPT_PATH" \
        --lut "$model_file" \
        --output "$output_path" \
        --show_points
    
    # check the result
    if [ $? -eq 0 ]; then
        echo "  OK"
        success=$((success + 1))
    else
        echo "  FAILED"
        failed=$((failed + 1))
    fi
done

echo "========================================="
echo "Batch processing complete."
echo "Total: $total files"
echo "Succeeded: $success"
echo "Failed: $failed"
echo "========================================="

# return 0 if all succeeded, otherwise the failure count
if [ $failed -eq 0 ]; then
    exit 0
else
    exit $failed
fi