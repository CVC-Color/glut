#!/bin/bash
# Batch GLUT-model visualization script.
# Must be run from the repository root, e.g.: ./visualization/draw_glut.sh

# model directory
MODEL_DIR="./pretrained_models/glut/7luts"

# check that the directories exist
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: directory $MODEL_DIR not found"
    exit 1
fi

# output directory (defaults to <model_dir>/visualizations)
OUTPUT_BASE_DIR="./results/vis_glut/7luts"
mkdir -p "$OUTPUT_BASE_DIR"

# GPU id (set when several GPUs are present)
GPU_ID="${3:-0}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Python script path
SCRIPT_PATH="visualization/plot_glut.py"

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
echo "Starting batch GLUT processing"
echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Using GPU: $GPU_ID"
echo "========================================="

# iterate over every .pth file
for model_file in "$MODEL_DIR"/*.pth; do
    # guard against an empty glob
    if [ ! -f "$model_file" ]; then
        echo "WARNING: no .pth files in $MODEL_DIR"
        break
    fi
    
    # base name without extension
    filename=$(basename -- "$model_file")
    filename_without_ext="${filename%.*}"
    
    # output path
    output_path="$OUTPUT_BASE_DIR/${filename_without_ext}_vis.png"
    
    # extract num_gaussians from the file name (optional)
    # expected name: GSLUT001_32_psnr58.67_dE0.2332.pth
    # the middle number is num_gaussians
    num_gaussians=$(echo "$filename_without_ext" | grep -oP '(?<=_)\d+(?=_)' | head -1)
    
    # fall back to 32 if it cannot be parsed
    if [ -z "$num_gaussians" ]; then
        num_gaussians=32
        echo "  WARNING: cannot parse num_gaussians, using 32"
    fi
    
    total=$((total + 1))
    
    echo "----------------------------------------"
    echo "Processing file [$total]: $filename"
    echo "  num_gaussians: $num_gaussians"
    echo "  Output: $output_path"
    
    # run the Python script
    python "$SCRIPT_PATH" \
        --pretrained_model "$model_file" \
        --num_gaussians "$num_gaussians" \
        --output "$output_path"
    
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