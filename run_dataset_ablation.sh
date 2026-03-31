#!/bin/bash

set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================================="
echo "   Starting Full Dataset Benchmark (Optimal Params)"
echo "   Datasets: ship, chair, drums, ficus, hotdog, materials, mic, lego"
echo "   Config: T=2^19, L=16, F=2, Iterations=100000"
echo "========================================================="

ITERATIONS=100000
SCRIPT="train.py" 
DATASETS=("ship" "chair" "drums" "ficus" "hotdog" "materials" "mic" "lego")

RUN_ID=1000

for DS in "${DATASETS[@]}"; do
    # SPECIFY THE DATA PATH FOR THIS DATASET (ADJUST AS NEEDED)
    DATA_PATH="./$DS"
    
    echo -e "\n*********************************************************"
    echo "   PROCESSING DATASET: $DS"
    echo "*********************************************************"

    # 1. PYTORCH RUN
    PREFIX=$(printf "%02d" $RUN_ID)
    EXP_NAME="${PREFIX}_${DS}_pytorch_optimal"
    
    echo -e "\n>>> RUNNING: $EXP_NAME..."
    mkdir -p "runs/$EXP_NAME"
    
    python -u $SCRIPT \
        --exp_name "$EXP_NAME" \
        --data_root "$DATA_PATH" \
        --iterations $ITERATIONS \
        --t 524288 \
        --l 16 \
        --f 2 2>&1 | tee "runs/$EXP_NAME/console.log"
    
    RUN_ID=$((RUN_ID + 1))

    # 2. WARP RUN
    PREFIX=$(printf "%02d" $RUN_ID)
    EXP_NAME="${PREFIX}_${DS}_warp_optimal"
    
    echo -e "\n>>> RUNNING: $EXP_NAME..."
    mkdir -p "runs/$EXP_NAME"
    
    python -u $SCRIPT \
        --exp_name "$EXP_NAME" \
        --use_warp \
        --data_root "$DATA_PATH" \
        --iterations $ITERATIONS \
        --t 524288 \
        --l 16 \
        --f 2 2>&1 | tee "runs/$EXP_NAME/console.log"
    
    RUN_ID=$((RUN_ID + 1))

done

echo -e "\n========================================================="
echo "   Full Benchmark Suite finished successfully!"
echo "   All datasets have been processed with both backends."
echo "========================================================="