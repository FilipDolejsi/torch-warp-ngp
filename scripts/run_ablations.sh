#!/bin/bash

set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ITERATIONS=100000
DATA_ROOT=""
SCRIPT="train.py" 

RUN_ID=1

# ---------------------------------------------------------
# 1. BASELINE (Optimal from paper: T=2^19, L=16, F=2)
# ---------------------------------------------------------
PREFIX=$(printf "%02d" $RUN_ID)
EXP_NAME="${PREFIX}_pytorch_baseline"
echo -e "\n>>> RUNNING: $EXP_NAME (T=2^19, L=16, F=2)..."
mkdir -p "runs/$EXP_NAME"
python -u $SCRIPT \
    --exp_name "$EXP_NAME" \
    --data_root $DATA_ROOT \
    --iterations $ITERATIONS 2>&1 | tee "runs/$EXP_NAME/console.log"
RUN_ID=$((RUN_ID + 1))

PREFIX=$(printf "%02d" $RUN_ID)
EXP_NAME="${PREFIX}_warp_baseline"
echo -e "\n>>> RUNNING: $EXP_NAME (T=2^19, L=16, F=2)..."
mkdir -p "runs/$EXP_NAME"
python -u $SCRIPT \
    --exp_name "$EXP_NAME" \
    --use_warp \
    --data_root $DATA_ROOT \
    --iterations $ITERATIONS 2>&1 | tee "runs/$EXP_NAME/console.log"
RUN_ID=$((RUN_ID + 1))

# ---------------------------------------------------------
# 2. ABLATION 1: Hash Table Size (T in {2^14, 2^19, 2^21})
# 2^14 = 16384 | 2^19 = 524288 | 2^21 = 2097152
# ---------------------------------------------------------
for T_VAL in 16384 524288 2097152; do
    
    # Map the value back to its power for cleaner folder names
    if [ "$T_VAL" -eq 16384 ]; then T_POW="14"; fi
    if [ "$T_VAL" -eq 524288 ]; then T_POW="19"; fi
    if [ "$T_VAL" -eq 2097152 ]; then T_POW="21"; fi

    PREFIX=$(printf "%02d" $RUN_ID)
    EXP_NAME="${PREFIX}_pytorch_ablation_T${T_POW}"
    echo -e "\n>>> RUNNING: $EXP_NAME (Hash T=2^${T_POW})..."
    mkdir -p "runs/$EXP_NAME"
    python -u $SCRIPT \
        --exp_name "$EXP_NAME" \
        --data_root $DATA_ROOT \
        --iterations $ITERATIONS \
        --t $T_VAL 2>&1 | tee "runs/$EXP_NAME/console.log"
    RUN_ID=$((RUN_ID + 1))

    PREFIX=$(printf "%02d" $RUN_ID)
    EXP_NAME="${PREFIX}_warp_ablation_T${T_POW}"
    echo -e "\n>>> RUNNING: $EXP_NAME (Hash T=2^${T_POW})..."
    mkdir -p "runs/$EXP_NAME"
    python -u $SCRIPT \
        --exp_name "$EXP_NAME" \
        --use_warp \
        --data_root $DATA_ROOT \
        --iterations $ITERATIONS \
        --t $T_VAL 2>&1 | tee "runs/$EXP_NAME/console.log"
    RUN_ID=$((RUN_ID + 1))

done

# ---------------------------------------------------------
# 3. ABLATION 2: Hierarchy Levels & Feature Dims (Figure 5)
# "To maintain a roughly equal trainable parameter count, 
# the hash table size T is set according to F * T * L = 2^24"
# ---------------------------------------------------------
TOTAL_PARAMS=16777216 # 2^24
for F in 1 2 4 8; do
    for L in 4 8 16; do
        
        # Calculate T dynamically: T = 2^24 / (F * L)
        T=$((TOTAL_PARAMS / (F * L)))
        
        PREFIX=$(printf "%02d" $RUN_ID)
        EXP_NAME="${PREFIX}_pytorch_ablation_F${F}_L${L}"
        echo -e "\n>>> RUNNING: $EXP_NAME (F=$F, L=$L, T=$T)..."
        mkdir -p "runs/$EXP_NAME"
        python -u $SCRIPT \
            --exp_name "$EXP_NAME" \
            --data_root $DATA_ROOT \
            --iterations $ITERATIONS \
            --f $F \
            --l $L \
            --t $T 2>&1 | tee "runs/$EXP_NAME/console.log"
        RUN_ID=$((RUN_ID + 1))
        
        PREFIX=$(printf "%02d" $RUN_ID)
        EXP_NAME="${PREFIX}_warp_ablation_F${F}_L${L}"
        echo -e "\n>>> RUNNING: $EXP_NAME (F=$F, L=$L, T=$T)..."
        mkdir -p "runs/$EXP_NAME"
        python -u $SCRIPT \
            --exp_name "$EXP_NAME" \
            --use_warp \
            --data_root $DATA_ROOT \
            --iterations $ITERATIONS \
            --f $F \
            --l $L \
            --t $T 2>&1 | tee "runs/$EXP_NAME/console.log"
        RUN_ID=$((RUN_ID + 1))
        
    done
done

echo -e "\n========================================================="
echo "   All comparative experiments finished successfully!"
echo "   Results are stored in the './runs/' directory."
echo "========================================================="