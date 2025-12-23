#!/bin/bash

# Targets to train
TARGETS=("Dry_Weight" "Chl_Per_Cell" "Fv_Fm" "Oxygen_Rate")

echo "=================================================="
echo "   STARTING MULTI-TARGET STATIC MODEL TRAINING    "
echo "=================================================="

for target in "${TARGETS[@]}"; do
    echo ""
    echo ">>> Training Static Model for Target: $target <<<"
    python3 main.py --target "$target" --mode cnn_only --max_folds 1
    
    echo ""
    echo ">>> Training DYNAMIC Model for Target: $target <<<"
    python3 main.py --target "$target" --mode cnn_only --stochastic --max_folds 1
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully trained $target (Static & Dynamic)"
    else
        echo "❌ Failed to train $target"
        exit 1
    fi
done

echo ""
echo "=================================================="
echo "          ALL MODELS TRAINED SUCCESSFULLY         "
echo "=================================================="
