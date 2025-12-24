#!/bin/bash

# Targets to train
TARGETS=("Dry_Weight" "Chl_Per_Cell" "Fv_Fm" "Oxygen_Rate")
CONDITIONS=("Light" "Dark")

echo "=================================================="
echo "   STARTING SEPARATE CONDITION TRAINING MODELS    "
echo "=================================================="

for condition in "${CONDITIONS[@]}"; do
    echo ""
    echo "##################################################"
    echo "       STARTING CONDITION: $condition             "
    echo "##################################################"

    for target in "${TARGETS[@]}"; do
        echo ""
    # # 1. Static Model (No --stochastic)
    # echo " -> Training Static Model for $target ($condition)..."
    # python3 main.py --target "$target" --mode full --condition "$condition"
    
    # 2. Dynamic Model (With --stochastic)
    echo " -> Training Dynamic Model for $target ($condition)..."
    python3 main.py --target "$target" --mode full --stochastic --condition "$condition"
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully trained $target ($condition)"
        else
            echo "❌ Failed to train $target ($condition)"
            exit 1
        fi
    done
done

echo ""
echo "=================================================="
echo "          ALL MODELS TRAINED SUCCESSFULLY         "
echo "=================================================="
