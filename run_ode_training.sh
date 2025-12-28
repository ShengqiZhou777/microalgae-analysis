#!/bin/bash

# Targets to train
TARGETS=("Dry_Weight" "Chl_Per_Cell" "Fv_Fm" "Oxygen_Rate")
CONDITIONS=("Light" "Dark")

echo "=================================================="
echo "   STARTING NEURAL ODE TRAINING WORKFLOW          "
echo "=================================================="

for condition in "${CONDITIONS[@]}"; do
    echo ""
    echo "##################################################"
    echo "       STARTING CONDITION: $condition             "
    echo "##################################################"

    for target in "${TARGETS[@]}"; do
        echo ""
        # 1. Neural ODE Model
        echo " -> Training ODE Model for $target ($condition)..."
        # ODE trains on standard sequences (no stochastic history needed)
        python3 main.py --target "$target" --mode ode --condition "$condition"

        if [ $? -eq 0 ]; then
            echo "✅ Successfully trained ODE for $target ($condition)"
            
            # 2. Forecasting Test
            echo " -> Testing Forecasting Ability (Cutoff 6.0h)..."
            python3 scripts/test_ode_forecast.py --target "$target" --condition "$condition" --cutoff 6.0
            
        else
            echo "❌ Failed to train ODE for $target ($condition)"
            exit 1
        fi
    done
done

echo ""
echo "=================================================="
echo "          ALL ODE MODELS TRAINED SUCCESSFULLY     "
echo "=================================================="

# Note: Prediction script (predict.py) requires MoE models (XGB/LGB), so it's not run here.
# ODE performance is visualized directly by the training pipeline.

# 2. Archiving Step
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_DIR="archive_ode_${TIMESTAMP}"
echo ""
echo "=================================================="
echo "           ARCHIVING ARTIFACTS                    "
echo "           -> ${ARCHIVE_DIR}                      "
echo "=================================================="

mkdir -p "${ARCHIVE_DIR}/plots"
mkdir -p "${ARCHIVE_DIR}/logs"
mkdir -p "${ARCHIVE_DIR}/weights"

# Move Weights (ODE models and scalers)
echo " -> Moving ODE weights..."
mv weights/ode_* "${ARCHIVE_DIR}/weights/" 2>/dev/null

# Move Plots (ODE generates them in results/ode_plots/)
if [ -d "results/ode_plots" ]; then
    echo " -> Moving ODE plots..."
    mv results/ode_plots/*.png "${ARCHIVE_DIR}/plots/" 2>/dev/null
    # Clean up empty dir if valid
    rmdir results/ode_plots 2>/dev/null
fi

# Move Logs
mv logs/*.log "${ARCHIVE_DIR}/logs/"

echo "✅ ODE Artifacts archived to ${ARCHIVE_DIR}"
