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
    # 1. Static Model (No --stochastic)
    echo " -> Training Static Model for $target ($condition)..."
    python3 main.py --target "$target" --mode boost_only --condition "$condition"
    
    # 2. Dynamic Model (With --stochastic)
    echo " -> Training Dynamic Model for $target ($condition)..."
    python3 main.py --target "$target" --mode boost_only --stochastic --condition "$condition"
        
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

# 3. Prediction Step
echo ""
echo "=================================================="
echo "           STARTING BATCH PREDICTION              "
echo "=================================================="
python3 scripts/predict.py --input data/dataset_test.csv --output results/test_predictions.csv

if [ $? -eq 0 ]; then
    echo "✅ Prediction completed successfully."
else
    echo "❌ Prediction failed."
    exit 1
fi

# 4. Archiving Step
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_DIR="archive_run_${TIMESTAMP}"
echo ""
echo "=================================================="
echo "           ARCHIVING ARTIFACTS                    "
echo "           -> ${ARCHIVE_DIR}                      "
echo "=================================================="

mkdir -p "${ARCHIVE_DIR}/weights"
mkdir -p "${ARCHIVE_DIR}/results"
mkdir -p "${ARCHIVE_DIR}/plots"
mkdir -p "${ARCHIVE_DIR}/logs"

# Move specific files
mv weights/*.json "${ARCHIVE_DIR}/weights/" 2>/dev/null
mv weights/*.pth "${ARCHIVE_DIR}/weights/" 2>/dev/null
mv weights/*.joblib "${ARCHIVE_DIR}/weights/" 2>/dev/null

# Move Results
mv results/test_predictions*.csv "${ARCHIVE_DIR}/results/" 2>/dev/null
# Move Plots
mv MoE_Result_*.png "${ARCHIVE_DIR}/plots/" 2>/dev/null

# Move Logs
mv logs/*.log "${ARCHIVE_DIR}/logs/"

echo "✅ Artifacts archived to ${ARCHIVE_DIR}"
