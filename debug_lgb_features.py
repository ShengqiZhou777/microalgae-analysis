
import joblib
import os
import sys

path = "archive_run_20251225_153705/weights/Dry_Weight_Light_stochastic_lgb.joblib"
print(f"Loading {path}...")

try:
    model = joblib.load(path)
    print("Model Loaded Successfully.")
    print(f"Type: {type(model)}")
    
    if hasattr(model, 'n_features_in_'):
        print(f"n_features_in_: {model.n_features_in_}")
        
    if hasattr(model, 'feature_name_'):
        print(f"Feature Names (len={len(model.feature_name_)}):")
        print(model.feature_name_[:10], "...")
        
    if hasattr(model, 'booster_'):
        print(f"Booster Num Features: {model.booster_.num_feature()}")
        print(f"Booster Feature Names: {model.booster_.feature_name()[:10]} ...")

except Exception as e:
    print(f"Error loading model: {e}")
