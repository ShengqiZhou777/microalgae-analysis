from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os

class XGBoostExpert:
    def __init__(self, n_estimators=500, learning_rate=0.05, max_depth=4, tree_method="hist"):
        self.model = XGBRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            tree_method=tree_method,
            early_stopping_rounds=None # Handled manually if needed, or set here
        )
        self.name = "XGBoost"

    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def save(self, path):
        # XGBoost built-in save
        self.model.save_model(path)
        
    def load(self, path):
        self.model.load_model(path)

class LightGBMExpert:
    def __init__(self, n_estimators=500, learning_rate=0.05, max_depth=4, num_leaves=31):
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            verbosity=-1
        )
        self.name = "LightGBM"

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        # Joblib for LGBM
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)
