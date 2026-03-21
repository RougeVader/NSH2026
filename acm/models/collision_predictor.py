import pandas as pd
import numpy as np
import xgboost as xgb
import os

class CollisionPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(__file__)
        
        self.model_path = os.path.join(model_dir, "xgb_model.json")
        self.features_path = os.path.join(model_dir, "xgb_model_features.txt")
        
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        
        with open(self.features_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
            
    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts collision risk for a given input dataframe.
        df: DataFrame containing the same features as the training set.
        Returns: array of probabilities.
        """
        # 1. Preprocess
        df_proc = df.copy()
        
        # Log transform large values
        for col in ['t_position_covariance_det', 'c_position_covariance_det']:
            if col in df_proc.columns:
                df_proc[col] = np.log1p(df_proc[col])
                df_proc[col] = df_proc[col].replace(-np.inf, 0)
        
        # One-hot encode categoricals
        categorical_cols = df_proc.select_dtypes(include=['object']).columns
        df_proc = pd.get_dummies(df_proc, columns=categorical_cols)
        
        # Ensure exact same columns as training
        # Add missing columns with 0
        for col in self.feature_names:
            if col not in df_proc.columns:
                df_proc[col] = 0
        
        # Drop extra columns
        df_proc = df_proc[self.feature_names]
        
        # 2. Predict
        dtest = xgb.DMatrix(df_proc)
        return self.model.predict(dtest)

# Global Instance
predictor = None
try:
    predictor = CollisionPredictor()
except Exception as e:
    print(f"Warning: Could not initialize CollisionPredictor: {e}")
