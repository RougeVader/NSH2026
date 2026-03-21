import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Define file paths
DATA_DIR = r"C:\Users\Ayush Khandwe\Downloads\isro\Collision Avoidance Challenge - Dataset\kelvins_competition_data"
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train_data.csv")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "xgb_model.json")

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    
    # Target variable is 'risk'
    # Features are all other columns.
    
    # Handle missing values (if any)
    # For simplicity, we'll fill with the mean for numerical columns.
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
            
    # For categorical columns, fill with the mode.
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Log transform columns with very large values
    for col in ['t_position_covariance_det', 'c_position_covariance_det']:
        if col in df.columns:
            print(f"Applying log transformation to '{col}'")
            df[col] = np.log1p(df[col])
            # Replace any -inf that might result from log(0)
            df[col] = df[col].replace(-np.inf, 0)


    # Convert categorical variables to numerical using one-hot encoding
    # We need to be careful with high-cardinality features.
    # For this dataset, let's assume one-hot encoding is fine for now.
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Data preprocessing complete.")
    return df

def inspect_data(df):
    print("\nInspecting data for issues...")
    
    # Check for infinity values
    inf_check = np.isinf(df).any().any()
    if inf_check:
        print("  - Found infinity values in the data.")
        # Find columns with infinity
        inf_cols = df.columns[np.isinf(df).any()].tolist()
        print(f"    Columns with infinity: {inf_cols}")
    else:
        print("  - No infinity values found.")
        
    # Check for very large values
    max_vals = df.max()
    large_vals_threshold = 1e12 # Arbitrary threshold for "very large"
    large_val_cols = max_vals[max_vals > large_vals_threshold].index.tolist()
    
    if large_val_cols:
        print(f"  - Found columns with values larger than {large_vals_threshold}: {large_val_cols}")
        for col in large_val_cols:
            print(f"    Max value in '{col}': {max_vals[col]}")
    else:
        print("  - No excessively large values found.")
    
    print("Data inspection complete.")


def train_model(X, y, use_gpu=False):
    print("Training XGBoost model...")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    if use_gpu:
        print("Using GPU for training.")
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
    else:
        print("Using CPU for training.")
        params['tree_method'] = 'hist'

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train the model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    print("Model training complete.")
    return model

def evaluate_model(model, X, y):
    print("Evaluating model...")
    
    # Predictions
    dtest = xgb.DMatrix(X)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

def save_model(model, filepath, feature_names):
    print(f"Saving model to {filepath}...")
    model.save_model(filepath)
    # Save feature names to a text file
    feature_names_path = filepath.replace(".json", "_features.txt")
    with open(feature_names_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print("Model and feature names saved successfully.")

if __name__ == "__main__":
    # Load data
    train_df = load_data(TRAIN_DATA_FILE)
    
    # Target variable is 'risk' (log10 of risk)
    # Threshold at -6 (10^-6) for "High Risk"
    target_col = 'risk'
    y = (train_df[target_col] > -6.0).astype(int)
    
    # Drop non-feature columns
    # 'event_id' is usually present in this dataset
    cols_to_drop = [target_col]
    if 'event_id' in train_df.columns:
        cols_to_drop.append('event_id')
    
    X = train_df.drop(columns=cols_to_drop)
    
    # Preprocess features
    X_processed = preprocess_data(X)
    feature_names = X_processed.columns.tolist()
    
    # Inspect data for issues
    inspect_data(X_processed)
    
    # Train model
    model = train_model(X_processed, y, use_gpu=True)
    
    # Evaluate model
    evaluate_model(model, X_processed, y)
    
    # Save model and feature names
    save_model(model, MODEL_SAVE_PATH, feature_names)
