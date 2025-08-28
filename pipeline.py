import joblib
import numpy as np
import pandas as pd
from utils_data import New_features
from fraud_detection_stacking import stacking_ensemble

# -----------------------------
# TRAIN & SAVE
# -----------------------------
def train_and_save_pipeline(train_val, save_path=r"models/stacking_pipeline.pkl", threshold=0.6):
    # Add engineered features
    train_val = New_features(train_val)


    # Train stacking ensemble
    rf, xgb, cat, meta_model, scaler, extra_scaler = stacking_ensemble(train_val, threshold)

    # Package everything
    pipeline = {
        "rf": rf,
        "xgb": xgb,
        "cat": cat,
        "meta_model": meta_model,
        "scaler": scaler,
        "extra_scaler": extra_scaler,
        "threshold": threshold,
        "features_main": [c for c in train_val.columns if c not in ["Class"]], 
        "features_extra": ['Hours','V17','V14','V12','V11','V4']
    }

    # Save
    joblib.dump(pipeline, save_path)
    print(f"\n✅ Pipeline saved to {save_path}")
    
    return pipeline


# -----------------------------
# LOAD & PREDICT
# -----------------------------
def load_pipeline(save_path=r"models/stacking_pipeline.pkl"):
    return joblib.load(save_path)

def predict_pipeline(pipeline, df):
    # Apply feature engineering
    df = New_features(df)

    # Scale main features
    X_scaled = pipeline["scaler"].transform(df[pipeline["features_main"]])

    # Base model predictions
    rf_prob  = pipeline["rf"].predict_proba(X_scaled)[:, 1]
    xgb_prob = pipeline["xgb"].predict_proba(X_scaled)[:, 1]
    cat_prob = pipeline["cat"].predict_proba(X_scaled)[:, 1]

    # Extra features
    extra_features = pipeline["extra_scaler"].transform(df[pipeline["features_extra"]])

    # Meta features
    meta_X = np.column_stack((rf_prob, xgb_prob, cat_prob, extra_features))

    # Meta prediction
    probs = pipeline["meta_model"].predict_proba(meta_X)[:, 1]
    preds = (probs >= pipeline["threshold"]).astype(int)
    

    return preds, probs

