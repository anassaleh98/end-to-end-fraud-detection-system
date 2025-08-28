from train_models import train_eval_rf_oversampling, train_eval_xgb, train_eval_cat
from utils_data import read_data , New_features
from fraud_detection_stacking import stacking_ensemble
from pipeline import train_and_save_pipeline, load_pipeline, predict_pipeline
from utils_data import evaluation_metrics
import pandas as pd

def main():
    save_path = r"E:\My_Github\Credit Card Fraud Detection\models\stacking_pipeline.pkl"

    model_pip = load_pipeline(save_path)
    
    X_test = [172792, 2.451888, 22.057729, 4.226108, 16.875344, 34.099309, 73.301626,
              120.589494, 18.282168, 10.392889, 15.331742, 11.669205, 7.848392, 7.126883,
              10.526766, 8.877742, 17.315112, 9.207059, 5.041069, 5.572113, 39.420904, 
              27.202839, 10.503090, 22.528412, 4.584549, 6.070850, 3.517346, 31.612198,
              16.129609, 25691.16]
    
    # Column names
    columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
               'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
               'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    # Convert to DataFrame (single row)
    df = pd.DataFrame([X_test], columns=columns)
    
    print("/n test one row: ")
    # Run prediction
    test_preds = predict_pipeline(model_pip, df)
    print("Prediction:", test_preds[0])


if __name__ == "__main__":
    main()

