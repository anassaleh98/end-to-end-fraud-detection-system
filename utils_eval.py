from train_models import train_eval_rf_oversampling, train_eval_xgb, train_eval_cat
from utils_data import read_data , New_features
from fraud_detection_stacking import stacking_ensemble
from pipeline import train_and_save_pipeline, load_pipeline, predict_pipeline
from utils_data import evaluation_metrics

def main():
    # =========================
    # 1. Load raw data
    # =========================
    train, val, train_val, test = read_data()

    # =========================
    # 2. Feature engineering
    # =========================
    train = New_features(train)
    val = New_features(val)
    train_val = New_features(train_val)

    # =========================
    # 3. Baseline model evaluations (on train/val)
    # =========================
    rf_model  = train_eval_rf_oversampling(train, val, threshold=0.32)
    xgb_model = train_eval_xgb(train, val, threshold=0.26)
    cat_model = train_eval_cat(train, val, threshold=0.36)

    # =========================
    # 4. Train and Save pipeline stacking ensemble (on train+val)
    # =========================
    save_path = r"E:\My_Github\Credit Card Fraud Detection\models\stacking_pipeline.pkl"
    _ = train_and_save_pipeline(train_val, save_path=save_path, threshold=0.75)

    # =========================
    # 5. Load pipeline & test evaluation
    # =========================
    model_pip = load_pipeline(save_path)

    X_test, y_test = test.drop(columns=['Class']), test['Class']
    test_preds, _ = predict_pipeline(model_pip, X_test)

    print("\n---- Final Test Evaluation with stacking model ----\n")
    evaluation_metrics(y_test, test_preds)


if __name__ == "__main__":
    main()
  
'''
-------- RandomForest with oversampling --------
train :

Precision: 0.9967858812529219
Recall: 1.0
F1 Score: 0.9983903538294946

Confusion Matrix:
 [[170524     55]
 [     0  17057]]

val :

Precision: 0.8020833333333334
Recall: 0.8555555555555555
F1 Score: 0.8279569892473119

Confusion Matrix:
 [[56851    19]
 [   13    77]]

-------- xgboost --------

train :

Precision: 0.993485342019544
Recall: 1.0
F1 Score: 0.9967320261437909

Confusion Matrix:
 [[170577      2]
 [     0    305]]

val :

Precision: 0.8444444444444444
Recall: 0.8444444444444444
F1 Score: 0.8444444444444444

Confusion Matrix:
 [[56856    14]
 [   14    76]]
-------- CatBoost --------

train :

Precision: 0.7984293193717278
Recall: 1.0
F1 Score: 0.8879184861717613

Confusion Matrix:
 [[170502     77]
 [     0    305]]

val :

Precision: 0.8
Recall: 0.8444444444444444
F1 Score: 0.8216216216216217

Confusion Matrix:
 [[56851    19]
 [   14    76]]
 
-------- stacking model --------
train_val :

Precision: 0.9825
Recall: 1.0
F1 Score: 0.9912

Confusion Matrix:
 [[227442      7]
 [     0    395]]

---- Final Test Evaluation with stacking model ----

Precision: 0.8269230769230769
Recall: 0.8865979381443299
F1 Score: 0.8557213930348259

Confusion Matrix:
 [[56845    18]
 [   11    86]]
 
 '''