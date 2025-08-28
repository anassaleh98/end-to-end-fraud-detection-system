import pandas as pd
import numpy as np
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.linear_model import LogisticRegression
from utils_data import over_sample, evaluation_metrics


    

def stacking_ensemble(train_val, threshold=0.3):
    # -----------------
    # Split features & target
    X_train, y_train = train_val.drop(columns=['Class']), train_val['Class']
    

    # Scale main features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # -----------------
    # RandomForest with oversampling
    X_rf_train_os, y_rf_train_os = over_sample(X_train_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        max_samples=0.8,
        bootstrap=True,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_rf_train_os, y_rf_train_os)

    # -----------------
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.09,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=1000 * (len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42,
        eval_metric="aucpr",
        objective="binary:logistic"
    )
    xgb.fit(X_train_scaled, y_train)

    # -----------------
    # CatBoost
    cat = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        eval_metric="Recall",
        loss_function="Logloss",
        class_weights=[1, 120],
        bagging_temperature=0.2,
        border_count=256,
        verbose=0,
        random_seed=42
    )
    cat.fit(X_train_scaled, y_train)

    # -----------------
    # Scale extra features ('Hours', 'V17','V14','V12', 'V11','V4')
    extra_scaler = StandardScaler()
    extra_features_train = extra_scaler.fit_transform(
        train_val[['Hours', 'V17','V14','V12', 'V11','V4']]
    )


    # -----------------
    # Build meta-features for TRAIN
    rf_train_prob  = rf.predict_proba(X_train_scaled)[:, 1]
    xgb_train_prob = xgb.predict_proba(X_train_scaled)[:, 1]
    cat_train_prob = cat.predict_proba(X_train_scaled)[:, 1]

    meta_X_train = np.column_stack((rf_train_prob, xgb_train_prob, cat_train_prob, extra_features_train))
    meta_y_train = y_train

    # Train meta-learner
    meta_model = LogisticRegression(
        max_iter=100,
        class_weight={0: 1, 1: 50},
        random_state=42,
        verbose=0
    )
    meta_model.fit(meta_X_train, meta_y_train)

    # -----------------
    # Evaluate on TRAIN_VAL
    meta_train_prob = meta_model.predict_proba(meta_X_train)[:, 1]
    meta_train_pred = (meta_train_prob >= threshold).astype(int)

    print("\n----stacking model----:")
    print('\ntrain_val :\n')
    evaluation_metrics(y_train, meta_train_pred)


    return rf, xgb, cat, meta_model, scaler, extra_scaler

