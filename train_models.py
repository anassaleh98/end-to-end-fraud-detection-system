from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from utils_data import over_sample, evaluation_metrics


# 1- Random Forest Model
def train_eval_rf_oversampling(train, val, threshold = 0.5):
    # Features and target
    X_train = train.drop(columns=['Class'])  # Remove target column
    y_train = train['Class']

    X_val = val.drop(columns=['Class'])
    y_val = val['Class']
    
    scaler = StandardScaler()

    # Fit on training data and transform both train and val
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # oversampling
    X_train , y_train = over_sample(X_train, y_train)

    rf_model = RandomForestClassifier(
    n_estimators=100,    
    min_samples_split=10,  
    min_samples_leaf=5,  
    max_features="sqrt",  
    max_samples=0.8,  
    bootstrap=True,  
    class_weight="balanced",  
    random_state=42
    )

    rf_model.fit(X_train, y_train)
    
    y_train_prob = rf_model.predict_proba(X_train)[:, 1]
    y_val_prob = rf_model.predict_proba(X_val)[:, 1]  # Probabilities for fraud class (Class = 1)
     
    
    y_train_pred = (y_train_prob >= threshold).astype(int) 
    y_val_pred = (y_val_prob >= threshold).astype(int)  # Convert probabilities to binary labels
    
    print('\n---- RandomForest with oversampling ----')
    print('\ntrain :\n')
    evaluation_metrics(y_train, y_train_pred)
    print('\nval :\n')
    evaluation_metrics(y_val, y_val_pred)    
    
    return rf_model
    
#-----------------------------
# 2- model  XGB    

def train_eval_xgb(train, val, threshold = 0.5):
    # Features and target
    X_train = train.drop(columns=['Class'])  # Remove target column
    y_train = train['Class']

    X_val = val.drop(columns=['Class'])
    y_val = val['Class']
    
    scaler = StandardScaler()

    # Fit on training data and transform both train and val
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)


    xgb_model = XGBClassifier(
        n_estimators=400,         
        max_depth=6,              
        learning_rate=0.09,        
        subsample=0.8,            
        colsample_bytree=0.7,     
        scale_pos_weight=1000 * (len(y_train) - sum(y_train)) / sum(y_train),  # handle imbalance
        random_state=42,
        eval_metric="aucpr",    
        objective="binary:logistic"
        
    )

    xgb_model.fit(
    X_train, y_train,
    #eval_set=[(X_val, y_val)],       
    #verbose=True
     )
    
    y_train_prob = xgb_model.predict_proba(X_train)[:, 1]
    y_val_prob = xgb_model.predict_proba(X_val)[:, 1] 
     
    
    y_train_pred = (y_train_prob >= threshold).astype(int) 
    y_val_pred = (y_val_prob >= threshold).astype(int) 
    
    print('\n----xgboost----')
    print('\ntrain :\n')
    evaluation_metrics(y_train, y_train_pred)
    print('\nval :\n')
    evaluation_metrics(y_val, y_val_pred)
    
    return xgb_model

#-----------------------------
# 3-  CatBoost model   

def train_eval_cat(train, val, threshold = 0.5):
    # Features and target
    X_train = train.drop(columns=['Class'])  
    y_train = train['Class']

    X_val = val.drop(columns=['Class'])
    y_val = val['Class']
    
    scaler = StandardScaler()

    # Fit on training data and transform both train and val
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    cat_model = CatBoostClassifier(
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

    cat_model.fit(X_train, y_train)
    
    y_train_prob = cat_model.predict_proba(X_train)[:, 1]
    y_val_prob = cat_model.predict_proba(X_val)[:, 1]  

    y_train_pred = (y_train_prob >= threshold).astype(int) 
    y_val_pred = (y_val_prob >= threshold).astype(int)  
    
    print('\n----CatBoost----')
    print('\ntrain :\n')
    evaluation_metrics(y_train, y_train_pred)
    print('\nval :\n')
    evaluation_metrics(y_val, y_val_pred)   
    
    return cat_model