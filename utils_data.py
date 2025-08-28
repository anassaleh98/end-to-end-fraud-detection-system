import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def read_data():
    train = pd.read_csv(r"E:\My_Github\Credit Card Fraud Detection\data\train.csv")
    val =  pd.read_csv(r"E:\My_Github\Credit Card Fraud Detection\data\val.csv")
    test = pd.read_csv(r"E:\My_Github\Credit Card Fraud Detection\data\test.csv")
    train_val = pd.concat([train, val], ignore_index=True)
    return train, val, train_val, test

def evaluation_metrics(y, y_pred):
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
    
    
def New_features(data):
    data['Minutes'] = data['Time'] / 60
    data['Hours'] = data['Time'] / 3600

    # # AMOUNT CATEGORIES 
    # def get_amount_category(amount):
    #     if amount <= 10:
    #         return 'micro'
    #     elif amount <= 50:
    #         return 'small'
    #     elif amount <= 200:
    #         return 'medium'
    #     elif amount <= 1000:
    #         return 'large'
    #     else:
    #         return 'huge'
    
    # data['amount_category'] = data['Amount'].apply(get_amount_category)
    
    # data = pd.get_dummies(data, columns=['amount_category'], drop_first=True)
   
    return data

def over_sample(X, y):
    counter = Counter(y)                   # counter({0: 170579 , 1: 305})
    factor, majoirty_size = 10, counter[0]
    new_size = int(majoirty_size/factor)
    
    oversample = SMOTE(random_state=42, sampling_strategy={1: new_size}, k_neighbors=2)
    X_os, y_os = oversample.fit_resample(X, y)
    
    return X_os, y_os

def under_sample(X, y, factor=10):
    counter = Counter(y)                 # Counter({0: 170579, 1: 305})
    factor, majoirty_size = 5, counter[0]
    new_size = int(majoirty_size / factor)

    undersample = RandomUnderSampler(random_state=42, sampling_strategy={0: new_size, 1: counter[1]})
    X_us, y_us = undersample.fit_resample(X, y)

    return X_us, y_us

def under_over_sample(X, y):
    counter = Counter(y)                   # counter({0: 170579 , 1: 305})
    min_size = int(counter[0] // 10)
    maj_size = int(counter[0] // 2)
    
    oversample = SMOTE(random_state=42, sampling_strategy={1: min_size}, k_neighbors=2)
    undersample = RandomUnderSampler(random_state=42, sampling_strategy={0: maj_size})
    
    pip = imb_pipeline(steps=[("over", oversample), ("under", undersample)])
    
    X_ovs, y_ovs = pip.fit_resample(X, y)
    
    return X_ovs, y_ovs