# 💳 End-to-End Fraud Detection System  

## 📌 Project Overview  
This project implements a **Credit Card Fraud Detection System** using Machine Learning with an **ensemble stacking model (Random Forest, XGBoost, CatBoost + Logistic Regression meta-learner)**.  
It includes:  
- **Data preprocessing** with feature engineering.  
- **Handling class imbalance** using SMOTE and undersampling.  
- **Model training and evaluation** (RandomForest, XGBoost, CatBoost).  
- **Stacking Ensemble** for improved fraud detection.  
- **Flask web application** for real-time and batch fraud detection.  
- **Visualization of fraud vs legitimate transactions**.  

---

##  Features  
✅ Fraud detection using **stacking ensemble** (RandomForest + XGBoost + CatBoost with Logistic Regression meta-learner)  
✅ **Feature engineering**                                                                                   
✅ **Class imbalance handling** with multiple strategies:  
   - Oversampling (SMOT)  
   - Undersampling (Random undersampling)  
   - Hybrid Over+Under sampling approaches

✅ **Hyperparameter tuning** using **GridSearchCV** for optimal model selection  
✅ **Confusion Matrix, ROC Curve, Precision, Recall, F1-score reports** saved in `outputs/`     
✅ **Interactive Flask Web App** with single prediction and batch CSV upload             
✅ **Visualization dashboards** for fraud vs legitimate transactions 

---

## 🖼️ Web Application Preview  

### 🔹 Home Page (Single Prediction + CSV Upload)  

### 🔹 Batch Results Page  
- Fraud transactions highlighted  
- Fraud vs Legitimate chart  
- Paginated results  

https://github.com/user-attachments/assets/0794c621-fde0-4e32-951f-066aff0250e0


## 📂 Repository Structure  

```
end-to-end-fraud-detection-system/
│── Notebook/
│   └── Credit Card Fraud Detection Notebook.ipynb   # Jupyter Notebook (EDA + model testing)
│── data/                                            # Train, Validation, Test datasets
│── models/
│   └── stacking_pipeline.pkl                        # Saved ensemble pipeline
│── outputs/                                         # Outputs(confusion_matrix, roc_curve)
│── templates/                                       # HTML templates for Flask app
│   ├── index.html
│   └── batch_result.html
│── static/                                          # Static assets
│   └── css/
│       └── style.css
│── app.py                                           # Flask application
│── utils_data.py                                    # Data loading, preprocessing, resampling
│── train_models.py                                  # Training individual models
│── fraud_detection_stacking.py                      # Stacking ensemble implementation
│── pipeline.py                                      # Training, saving, and predicting pipeline
│── utils_eval.py                                    # Utility functions for evaluation
│── requirements.txt                                 # Dependencies
│── README.md                                        # Project documentation
```


## ⚙️ Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/anassaleh98/end-to-end-fraud-detection-system.git
cd end-to-end-fraud-detection-system
```

### 2️⃣ Create a Virtual Environment  
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

---

## 📊 Model Training  

To train and save the stacking pipeline:
```bash
python pipeline.py
```

This will save the model to `models/stacking_pipeline.pkl`.

---

## 🌐 Running the Flask App  

```bash
python app.py
```

Then open:  
👉 http://127.0.0.1:5000/


