"""
baseline_evaluation.py
----------------------
Evaluates the baseline performance of 5 trained models:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
5. XGBoost

Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
Results saved as baseline_results.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

# ==============================
# STEP 1: LOAD CLEANED DATA
# ==============================
data_path = r"C:\Users\USER\Documents\SECOM PRESENTATION DATASET\cleaned_secom.csv"
data = pd.read_csv(data_path)
print(f"✅ Loaded dataset with shape: {data.shape}")

# ==============================
# STEP 2: SPLIT FEATURES AND LABELS
# ==============================
X = data.drop('Label', axis=1)
y = data['Label']

# Use the same split for fairness
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# STEP 3: LOAD SCALER
# ==============================
scaler = joblib.load("trained_models/scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# ==============================
# STEP 4: LOAD TRAINED MODELS
# ==============================
model_dir = "trained_models"
model_files = {
    "Logistic Regression": "LogisticRegression.pkl",
    "SVM": "SVM.pkl",
    "Decision Tree": "DecisionTree.pkl",
    "Random Forest": "RandomForest.pkl",
    "XGBoost": "XGBoost.pkl"
}

models = {}
for name, file in model_files.items():
    path = os.path.join(model_dir, file)
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"✅ Loaded {name}")
    else:
        print(f"⚠️ Model file not found for {name}: {path}")

# ==============================
# STEP 5: EVALUATE EACH MODEL
# ==============================
results = []

for name, model in models.items():
    print(f"🔍 Evaluating {name}...")
    y_pred = model.predict(X_test_scaled)

    # Some models may not support predict_proba
    try:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4) if auc is not None else "N/A"
    })

# ==============================
# STEP 6: SAVE RESULTS
# ==============================
results_df = pd.DataFrame(results)
results_df.sort_values(by="F1-Score", ascending=False, inplace=True)
results_df.to_csv("baseline_results.csv", index=False)

print("\n📊 Baseline Evaluation Results:")
print(results_df)
print("\n💾 Results saved as baseline_results.csv")
