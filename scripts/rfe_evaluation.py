"""
rfe_evaluation.py
-----------------
Evaluates the 5 trained models using the RFE-selected features dataset.
Compares performance against baseline to assess the impact of feature selection.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

# ==============================
# STEP 1: LOAD CLEANED DATA
# ==============================
cleaned_data_path = r"C:\Users\USER\Documents\SECOM PRESENTATION DATASET\cleaned_secom.csv"
data = pd.read_csv(cleaned_data_path, dtype=str)
data['Label'] = pd.to_numeric(data['Label'])
print(f"✅ Loaded original dataset with shape: {data.shape}")

# ==============================
# STEP 2: SPLIT FEATURES AND LABELS (before RFE selection)
# ==============================
X_full = data.drop('Label', axis=1)
y = data['Label']

# ==============================
# STEP 3: LOAD SCALER AND SCALE FULL DATA
# ==============================
scaler = joblib.load("trained_models/scaler.pkl")
X_full_scaled = scaler.transform(X_full)
X_full_scaled_df = pd.DataFrame(X_full_scaled, columns=X_full.columns)

# ==============================
# STEP 4: LOAD RFE-SELECTED FEATURES AND FILTER SCALED DATA
# ==============================
rfe_features_path = r"C:\Users\USER\Documents\SECOM PRESENTATION DATASET\Cleaned_Data\rfe_selected_features.csv"
rfe_features_df = pd.read_csv(rfe_features_path, dtype={'Feature': str})
selected_features = rfe_features_df[rfe_features_df['Selected'] == True]['Feature'].tolist()

# Filter the scaled full data to keep only the selected features
X_rfe_scaled = X_full_scaled_df[selected_features]
print(f"✅ Filtered scaled dataset with RFE-selected features. New shape: {X_rfe_scaled.shape}")

# ==============================
# STEP 5: TRAIN-TEST SPLIT (on RFE-selected scaled data)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_rfe_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# STEP 6: LOAD TRAINED MODELS
# ==============================
model_dir = "trained_models_rfe"
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
        print(f"⚠️ Model not found for {name}: {path}")

# ==============================
# STEP 5: EVALUATE EACH MODEL
# ==============================
results = []

for name, model in models.items():
    print(f"🔍 Evaluating {name} on RFE dataset...")
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
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
results_df.to_csv("rfe_results.csv", index=False)

print("\n📊 RFE Evaluation Results:")
print(results_df)
print("\n💾 Results saved as rfe_results.csv")
