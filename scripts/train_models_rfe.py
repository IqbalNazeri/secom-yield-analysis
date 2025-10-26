"""
train_models_rfe.py
-------------------
Retrains the 5 machine learning models using the RFE-selected features dataset.
Outputs: trained_models_rfe/ folder containing models and scaler.
"""

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==============================
# STEP 1: LOAD CLEANED DATA AND RFE-SELECTED FEATURES
# ==============================
cleaned_data_path = r"C:\Users\USER\Documents\SECOM PRESENTATION DATASET\cleaned_secom.csv"
data = pd.read_csv(cleaned_data_path, dtype=str)
data['Label'] = pd.to_numeric(data['Label'])
print(f"✅ Loaded original dataset with shape: {data.shape}")

rfe_features_path = r"C:\Users\USER\Documents\SECOM PRESENTATION DATASET\Cleaned_Data\rfe_selected_features.csv"
rfe_features_df = pd.read_csv(rfe_features_path, dtype={'Feature': str})
selected_features = rfe_features_df[rfe_features_df['Selected'] == True]['Feature'].tolist()

# Ensure 'Label' is not in selected_features if it was mistakenly included
if 'Label' in selected_features:
    selected_features.remove('Label')

# Filter the original data to keep only the selected features and the 'Label' column
data = data[selected_features + ['Label']]
print(f"✅ Filtered dataset with RFE-selected features. New shape: {data.shape}")

# ==============================
# STEP 2: SPLIT FEATURES AND LABELS
# ==============================
X = data.drop('Label', axis=1)
y = data['Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# STEP 3: SCALING
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create folder for models
save_dir = "trained_models_rfe"
os.makedirs(save_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
print("💾 Saved new RFE scaler")

# ==============================
# STEP 4: DEFINE MODELS
# ==============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=100000, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
}

# ==============================
# STEP 5: TRAIN AND SAVE MODELS
# ==============================
for name, model in models.items():
    print(f"🚀 Training {name}...")
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, os.path.join(save_dir, f"{name}.pkl"))
    print(f"✅ {name} trained and saved!")

print("\n🎉 All models trained successfully on RFE-selected features!")
print(f"Models and scaler saved in: {os.path.abspath(save_dir)}")
