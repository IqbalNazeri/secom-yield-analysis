"""
model_training.py
-----------------
Trains 5 different machine learning models on the cleaned SECOM dataset:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
5. XGBoost

Each trained model is saved as a .pkl file for future evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

# ==============================
# STEP 1: LOAD CLEANED DATA
# ==============================
data_path = r"C:\Users\USER\Documents\SECOM PRESENTATION DATASET\Cleaned_Data\cleaned_secom.csv"
data = pd.read_csv(data_path, header=None)

print(f"[OK] Loaded dataset with shape: {data.shape}")

# ==============================
# STEP 2: SPLIT FEATURES AND LABELS
# ==============================
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ==============================
# STEP 3: TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# STEP 4: FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, "scaler.pkl")
print("[INFO] Scaler saved as scaler.pkl")

# ==============================
# STEP 5: MODEL INITIALIZATION
# ==============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=100000, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ==============================
# STEP 6: TRAINING LOOP
# ==============================
model_save_dir = "trained_models"
os.makedirs(model_save_dir, exist_ok=True)

for name, model in models.items():
    print(f"[TRAINING] Training {name}...")
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, os.path.join(model_save_dir, f"{name}.pkl"))
    print(f"[OK] {name} model saved successfully!")

print("\n[DONE] Training complete! All models saved in 'trained_models' folder.")
