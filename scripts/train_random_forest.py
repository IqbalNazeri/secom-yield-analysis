"""
train_random_forest.py
----------------------
Trains only the Random Forest model on the cleaned SECOM dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# ==============================
# STEP 3: TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# STEP 4: SCALING
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("trained_models", exist_ok=True)
joblib.dump(scaler, "trained_models/scaler.pkl")

# ==============================
# STEP 5: TRAIN RANDOM FOREST MODEL
# ==============================
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("🚀 Training Random Forest model...")
rf_model.fit(X_train_scaled, y_train)

# ==============================
# STEP 6: SAVE MODEL
# ==============================
joblib.dump(rf_model, "trained_models/RandomForest.pkl")
print("✅ Random Forest model saved successfully as 'trained_models/RandomForest.pkl'")
