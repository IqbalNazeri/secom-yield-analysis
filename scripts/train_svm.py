"""
train_svm.py
-------------
Trains only the Support Vector Machine (SVM) model on the cleaned SECOM dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
# STEP 4: FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("trained_models", exist_ok=True)
joblib.dump(scaler, "trained_models/scaler.pkl")

# ==============================
# STEP 5: TRAIN SVM MODEL
# ==============================
svm_model = SVC(
    kernel='rbf',        # Gaussian kernel, good for non-linear separation
    probability=True,    # Enables probability estimates for evaluation later
    random_state=42
)

print("🚀 Training SVM model...")
svm_model.fit(X_train_scaled, y_train)

# ==============================
# STEP 6: SAVE MODEL
# ==============================
joblib.dump(svm_model, "trained_models/SVM.pkl")
print("✅ SVM model saved successfully as 'trained_models/SVM.pkl'")
