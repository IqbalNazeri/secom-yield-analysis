"""
feature_selection_rfe.py

Recursive Feature Elimination (RFE) for SECOM dataset.
Refines feature subset after LASSO selection.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================================
# Step 1: Load Cleaned SECOM Dataset
# ==========================================================
print("Loading cleaned SECOM dataset...")
data_path = r"C:\Users\USER\Documents\SECOM PRESENTATION DATASET\cleaned_secom.csv"
df = pd.read_csv(data_path)

# ==========================================================
# Step 2: Separate Features (X) and Labels (y)
# ==========================================================
# Assuming the last column is 'label' or 'target'
if 'label' in df.columns:
    X = df.drop(columns=['label'])
    y = df['label']
else:
    # In case label is not named 'label'
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

print(f"Dataset shape: {X.shape}, Target shape: {y.shape}")

# ==========================================================
# Step 3: Data Scaling (Z-score Standardization)
# ==========================================================
print("Standardizing features (Z-score normalization)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================================
# Step 4: Split Dataset for Validation
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================================
# Step 5: Initialize Logistic Regression Model
# ==========================================================
model = LogisticRegression(max_iter=100000, solver='liblinear')

# ==========================================================
# Step 6: Apply Recursive Feature Elimination (RFE)
# ==========================================================
print("Applying Recursive Feature Elimination (RFE)...")
n_features_to_select = 50  # Adjust based on your analysis goals
rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)

# ==========================================================
# Step 7: Retrieve Selected Features
# ==========================================================
selected_features = X.columns[rfe.support_]
ranking = rfe.ranking_

# Combine into a DataFrame for clarity
rfe_results = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': ranking,
    'Selected': rfe.support_
}).sort_values(by='Ranking', ascending=True)

# ==========================================================
# Step 8: Save Results
# ==========================================================
output_path = r"C:\Users\USER\Documents\SECOM PRESENTATION DATASET\Cleaned_Data\rfe_selected_features.csv"
rfe_results.to_csv(output_path, index=False)
print(f"RFE feature selection completed. Results saved to:\n{output_path}")

# ==========================================================
# Step 9: Summary
# ==========================================================
print(f"\nTop {n_features_to_select} Selected Features:")
print(selected_features.tolist())
