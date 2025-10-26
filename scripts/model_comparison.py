"""
model_comparison.py
-------------------
Compares baseline and RFE model performance results.
Shows metric improvements after feature selection.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# STEP 1: LOAD BOTH RESULTS
# ==============================
baseline_path = "baseline_results.csv"
rfe_path = "rfe_results.csv"

baseline = pd.read_csv(baseline_path)
rfe = pd.read_csv(rfe_path)

baseline.rename(columns={
    "Accuracy": "Accuracy_Baseline",
    "Precision": "Precision_Baseline",
    "Recall": "Recall_Baseline",
    "F1-Score": "F1_Baseline",
    "ROC-AUC": "ROC_Baseline"
}, inplace=True)

rfe.rename(columns={
    "Accuracy": "Accuracy_RFE",
    "Precision": "Precision_RFE",
    "Recall": "Recall_RFE",
    "F1-Score": "F1_RFE",
    "ROC-AUC": "ROC_RFE"
}, inplace=True)

# ==============================
# STEP 2: MERGE AND COMPARE
# ==============================
comparison = pd.merge(baseline, rfe, on="Model", how="inner")

# Compute percentage improvement for each metric
for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC"]:
    comparison[f"{metric}_Improvement (%)"] = (
        (comparison[f"{metric}_RFE"] - comparison[f"{metric}_Baseline"]) /
        comparison[f"{metric}_Baseline"]
    ) * 100

# Save to CSV
comparison.to_csv("model_comparison_results.csv", index=False)

# ==============================
# STEP 3: DISPLAY RESULTS
# ==============================
print("\n📊 Model Performance Comparison (Baseline vs RFE):")
print(comparison.round(3))

# ==============================
# STEP 4 (Optional): VISUALIZATION
# ==============================
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(comparison["Model"]))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics):
    ax.bar(x - width/2, comparison[f"{metric}_Baseline"], width, label=f"{metric} Baseline" if i == 0 else "", alpha=0.7)
    ax.bar(x + width/2, comparison[f"{metric}_RFE"], width, label=f"{metric} RFE" if i == 0 else "", alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(comparison["Model"], rotation=30)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison: Baseline vs RFE")
ax.legend(["Baseline", "RFE"])
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
