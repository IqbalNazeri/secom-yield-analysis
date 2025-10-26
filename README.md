\# SECOM Yield Analysis using Machine Learning



This repository contains the full pipeline for the SECOM Semiconductor Yield Prediction project.  

The project applies data preprocessing, feature selection, and machine learning models to improve yield prediction accuracy.





\## ⚙️ Pipeline Overview

1\. \*\*Data Preprocessing\*\* → Handle missing values, duplicates, and type consistency.  

2\. \*\*Exploratory Data Analysis (EDA)\*\* → Understand feature distributions and correlations.  

3\. \*\*Feature Engineering\*\* → Apply scaling and normalization.  

4\. \*\*Feature Selection (LASSO \& RFE)\*\* → Identify key variables impacting yield.  

5\. \*\*Model Training (Baseline \& RFE)\*\* → Train models including Decision Tree, Logistic Regression, SVM, Random Forest, and XGBoost.  

6\. \*\*Evaluation \& Comparison\*\* → Compare models before and after RFE using Accuracy, Precision, Recall, and F1-score.



\## 🧠 Notes

\- Target column standardized as \*\*`Label`\*\*

\- Data files must be placed in `data/` before running any script

\- Each stage produces intermediate outputs for reproducibility



\## ▶️ Running the Pipeline

```bash

python scripts/preprocess\_secom.py

python scripts/eda\_secom.py

python scripts/feature\_selection\_lasso.py

python scripts/feature\_selection\_rfe.py

python scripts/model\_train\_baseline.py

python scripts/model\_train\_rfe.py

python scripts/model\_evaluation.py



