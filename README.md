# Student Performance Classification using Machine Learning and Dimensionality Reduction

A machine learning and pattern recognition project predicting student academic performance (`GradeClass`) on a dataset of over 30,000 records, evaluating the impact of dimensionality reduction techniques.

## Overview & Methodology
- **Preprocessing:** Handled missing values, encoded categorical features, split data (80% train / 20% test), and applied feature scaling.
- **Models:** Compared **Random Forest** and **Logistic Regression**.
- **Dimensionality Reduction:** Tested models across three setups:
  1. All features (Baseline)
  2. **PCA** (Principal Component Analysis)
  3. **Feature Selection** (Chi-Square)
- **Hyperparameter Tuning:** Tuned Random Forest (`n_estimators`: 50, 100, 150) using Grid Search.
- **Evaluation:** Evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

