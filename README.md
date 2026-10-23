# Fraud Detection Analysis

An end-to-end machine learning pipeline for detecting fraudulent transactions, built to demonstrate a complete data science workflow: from raw data through to a deployable, evaluated model.

## Problem Statement

Financial fraud detection is a classic imbalanced classification problem — fraudulent transactions typically make up less than 2% of all transactions, which makes naive accuracy-based evaluation misleading and requires deliberate handling of class imbalance, feature design, and metric selection.

## Dataset

This project uses a **synthetically generated** transaction dataset (50,000 records, ~1.5–2% fraud rate), created to closely mimic the statistical patterns of real-world fraud data:

- Fraudulent transactions skew toward larger amounts, unusual hours, greater distance from home, and higher merchant risk scores
- Deliberate overlap between fraud/normal distributions (to avoid unrealistically perfect separability)
- Injected missing values, duplicate rows, and ~0.6% label noise to simulate real-world data quality issues

> **Note:** Synthetic data was used since no real transaction dataset was available. The pipeline is designed to be dataset-agnostic — replace the data-generation step with `pd.read_csv('your_file.csv')` and it runs unchanged on real data (see [Using Your Own Data](#using-your-own-data) below).

## Pipeline Overview

| Step | Description |
|------|-------------|
| 1. Data Generation | Synthetic transaction dataset with realistic fraud patterns |
| 2. Exploratory Data Analysis | Class balance, distributions, correlations |
| 3. Data Cleaning | Deduplication, missing value imputation |
| 4. Feature Engineering | Behavioral features: transaction velocity, night-time flag, distance ratios, risk-amount interaction |
| 5. Train/Test Split & SMOTE | Stratified split; SMOTE oversampling applied *only* to training data to avoid leakage |
| 6. Model Training | Logistic Regression, Random Forest, XGBoost |
| 7. Evaluation | Precision, Recall, F1, PR-AUC, ROC-AUC, confusion matrices |
| 8. Interpretation | Feature importance for the best-performing model |

## Results

| Model | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
|-------|-----------|--------|----------|--------|---------|
| Logistic Regression | 0.500 | 0.688 | 0.579 | 0.714 | 0.845 |
| Random Forest | 0.918 | 0.688 | 0.787 | 0.705 | 0.836 |
| XGBoost | **0.957** | 0.688 | **0.801** | 0.704 | 0.842 |

*(Exact numbers will vary slightly on re-run due to random data generation — see script output.)*

**Key takeaway:** All three models catch roughly the same share of fraud (~69% recall), but XGBoost and Random Forest do so with far fewer false alarms (~92–96% precision) than Logistic Regression (50%), demonstrating the value of non-linear models on this feature set.

Accuracy is not reported as a headline metric — with a ~2% fraud rate, a model predicting "no fraud" for everything would already score ~98% accuracy while catching zero fraud. **PR-AUC** and **Recall/Precision** are used instead, as they reflect actual performance on the minority (fraud) class.

## Repository Structure

```
├── fraud_detection_analysis.py   # Main pipeline script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── outputs/
    ├── eda_overview.png           # Exploratory analysis charts
    ├── confusion_matrices.png     # Per-model confusion matrices
    ├── precision_recall_curves.png# PR curve comparison across models
    ├── feature_importance.png     # Top features driving predictions
    └── model_comparison.csv       # Metrics table for all models
```

## How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python fraud_detection_analysis.py
   ```
4. Outputs (charts, CSVs) are saved to the `outputs/` folder.

## Using Your Own Data

To run this pipeline on a real dataset instead of synthetic data:

1. Replace the Step 1 data-generation block with:
   ```python
   df = pd.read_csv('your_file.csv')
   ```
2. Ensure your target/label column is named `is_fraud` (0 = normal, 1 = fraud), or rename the `feature_cols` and `y` references in Step 5 to match your column names.
3. Adjust the feature engineering section (Step 4) to reflect fields actually present in your data.

Steps 2–8 (EDA, cleaning, imbalance handling, modeling, evaluation) will run unchanged.

## Tech Stack

- **Python** — pandas, NumPy for data handling
- **scikit-learn** — preprocessing, Logistic Regression, Random Forest, evaluation metrics
- **XGBoost** — gradient boosting classifier
- **imbalanced-learn** — SMOTE for class imbalance
- **matplotlib / seaborn** — visualization

## Author

*Add your name here*

## License

*Add a license if this is a public repo (e.g. MIT)*
