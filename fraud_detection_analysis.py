import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = "outputs"
import os
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: Generating synthetic transaction dataset")
print("=" * 70)

N = 50000
FRAUD_RATE = 0.015  # ~1.5% fraud, realistic imbalance

n_fraud = int(N * FRAUD_RATE)
n_normal = N - n_fraud

def generate_normal_transactions(n):
    return pd.DataFrame({
        "amount": np.round(np.random.gamma(shape=2.0, scale=40, size=n), 2),
        "hour_of_day": np.random.normal(14, 4, n).clip(0, 23).astype(int),
        "distance_from_home_km": np.round(np.abs(np.random.normal(10, 15, n)), 2),
        "distance_from_last_txn_km": np.round(np.abs(np.random.normal(5, 10, n)), 2),
        "txns_last_24h": np.random.poisson(3, n),
        "is_foreign_country": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "is_new_device": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "merchant_risk_score": np.round(np.random.beta(2, 8, n), 3),  # low risk skew
        "card_present": np.random.choice([0, 1], size=n, p=[0.3, 0.7]),
    })

def generate_fraud_transactions(n):
    return pd.DataFrame({
        "amount": np.round(np.random.gamma(shape=2.5, scale=110, size=n), 2),  # overlaps with normal
        "hour_of_day": np.random.choice(
            list(range(24)), size=n,
            p=np.array([0.06]*3 + [0.03]*15 + [0.04]*6) / np.sum([0.06]*3 + [0.03]*15 + [0.04]*6)
        ),
        "distance_from_home_km": np.round(np.abs(np.random.normal(120, 180, n)), 2),
        "distance_from_last_txn_km": np.round(np.abs(np.random.normal(80, 150, n)), 2),
        "txns_last_24h": np.random.poisson(6, n),
        "is_foreign_country": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
        "is_new_device": np.random.choice([0, 1], size=n, p=[0.55, 0.45]),
        "merchant_risk_score": np.round(np.random.beta(4, 4, n), 3),  # overlaps with normal
        "card_present": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
    })

normal_df = generate_normal_transactions(n_normal)
normal_df["is_fraud"] = 0

fraud_df = generate_fraud_transactions(n_fraud)
fraud_df["is_fraud"] = 1

df = pd.concat([normal_df, fraud_df], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle
df["transaction_id"] = ["TXN" + str(i).zfill(7) for i in range(len(df))]

# introduce a few missing values and duplicates, like real-world messy data
missing_idx = np.random.choice(df.index, size=50, replace=False)
df.loc[missing_idx, "merchant_risk_score"] = np.nan
df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)  # inject dupes

# flip a small % of labels to simulate real-world label noise / mislabeled cases
# (kept low since fraud is already a tiny minority - too much noise would erase the signal)
flip_idx = np.random.choice(df.index, size=int(0.006 * len(df)), replace=False)
df.loc[flip_idx, "is_fraud"] = 1 - df.loc[flip_idx, "is_fraud"]

df.to_csv(f"{OUT_DIR}/transactions_raw.csv", index=False)
print(f"Generated {len(df)} transactions ({df['is_fraud'].sum()} fraud, "
      f"{df['is_fraud'].mean()*100:.2f}% fraud rate)")
print(f"Saved raw dataset to {OUT_DIR}/transactions_raw.csv\n")

print("=" * 70)
print("STEP 2: Exploratory Data Analysis")
print("=" * 70)

print("\n--- Basic info ---")
print(df.info())

print("\n--- Missing values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n--- Duplicate rows ---")
print(f"Duplicates found: {df.duplicated().sum()}")

print("\n--- Class balance ---")
print(df["is_fraud"].value_counts())
print(df["is_fraud"].value_counts(normalize=True) * 100)

print("\n--- Summary statistics by class ---")
print(df.groupby("is_fraud")[
    ["amount", "distance_from_home_km", "txns_last_24h", "merchant_risk_score"]
].mean().round(2))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.countplot(data=df, x="is_fraud", ax=axes[0, 0], palette=["#4C72B0", "#C44E52"])
axes[0, 0].set_title("Class Balance (0 = Normal, 1 = Fraud)")

sns.boxplot(data=df, x="is_fraud", y="amount", ax=axes[0, 1], palette=["#4C72B0", "#C44E52"])
axes[0, 1].set_title("Transaction Amount by Class")
axes[0, 1].set_ylim(0, df["amount"].quantile(0.95))

sns.histplot(data=df, x="hour_of_day", hue="is_fraud", bins=24, multiple="fill",
             palette=["#4C72B0", "#C44E52"], ax=axes[0, 2])
axes[0, 2].set_title("Fraud Proportion by Hour of Day")

sns.boxplot(data=df, x="is_fraud", y="distance_from_home_km", ax=axes[1, 0],
            palette=["#4C72B0", "#C44E52"])
axes[1, 0].set_title("Distance From Home by Class")
axes[1, 0].set_ylim(0, df["distance_from_home_km"].quantile(0.95))

sns.boxplot(data=df, x="is_fraud", y="merchant_risk_score", ax=axes[1, 1],
            palette=["#4C72B0", "#C44E52"])
axes[1, 1].set_title("Merchant Risk Score by Class")

corr = df.drop(columns=["transaction_id"]).corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2])
axes[1, 2].set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_overview.png", dpi=120)
plt.close()
print(f"\nSaved EDA charts to {OUT_DIR}/eda_overview.png\n")

print("=" * 70)
print("STEP 3: Data Cleaning & Preprocessing")
print("=" * 70)

before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

# impute missing merchant_risk_score with median (robust to outliers)
median_risk = df["merchant_risk_score"].median()
df["merchant_risk_score"] = df["merchant_risk_score"].fillna(median_risk)
print(f"Filled missing merchant_risk_score with median: {median_risk:.3f}")

# sanity clip on hour_of_day
df["hour_of_day"] = df["hour_of_day"].clip(0, 23)

print(f"Final clean dataset shape: {df.shape}\n")

print("=" * 70)
print("STEP 4: Feature Engineering")
print("=" * 70)

# Behavioral / derived features that often carry strong fraud signal
df["is_night_txn"] = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h >= 23) else 0)
df["high_amount_flag"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
df["velocity_score"] = df["txns_last_24h"] * (1 + df["is_new_device"])
df["distance_ratio"] = df["distance_from_last_txn_km"] / (df["distance_from_home_km"] + 1)
df["risk_amount_interaction"] = df["merchant_risk_score"] * np.log1p(df["amount"])

new_features = ["is_night_txn", "high_amount_flag", "velocity_score",
                 "distance_ratio", "risk_amount_interaction"]
print("Engineered features:", new_features)
print(df[new_features + ["is_fraud"]].groupby("is_fraud").mean().round(3))
print()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

print("=" * 70)
print("STEP 5: Train/Test Split & Handling Class Imbalance (SMOTE)")
print("=" * 70)

feature_cols = [
    "amount", "hour_of_day", "distance_from_home_km", "distance_from_last_txn_km",
    "txns_last_24h", "is_foreign_country", "is_new_device", "merchant_risk_score",
    "card_present", "is_night_txn", "high_amount_flag", "velocity_score",
    "distance_ratio", "risk_amount_interaction",
]

X = df[feature_cols]
y = df["is_fraud"]

# Stratified split keeps the same fraud ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean()*100:.2f}% | Test fraud rate: {y_test.mean()*100:.2f}%")

# Scale features (helps Logistic Regression; tree models don't need it but it's harmless)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE ONLY to training data (never touch the test set - that would leak information)
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}\n")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("=" * 70)
print("STEP 6: Model Training")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    ),
}

trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_res, y_train_res)
    trained_models[name] = model
print("All models trained.\n")


from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score
)

print("=" * 70)
print("STEP 7: Model Evaluation")
print("=" * 70)

results_summary = []
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    precision = classification_report(y_test, y_pred, output_dict=True)["1"]["precision"]
    recall = classification_report(y_test, y_pred, output_dict=True)["1"]["recall"]
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    results_summary.append({
        "Model": name, "Precision": round(precision, 3), "Recall": round(recall, 3),
        "F1-Score": round(f1, 3), "PR-AUC": round(pr_auc, 3), "ROC-AUC": round(roc_auc, 3),
    })

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    axes[i].set_title(f"{name}\nConfusion Matrix")
    axes[i].set_ylabel("Actual")
    axes[i].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=120)
plt.close()

results_df = pd.DataFrame(results_summary).sort_values("PR-AUC", ascending=False)
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY (sorted by PR-AUC — best metric for imbalanced fraud data)")
print("=" * 70)
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# Precision-Recall curve comparison (more informative than ROC for imbalanced data)
plt.figure(figsize=(8, 6))
for name, model in trained_models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, y_proba):.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (Test Set)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{OUT_DIR}/precision_recall_curves.png", dpi=120)
plt.close()
print(f"\nSaved confusion matrices and PR curves to {OUT_DIR}/\n")

print("=" * 70)
print("STEP 8: Feature Importance (best model)")
print("=" * 70)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"Best model by PR-AUC: {best_model_name}")

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title(f"Feature Importance — {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120)
    plt.close()

    print("\nTop features driving fraud predictions:")
    print(importances.head(8).round(4).to_string())
    print(f"\nSaved feature importance chart to {OUT_DIR}/feature_importance.png")
elif hasattr(best_model, "coef_"):
    importances = pd.Series(best_model.coef_[0], index=feature_cols).sort_values(
        key=abs, ascending=False
    )
    print("\nTop features (by absolute coefficient weight):")
    print(importances.head(8).round(4).to_string())

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"""
All outputs saved to the '{OUT_DIR}/' folder:
  - transactions_raw.csv       : the generated dataset
  - eda_overview.png           : exploratory analysis charts
  - confusion_matrices.png     : confusion matrix per model
  - precision_recall_curves.png: PR curve comparison
  - model_comparison.csv       : metrics table for all models
  - feature_importance.png     : which features matter most

Recommended model: {best_model_name}
""")
