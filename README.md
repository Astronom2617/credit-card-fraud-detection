# Credit Card Fraud Detection

Binary classification on a highly imbalanced dataset. The goal is to detect fraudulent transactions while minimizing false negatives — a missed fraudster causes real financial losses, a false alarm is just an inconvenience.

---

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions made by European cardholders over 48 hours
- 492 fraudulent transactions — **0.17% of all transactions**
- Features: `V1`–`V28` (anonymized PCA components), `Time`, `Amount`
- Target: `Class` — 0 = legitimate, 1 = fraud

---

## Project Structure

| Notebook | Description |
|---|---|
| `Fraud_01_EDA.ipynb` | Class imbalance analysis, Amount/Time distributions, top discriminative V-features |
| `Fraud_02_Baseline.ipynb` | Time-based split, DummyClassifier (accuracy paradox), Logistic Regression baseline |
| `Fraud_03_Models.ipynb` | class_weight, SMOTE oversampling, threshold tuning, feature importance |

---

## Key Design Decisions

**Time-based split (not random)**
Transactions are sorted by `Time` — first 80% go to train, last 20% to test.
Random split would leak future data into training, producing unrealistically optimistic metrics.

**Why not accuracy**
A model that always predicts "not fraud" achieves 99.87% accuracy while catching zero fraudsters.
This is the accuracy paradox. Primary metrics here are **Precision, Recall, F1, and PR-AUC** for the fraud class.

**Recall over Precision**
A missed fraudster = real financial loss. A blocked honest client = temporary inconvenience.
All modeling decisions prioritize recall, with precision as a secondary constraint.

**Feature Engineering**
`Hour` was derived from `Time` via `(Time % 86400) // 3600`. EDA confirmed fraud peaks at hours 0–5 with density ~4–5x higher than non-fraud — fraudsters are most active when victims are asleep.

---

## Results

All models evaluated on the same time-based test split (56,962 transactions, 75 fraud cases).

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Naive Baseline (DummyClassifier) | 0.00 | 0.00 | 0.00 | - |
| Logistic Regression (default) | 0.86 | 0.57 | 0.69 | - |
| Logistic Regression (balanced) | 0.05 | 0.91 | 0.09 | - |
| Random Forest (balanced) | 0.98 | 0.68 | 0.80 | - |
| Random Forest + SMOTE (0.1) | 0.98 | 0.72 | 0.83 | - |
| **RF + SMOTE + threshold 0.28** | **0.76** | **0.80** | **0.78** | **0.807** |

**Final model: Random Forest + SMOTE (sampling_strategy=0.1) + threshold 0.28**

---

## Key Findings

- **Accuracy paradox confirmed**: DummyClassifier achieves 99.87% accuracy while detecting zero fraud
- **SMOTE at 1:10 ratio** outperforms 50/50 balancing — synthetic data grounded in real examples, no noise introduced into the majority class region
- **Threshold tuning is a business decision**: moving threshold from 0.5 to 0.28 increases recall from 0.72 to 0.80 at the cost of precision dropping from 0.98 to 0.76
- **EDA validated by model**: top V-features from EDA (V14, V17, V12, V10) match RF feature importances — the statistical signal found manually is the same signal the model learned

---

## Stack

- Python, pandas, numpy
- scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
