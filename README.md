# 💳 Fraud Detection with Drift-Aware Evaluation

> A **data mining project** focused on building a **recall-optimized fraud detection model** and evaluating its **performance drift over time** using ~30 K anonymized credit-card transactions.

---

## 📘 Overview

This repository contains the code, notebooks, and documentation for my **CSE 572 – Data Mining** course project at **Arizona State University**.

The project addresses the challenge of detecting **rare fraudulent transactions** while monitoring **concept drift** — gradual changes in data patterns that degrade model recall over time.  
We designed a **recall-focused pipeline** that adapts dynamically to maintain consistent detection quality in streaming data.

---

## 🎯 Objectives

- Develop a **binary classification** model to distinguish fraud vs. legitimate transactions.  
- Optimize for **recall**, ensuring maximum fraud capture even at moderate precision.  
- Simulate **streaming data** using time-ordered batches.  
- Measure **performance drift** via recall trends, PSI (Population Stability Index), and KS statistics.  
- Quantify the trade-off between **recall stability** and **manual review workload**.

---

## 📂 Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
  - 284,807 transactions with 492 frauds (~0.17 %).  
  - Features: `Time`, `Amount`, and anonymized PCA-transformed variables `V1–V28`.

For this project, a **30 K subset** was sampled chronologically and split into 4 batches:

| Batch | Purpose | Size | Fraud Rate |
|-------|----------|------|------------|
| batch1_train.csv | Training | 10 000 | 0.38 % |
| batch2_test.csv  | Testing  | 10 000 | 0.47 % |
| batch3_stream.csv | Stream 1 (Drift eval) | 10 000 | 0.09 % |
| batch4_stream.csv | Stream 2 (Drift eval) | 10 000 | 0.10 % |

---

## ⚙️ Data-Mining Pipeline

1. **Data Ingestion** – load raw and processed batches.  
2. **Preprocessing** – `AmountTimeScaler` standardizes only `Amount` and `Time`.  
3. **Exploratory Data Analysis** – imbalance plots, histograms, scatter (Time vs Amount).  
4. **Model Training & Evaluation**
   - Models: Logistic Regression (baseline), Random Forest, XGBoost.  
   - Metrics: Precision, Recall, F1, PR-AUC.  
5. **Concept Drift Detection**
   - Compute PSI and KS statistics between Test ↔ Stream batches.  
   - Monitor recall decay over time.  
6. **Adaptive Threshold Recalibration**
   - Re-estimate decision threshold per batch to restore target recall (≈ 0.80).  
   - Compare “Fixed” vs “Adaptive” recall curves.

---

## 📈 Key Results (Checkpoint 2)

| Batch | Mode | Threshold | Precision | Recall | F1 | PR-AUC | Flagged % |
|:------|:-----|:-----------:|:-----------:|:-----------:|:------:|:---------:|:---------:|
| TEST | Fixed | 0.9826 | 0.54 | 0.83 | 0.65 | 0.42 | 0.72 % |
| S1 | Fixed | 0.9826 | 0.56 | 0.56 | 0.56 | 0.37 | 0.09 % |
| S2 | Fixed | 0.9826 | 0.80 | 0.80 | 0.80 | 0.77 | 0.10 % |
| TEST | Adaptive | 0.9826 | 0.54 | 0.83 | 0.65 | 0.42 | 0.72 % |
| S1 | Adaptive | 0.974 | 0.56 | 0.82 | 0.67 | 0.39 | 0.70 % |
| S2 | Adaptive | 0.969 | 0.55 | 0.80 | 0.65 | 0.38 | 0.68 % |

**Interpretation:**
- Recall initially drops on Stream 1 due to distribution shift, but adaptive thresholding recovers performance.  
- Precision remains stable (< ±0.02), indicating manageable false positive rates.  
- Manual review load (~0.7 %) stays constant across streams.

---

## 🌊 Drift Metrics Summary

| Drift Test | PSI (Amount) | PSI (Time) | KS (Score) |
|-------------|--------------|-------------|-------------|
| Test → S1 | 0.073 | 11.513 | 0.165 |
| Test → S2 | 0.116 | 11.513 | 0.244 |

**Interpretation:**  
- Low PSI (< 0.1) on Amount → minor distribution shift.  
- High PSI on Time → strong temporal drift (uneven transaction times).  
- Increasing KS values confirm model-score distribution shift across streams.

---

## 🧩 Technologies Used

- Python 3.12 (VS Code / Anaconda / Jupyter)  
- Libraries: NumPy, Pandas, Scikit-Learn, XGBoost, Matplotlib  
- Version Control: Git + GitHub  
- Environment: `.venv` for reproducible dependencies

---

## 🧠 Insights & Learning

- **Data imbalance** demands careful recall-precision trade-offs.  
- **Concept drift** can be quantified with PSI and KS tests.  
- **Adaptive thresholding** stabilizes recall over time without retrains.  
- **Recall-driven evaluation** aligns better with fraud prevention goals than accuracy.

---

## 🚀 Future Work

- Add **ensemble comparisons** (Random Forest, XGBoost).  
- Evaluate **SMOTE vs class-weight balancing**.  
- Automate **threshold recalibration** in streaming pipeline.  
- Incorporate **real-time drift alerts** using PSI and KS thresholds.  
- Extend dataset to full 284 K transactions for robustness.

---

## 📎 Repository Structure

```text
fraud-detection-drift-aware/
│
├── data/
│   ├── raw/                ← (download from Kaggle)
│   └── processed/          ← 10 k batches for training/testing
│
├── notebooks/
│   ├── 01_setup_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_model_compare.ipynb
│   ├── 04_drift_metrics.ipynb
│   └── 05_threshold_recalibration.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```
---

## 📚 Citation

Dataset source: Dal Pozzolo et al., **“Credit Card Fraud Detection: A Realistic Modeling and New Learning Strategy”**, ULB MLG (2015).  
Available on Kaggle → [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

