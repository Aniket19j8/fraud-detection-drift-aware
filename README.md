# Fraud Detection with Drift-Aware Evaluation

> A data mining project focused on building a **recall-optimized fraud detection model** and evaluating its **performance drift over time** using ~30K transaction records.

---

## 📌 Project Overview

This repository contains all code, notebooks, and documentation for my individual project in the Data Mining course at : CSE 572.

The project addresses the real-world challenge of detecting rare fraudulent transactions in streaming financial data, while **monitoring performance drift** to ensure long-term model reliability.

---

## 🎯 Objectives

- Build a baseline **binary classification** model for fraud detection.
- Focus on achieving **high recall** (catching as many frauds as possible).
- Simulate **streaming data** by splitting dataset into time-based batches.
- Evaluate **performance drift** using recall trends and **Population Stability Index (PSI)**.
- Analyze **recall vs review workload trade-offs** to inform manual review staffing.

---

## 📂 Dataset

- Total: ~30,000 anonymized transaction records  
- Split into four chronological batches:
  - `batch1_train.csv` → Train (10K)
  - `batch2_test.csv` → Test (10K)
  - `batch3_stream.csv` → Drift evaluation
  - `batch4_stream.csv` → Drift evaluation



---

## ⚙️ Data Mining Pipeline

**steps:**
1. **Data Ingestion** — load CSV batches
2. **Data Cleaning & Preprocessing**  
   - Handle missing values, encode categoricals  
   - Scale numeric features  
   - Address class imbalance (SMOTE / class weights)
3. **Exploratory Data Analysis (EDA)**  
   - Class distribution plots  
   - Amount histograms  
   - Correlation heatmaps
4. **Model Training & Evaluation**  
   - Models: Logistic Regression, Random Forest, :contentReference[oaicite:3]{index=3}  
   - Metrics: Precision, Recall, F1, Recall@FPR, PR-AUC
5. **Concept Drift Evaluation**  
   - Train on Batch 1, test on later batches  
   - Track recall decay and PSI
6. **Threshold Tuning for Recall vs Review Analysis**  
   - Fix target recall (e.g. 0.80)  
   - Estimate manual review workload at this threshold

---

## 📈 Evaluation Metrics

| Metric            | Purpose                                  |
|-------------------|-------------------------------------------|
| Precision / Recall / F1 | Baseline classification metrics |
| PR-AUC             | Robust to class imbalance                |
| Recall@Fixed FPR   | Measures ability to catch fraud while limiting false positives |
| PSI (Population Stability Index) | Detects data drift across batches |

---

## 🧪 Results (Planned)

- Achieve recall ≥ 0.80 on holdout test data
- Visualize how recall drops on future batches
- Quantify workload needed to maintain target recall
- Highlight any concept drift impacting performance

---



