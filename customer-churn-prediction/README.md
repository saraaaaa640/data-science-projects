
---

# 📊 Customer Churn Prediction – IBM Telco Dataset

## 🔎 Project Overview

This project focuses on predicting customer churn using the IBM Telco Customer Churn dataset. The goal is to identify patterns that lead to customer attrition and provide insights that can help businesses design effective retention strategies.

The workflow covers data exploration, preprocessing, model training, and evaluation, with a dedicated notebook for feature visualization and analysis.

---

## 📂 Repository Structure

* **`churn_prediction.ipynb`** → Main notebook for preprocessing, modeling, and evaluation.
* **`dataset_visualization.ipynb`** → Separate notebook with detailed exploratory data analysis (EDA) and feature insights.
* **`README.md`** → Project documentation (this file).

---

## 🛠 Workflow Summary

### 1. Data Understanding

* Explored dataset structure and target variable (`Churn`).
* Analyzed distributions of key features such as tenure, monthly charges, and contract type.
* Visualizations and detailed EDA are available in **`dataset_visualization.ipynb`**.

### 2. Data Preprocessing

* Dropped irrelevant identifiers (`customerID`).
* Fixed data types (e.g., converted `TotalCharges` to numeric).
* Encoded categorical variables (binary and one-hot encoding).
* Scaled continuous features: tenure, monthly charges, total charges.
* Handled class imbalance with **SMOTE** for certain models.

### 3. Modeling

Trained and compared multiple models:

* **Balanced Data Models:** Logistic Regression, SVM.
* **Imbalanced Data Models:** Random Forest, XGBoost.
* Also tested Random Forest and XGBoost on balanced data for comparison.

### 4. Evaluation

* Metrics: Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.
* Compared the effect of class balancing across different algorithms.
* Identified top features driving churn (tenure, monthly charges, total charges, contract type, etc.).

---

## 📈 Key Insights

* **Financial/account features** (tenure, total charges, monthly charges) are the strongest churn predictors.
* **Service-related features** (contract type, internet service, payment method) also influence churn.
* **Demographics** (gender, dependents, partner) showed weaker predictive power.
* Balancing the dataset significantly improves linear model performance (Logistic Regression, SVM).
* Tree-based models (RandomForest, XGBoost) are more robust to imbalance but still benefit from balanced data in some cases.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Open the notebooks:

   ```bash
   jupyter notebook
   ```

---

## 📊 Visualizations

All data exploration and feature analysis plots are available in the separate notebook: **`dataset_visualization.ipynb`**.
This keeps the main prediction notebook focused on preprocessing, modeling, and evaluation.

---
## 📂 Dataset  

This project uses the **IBM Telco Customer Churn** dataset, available on Kaggle:  
🔗 [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  

> Note: You need a Kaggle account to download the dataset. After downloading, place the CSV file in the project directory before running the notebooks.

---

## 📌 Conclusion

This project provides an end-to-end churn prediction pipeline, from EDA to model comparison. It highlights how preprocessing choices (like class balancing) directly affect model outcomes and offers practical insights for improving customer retention strategies.

---
