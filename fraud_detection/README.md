---

# **Credit Card Fraud Detection**

This project implements a **machine learning pipeline** to detect fraudulent credit card transactions using a public dataset. The workflow includes **data preprocessing, handling imbalanced classes, feature scaling, model training, evaluation, and visualization**.

---

## **Project Overview**

* **Dataset:** Credit card transactions with features `Time`, `F1..F28`, `Amount`, and `Class`.
* **Problem Type:** Binary classification (fraud vs normal).
* **Challenge:** Extremely imbalanced dataset — fraud cases are very rare.

---

## **Project Steps**

1. **Data Understanding**

   * Explore dataset shape, missing values, and class distribution.
   * Visualize transaction amounts, class distribution, and correlations.

2. **Data Preprocessing**

   * Drop irrelevant features (`Time`).
   * Scale numerical features using `StandardScaler`.
   * Split data into training and test sets.
   * Handle class imbalance with **SMOTE**.

3. **Modeling**

   * Train three machine learning models:

     1. Logistic Regression (baseline, interpretable)
     2. Random Forest (robust, handles non-linear relationships)
     3. XGBoost (optimized for tabular and imbalanced data)
   * Evaluate models using:

     * **Precision, Recall, F1-score** (especially for fraud class)
     * **ROC-AUC Score**
     * **Confusion Matrices**

4. **Visualization**

   * Compare model performance visually:

     * Precision / Recall / F1-score bar chart
     * ROC-AUC bar chart
     * Confusion matrices side by side

**Optional Code for Visualizations** (run after model predictions):



## **Results Summary**

* **Logistic Regression:** High recall but very low precision → flags too many normal transactions.
* **Random Forest:** Balanced performance with high precision and solid recall → few false positives.
* **XGBoost:** Highest recall and ROC-AUC → best for detecting most frauds, slightly more false positives than Random Forest.

**Overall:** XGBoost is the best-performing model for maximizing fraud detection, while Random Forest is a strong alternative if reducing false alarms is a priority.

---

## **Requirements**

* Python 3.8+
* Libraries:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn xgboost
```

---

## **Usage**

1. Clone the repo:

```bash
git clone <repo_url>
```

2. Run the notebook `fraud_detection.ipynb`.
3. Generate visualizations and inspect model performance.

---

