# ATM Cash Demand Forecasting

## 📌 Project Overview

This project focuses on **forecasting ATM cash demand** using historical transaction data and contextual features.
The goal is to help banks and financial institutions **optimize cash management** by reducing shortages and excess loads at ATMs.

Dataset contains:

* Transaction data (withdrawals, deposits)
* Temporal features (date, day of week, time of day)
* Contextual features (holidays, events, location type, weather, competitor ATMs)
* Target variable: `Cash_Demand_Next_Day`

---

## ⚙️ Features

* **EDA (Exploratory Data Analysis):**

  * Distribution analysis of numerical and categorical features
  * Correlation and feature-target relationships
  * Time-series demand trends and seasonality
  * ATM-level demand insights

* **Feature Engineering:**

  * Lag features for past demand
  * Rolling statistics (mean, std)
  * Encoding categorical variables

* **Modeling:**

  * Tree-based regressors (RandomForest, XGBoost, LightGBM)
  * Cross-validation and hyperparameter tuning
  * Error metrics: MAE, RMSE, R²

* **Evaluation:**

  * Comparison of different models
  * Feature importance ranking
  * Insights for business decision-making

---

## 📊 Dataset Structure

| Column                    | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `ATM_ID`                  | Unique identifier for each ATM                 |
| `Date`                    | Transaction date                               |
| `Day_of_Week`             | Day name (Monday–Sunday)                       |
| `Time_of_Day`             | Morning, Afternoon, Evening, Night             |
| `Total_Withdrawals`       | Total withdrawals on that day                  |
| `Total_Deposits`          | Total deposits on that day                     |
| `Location_Type`           | Type of ATM location (branch, mall, etc.)      |
| `Holiday_Flag`            | Whether the day was a holiday (0/1)            |
| `Special_Event_Flag`      | Whether a special event occurred (0/1)         |
| `Previous_Day_Cash_Level` | Cash left from the previous day                |
| `Weather_Condition`       | Weather status (Sunny, Rainy, etc.)            |
| `Nearby_Competitor_ATMs`  | Count of competitor ATMs nearby                |
| `Cash_Demand_Next_Day`    | **Target** – Cash demand forecast for next day |

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Libraries:**

  * Data Handling → `pandas`, `numpy`
  * Visualization → `matplotlib`, `seaborn`, `plotly`
  * Modeling → `scikit-learn`, `xgboost`, `lightgbm`
  * Time-Series → `statsmodels`

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/atm-cash-demand-forecasting.git
   cd atm-cash-demand-forecasting
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run EDA notebook:

   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```
4. Train models:

   ```bash
   jupyter notebook notebooks/Modeling.ipynb
   ```

---

## 📈 Results

* Best-performing model: **LightGBM** (MAE ≈ XXXX, RMSE ≈ XXXX, R² ≈ XXXX)
* Key features influencing demand:

  * Previous day cash level
  * Withdrawals & deposits
  * Day of week & time of day
  * Holidays & events

---

## 📌 Future Work

* Add deep learning models (LSTM/GRU for time-series).
* Deploy model as an API for real-time forecasting.
* Build dashboard for ATM managers.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo, create a branch, and submit a pull request.

---
