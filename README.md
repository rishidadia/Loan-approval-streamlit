# 💸 Loan Approval Prediction App — Streamlit + XGBoost 🚀

An interactive, intelligent dashboard that predicts the likelihood of a loan being approved based on user-provided financial and personal information.  
Built using **Streamlit**, powered by a **fine-tuned XGBoost model**, and enriched with engineered financial health ratios.

---

## 📊 Project Summary

This project simulates a real-world loan approval system. A user-friendly dashboard collects inputs like age, income, credit history, etc., and returns a real-time prediction using a trained machine learning model.

### 🔍 Objective:
> Predict whether a loan will be approved or not — based on historical borrower data.

---

## 🧠 Machine Learning Approach

I trained and evaluated multiple classification models:
- ✅ **Logistic Regression** (Baseline)
- ✅ **Random Forest Classifier**
- ✅ **XGBoost Classifier** *(Selected Final Model)*

### ⚙️ Hyperparameter Tuning:
- 🔄 **RandomizedSearchCV** for wide-range exploration  
- 🎯 **GridSearchCV** for fine-tuning  
- 🧪 **Optuna** for intelligent optimization and automation

---

## 📐 Feature Engineering

To improve model performance, I engineered several key financial ratios:

| Feature Name             | Description                                               |
|--------------------------|-----------------------------------------------------------|
| `loan_percent_income`    | Ratio of loan amount to income                            |
| `debt_burden_score`      | `loan_percent_income / interest_rate`                     |
| `income_per_year_exp`    | `income / employment experience (years)`                  |
| `credit_age_score`       | `credit score / credit history length (in years)`         |

These features were found to be highly predictive and improved model generalization.

---

## 🏆 Model Selection

After comparing all models on Accuracy, Precision, Recall, F1 Score, and ROC AUC:

✅ **XGBoost** was selected as the final model due to:
- Highest ROC AUC
- Balanced Precision/Recall
- Excellent handling of categorical features after dummy encoding

---

## 🖥️ Streamlit App Features

- 🧾 Clean, intuitive UI for entering borrower details
- ⚙️ Feature processing and encoding that mirrors the training pipeline
- 🔮 Real-time prediction output from the XGBoost model
- 🧠 Fully offline — model is loaded from `.pkl` file

---

## 📁 File Structure

Loan-approval-streamlit/
│
├── app.py # Streamlit dashboard
├── best_xgboost_model.pkl # Final trained XGBoost model
├── model_columns.pkl # Dummy-encoded input column names
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files (venv, cache, etc.)
└── .venv/ # Virtual environment (ignored)