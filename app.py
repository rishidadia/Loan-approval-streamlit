import streamlit as st
import pickle
import pandas as pd

st.title("LOAN APPROVAL CHANCE PREDICTION")
st.header("Enter your details below:")
left_spacer,col1,mid_spacer,col2,right_spacer=st.columns([1,4,4,4,1])
with col1:
    age=st.number_input("Enter your age",min_value=18)
    education=st.selectbox("Select your education",["Master","High School","Bachelor","Associate","Doctorate"])
    income=st.number_input("Enter your annual income",min_value=0,step=5000)
    emp_exp=st.number_input("Enter your years of employment",min_value=0)
    home_own=st.selectbox("Select your home ownership status",["RENT","OWN","MORTGAGE","OTHER"])
with col2:
    loan_amt=st.number_input("Enter your desired loan amount",min_value=0,step=1000)
    intent=st.selectbox("Select purpose of loan",['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT','DEBTCONSOLIDATION'])
    int_rate=st.number_input("Enter your desired interest rate(%)",min_value=0.0,format="%.2f",step=0.5)
    credit_hist=st.number_input("Enter your credit history length in years(rounded off)",min_value=0.0,format="%.2f",step=1.0)
    credit_score=st.number_input("Enter your credit score",min_value=300,max_value=850, step=10)
    previous_default=st.radio("Have you defaulted on a loan before?",["yes","no"])

loan_percent_income=income/loan_amt if income>0 and loan_amt>0 else 0

debt_burden_score=loan_percent_income/int_rate if int_rate>0 else 0
income_per_year_exp=income/emp_exp if emp_exp>0 else income
credit_age_score=credit_score/credit_hist if credit_hist>0 else credit_score


input_dict={
        "age":age,
        "education":education,
        "income":income,
        "emp_exp":emp_exp,
        "home_own":home_own,
        "loan_amt":loan_amt,
        "intent":intent,
        "int_rate":int_rate,
        "credit_hist":credit_hist,
        "credit_score":credit_score,
        "previous_default":previous_default,
        "loan_percent_income":loan_percent_income,
        "debt_burden_score":debt_burden_score,
        "income_per_year_exp":income_per_year_exp,
        "credit_age_score":credit_age_score
    }

input_df=pd.DataFrame([input_dict])

submit=st.button("Calculate chance")

if submit:
    with open("best_xgboost_model.pkl","rb") as f:
        model=pickle.load(f)
    with open("model_columns.pkl","rb") as f:
        model_columns=pickle.load(f)
    
    input_df_encoded=pd.get_dummies(input_df)
    input_df_encoded=input_df_encoded.reindex(columns=model_columns,fill_value=0)
    output=model.predict(input_df_encoded)
    if(output[0]==1):
        st.success("Congratulations! You have a high chance of loan approval.")
    else:
        st.error("Unfortunately, you have a low chance of loan approval.")
    