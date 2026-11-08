import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

dependents_map = {
    '0': 0, '1': 1, '2': 2, '3': 3, '3+': 3
}
BASE_DIR = Path(__file__).parent

def map_dependts(df_or_array):
    s = pd.Series(df_or_array.squeeze()).astype(str)
    return s.map(dependents_map).fillna(0).to_numpy().reshape(-1, 1)

@st.cache_resource
def load_model(path:str):
    try:
        model_path = BASE_DIR / path
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None
    
@st.cache_resource
def load_encoder():
    try:
        label_path = BASE_DIR / "label_encoder.joblib"
        label_encoder = joblib.load(label_path)
        return label_encoder
    except Exception as e:
        st.error(f"Erro ao carregar o label encoder do modelo: {e}")
        return None, None

model = load_model(r"logistic_regression_pipeline.joblib")
label_encoder = load_encoder()

st.title("Simulador de Aprovação de Empréstimo")
col1, col2 = st.columns(2)

with col1:
    gender = st.radio('Sexo', ["Male", "Female"])
    married = st.radio('Casado(a)', ["No", "Yes"])
    education = st.radio('Educação', ["Graduate", "Not Graduate"])
    self_employed = st.radio('Autônomo(a)', ["No", "Yes"])
with col2:
    dependents = st.selectbox("Dependentes", ("0","1","2","3","3+"))
    applicant_income = st.text_input('Rendimento')
    loan_amount = st.text_input('Valor do empréstimo')

if st.button('Enviar', type='primary'):
    data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Education": [education],
        "Self_Employed": [self_employed],
        "Dependents": [dependents],
        "ApplicantIncome": [float(applicant_income) if applicant_income else 0.0],
        "LoanAmount": [float(loan_amount) if loan_amount else 0.0]
    })

    pred = model.predict(data)[0]
    final_pred = label_encoder.inverse_transform([pred])[0]
    prob = model.predict_proba(data)[0, 1]
        
    if final_pred == "Y":
        st.success("### ✅ Empréstimo APROVADO!")
    else:
        st.error("### ❌ Empréstimo REPROVADO!")

    if prob:
        st.metric(
          label="Probabilidade de aprovação",
          value=f"{prob:.1%}"
        )