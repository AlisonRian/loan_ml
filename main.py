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
def load_model(path: str):
    try:
        model_path = BASE_DIR / path
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo ({path}): {e}")
        return None

@st.cache_resource
def load_encoder():
    try:
        label_path = BASE_DIR / "label_encoder.joblib"
        label_encoder = joblib.load(label_path)
        return label_encoder
    except Exception as e:
        st.error(f"Erro ao carregar o label encoder: {e}")
        return None

st.set_page_config(page_title="Simulador de Empréstimo", page_icon="💳", layout="centered")

st.title("💳 Simulador de Aprovação de Empréstimo")

# Seletor do modelo
model_option = st.selectbox(
    "Escolha a versão do modelo preditivo:",
    [
        "Modelo Completo (Todas as Features - Recomendado)",
        "Modelo Incompleto (Features Reduzidas)"
    ]
)

is_full_model = model_option == "Modelo Completo (Todas as Features - Recomendado)"

if is_full_model:
    st.info("ℹ️ **Modelo Completo:** Utiliza todas as 11 variáveis do cliente (incluindo histórico de crédito e prazo). Apresenta maior precisão (ROC-AUC ~76%).")
    model = load_model("lr_all_features.joblib")
else:
    st.warning("⚠️ **Modelo Incompleto:** Utiliza apenas 7 variáveis básicas. Menor capacidade preditiva em comparação ao modelo completo.")
    model = load_model("logistic_regression_pipeline.joblib")

label_encoder = load_encoder()

st.subheader("📋 Dados do Solicitante")
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Sexo", ["Male", "Female"], horizontal=True)
    married = st.radio("Casado(a)", ["No", "Yes"], horizontal=True)
    education = st.radio("Escolaridade", ["Graduate", "Not Graduate"], horizontal=True)
    self_employed = st.radio("Autônomo(a)", ["No", "Yes"], horizontal=True)

with col2:
    dependents = st.selectbox("Dependentes", ("0", "1", "2", "3", "3+"))
    applicant_income = st.number_input("Rendimento do Solicitante", min_value=0.0, value=5000.0, step=500.0)
    loan_amount = st.number_input("Valor do Empréstimo (em milhares)", min_value=0.0, value=150.0, step=10.0)

# Campos adicionais específicos para o Modelo Completo
if is_full_model:
    st.subheader("🔍 Informações Adicionais (Exigidas pelo Modelo Completo)")
    col3, col4 = st.columns(2)
    
    with col3:
        coapplicant_income = st.number_input("Rendimento do Co-solicitante", min_value=0.0, value=0.0, step=500.0)
        term_options = sorted([12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0])
        loan_amount_term = st.selectbox(
            "Prazo do Empréstimo (em meses)",
            options=term_options,
            index=term_options.index(360.0)
        )
    
    with col4:
        credit_history_option = st.radio(
            "Histórico de Crédito",
            ["Sim (Positivo / Sem Restrição)", "Não (Com Restrição / Sem Histórico)"]
        )
        credit_history = 1.0 if "Sim" in credit_history_option else 0.0

        property_area = st.selectbox("Localização do Imóvel", ["Urban", "Semiurban", "Rural"])

if st.button("Enviar Simulação", type="primary", use_container_width=True):
    if model is None or label_encoder is None:
        st.error("Não foi possível carregar o modelo ou codificador.")
    else:
        # Montagem do DataFrame conforme o modelo selecionado
        if is_full_model:
            data = pd.DataFrame({
                "Gender": [gender],
                "Married": [married],
                "Education": [education],
                "Self_Employed": [self_employed],
                "Dependents": [dependents],
                "ApplicantIncome": [float(applicant_income)],
                "LoanAmount": [float(loan_amount)],
                "Property_Area": [property_area],
                "Credit_History": [float(credit_history)],
                "CoapplicantIncome": [float(coapplicant_income)],
                "Loan_Amount_Term": [float(loan_amount_term)]
            })
        else:
            data = pd.DataFrame({
                "Gender": [gender],
                "Married": [married],
                "Education": [education],
                "Self_Employed": [self_employed],
                "Dependents": [dependents],
                "ApplicantIncome": [float(applicant_income)],
                "LoanAmount": [float(loan_amount)]
            })

        try:
            pred = model.predict(data)[0]
            final_pred = label_encoder.inverse_transform([pred])[0]
            prob = model.predict_proba(data)[0, 1]

            st.write("---")
            if final_pred == "Y":
                st.success("### ✅ Empréstimo APROVADO!")
            else:
                st.error("### ❌ Empréstimo REPROVADO!")

            st.metric(
                label="Probabilidade Estimada de Aprovação",
                value=f"{prob:.1%}"
            )
        except Exception as e:
            st.error(f"Erro ao processar predição: {e}")
