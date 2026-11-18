# 💳 Simulador de Aprovação de Empréstimo

Aplicação em Machine Learning para prever a aprovação ou reprovação de empréstimos bancários com base no perfil financeiro e pessoal do solicitante. O projeto conta com uma interface interativa desenvolvida em **Streamlit** e um modelo treinado com **Scikit-Learn**.

---

## 🚀 Como Executar

### 1. Criar e ativar o ambiente virtual
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 3. Iniciar a aplicação
```bash
streamlit run main.py
```
Acesse no navegador: `http://localhost:8501`

---

## 📂 Arquivos do Projeto

- **`main.py`**: Aplicação web com Streamlit para simulação em tempo real com alternância dinâmica entre modelos.
- **`loan_classifier.ipynb`**: Notebook com análise exploratória, pré-processamento e treinamento dos modelos.
- **`loan.csv`**: Base de dados histórica de empréstimos.
- **`lr_all_features.joblib`**: Modelo completo treinado com todas as 11 variáveis (maior precisão).
- **`logistic_regression_pipeline.joblib`**: Modelo simplificado treinado com 7 variáveis básicas.
- **`label_encoder.joblib`**: Codificador das classes do modelo.
- **`requirements.txt`**: Bibliotecas necessárias para execução.

---

## 🛠️ Tecnologias

- Python 3
- Streamlit
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Pandas & NumPy
- Joblib
