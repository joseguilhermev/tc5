# 📘 Documentação do Projeto: **Predição de Encaminhamento de Candidatos**

## 🧠 Objetivo

Criar um sistema inteligente capaz de prever se um candidato será **encaminhado para uma vaga** com base em suas informações profissionais e acadêmicas. O pipeline inclui:

* Pré-processamento de dados
* Engenharia de atributos
* Treinamento de modelo com balanceamento (SMOTE)
* Otimização de hiperparâmetros (Optuna)
* API de inferência via FastAPI

---

## 🗂 Estrutura de Diretórios

```
.
├── app.py                      # FastAPI com endpoint /predict
├── compose.yaml               # Docker Compose para subir a API
├── Dockerfile                 # Container Docker da aplicação
├── requirements.txt           # Dependências do projeto
├── models/                    # Modelos treinados e serializados
├── eda/                       # Análises exploratórias (EDA)
├── pipeline/                  # Lógica de pipeline, feature engineering e modelo
├── raw_data/                  # Dados brutos (não versionados)
└── tests/                     # Testes unitários e de integração
```

---

## ⚙️ Como Funciona o Pipeline

### 1. Pré-processamento (`pre_processing.py`)

Carrega os dados dos arquivos `applicants.json` e `prospects.json` e junta as informações por `codigo`.

### 2. Feature Engineering (`feature_engineering.py`)

* Converte a remuneração para numérico.
* Cria a variável alvo `y`: `1` se "Encaminhado", `0` caso contrário.
* Remove colunas irrelevantes para o modelo.

### 3. Treinamento do Modelo (`model_training.py`)

* OneHotEncoder + Imputer (categorias e números).
* Balanceamento com SMOTE.
* Otimização com Optuna (LightGBM).
* Salva modelo com `joblib`.

### 4. Execução da Pipeline (`pipeline.py`)

```bash
python -m pipeline.pipeline
```

---

## 🚀 API de Inferência (FastAPI)

### 📦 Execução Local

```bash
uvicorn app:app --reload
```

Ou com Docker:

```bash
docker compose up --build
```

### 📥 Endpoint

**POST** `/predict`

**Request Body**:

```json
{
  "local": "São Paulo",
  "objetivo": "Analista de Dados",
  "nivel_academico": "Superior",
  "ingles": "Avançado",
  "espanhol": "Básico",
  "remuneracao": 7000.0
}
```

**Resposta**:

```json
{
  "encaminhado": true,
  "probabilidade": 0.84,
  "dados_candidato": {...}
}
```

Acesse a documentação Swagger automática em:
👉 [`/docs`](http://localhost:8000/docs)

---

## 🔬 EDA (Análise Exploratória)

Scripts em `eda/` mostram:

* Distribuição de candidatos por nível acadêmico
* Remunerações
* Evolução temporal das candidaturas
* Recrutadores com mais atividade

Execute individualmente os arquivos `eda_*.py` para análises.

---

## 🧪 Testes

Executar todos os testes:

```bash
pytest tests/
```

Testes cobrem:

* Pré-processamento
* Feature Engineering
* Treinamento
* Pipeline completa
* API (opcional de adicionar)

---

## 📦 Requisitos

Instale via:

```bash
pip install -r requirements.txt
```

Inclui:

* `lightgbm`, `optuna`, `scikit-learn`, `imbalanced-learn`, `fastapi`, `uvicorn` etc.

---

## 📝 Observações

* Os dados estão localizados em `raw_data/` (não versionados por segurança).
* O modelo treinado é salvo em `models/modelo_lgbm_optuna.pkl`.
* O projeto segue boas práticas com modularização, pipeline clara e testes automatizados.
