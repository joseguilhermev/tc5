import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn


# Carregando o modelo e o preprocessador
try:
    import os

    print(f"Current directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    print(f"Models directory exists: {os.path.exists('models')}")
    if os.path.exists("models"):
        print(f"Models directory contents: {os.listdir('models')}")

    model_path = "models/modelo_lgbm_optuna.pkl"
    print(f"Model path exists: {os.path.exists(model_path)}")

    model_data = joblib.load(model_path)
    model = model_data["model"]
    preprocessor = model_data["preprocessor"]
    print("Model loaded successfully!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None
    preprocessor = None


app = FastAPI(
    title="API de Predição de Match de Candidatos",
    description="API para prever se um candidato será encaminhado para uma vaga",
    version="1.0.0",
)


class Candidate(BaseModel):
    local: Optional[str] = None
    objetivo: Optional[str] = None
    nivel_academico: Optional[str] = None
    ingles: Optional[str] = None
    espanhol: Optional[str] = None
    remuneracao: Optional[float] = None


@app.get("/")
def read_root():
    return {
        "message": "API de Predição de Encaminhamento de Candidatos. Acesse /docs para mais informações."
    }


@app.post("/predict")
def predict(candidate: Candidate):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado corretamente")

    try:
        # Criando um DataFrame com os dados do candidato
        df = pd.DataFrame(
            [
                {
                    "local": candidate.local,
                    "objetivo": candidate.objetivo,
                    "nivel_academico": candidate.nivel_academico,
                    "ingles": candidate.ingles,
                    "espanhol": candidate.espanhol,
                    "remuneracao": candidate.remuneracao,
                }
            ]
        )

        # Pré-processando os dados
        processed_data = preprocessor.transform(df)

        # Fazendo a predição
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        result = {
            "encaminhado": bool(prediction),
            "probabilidade": float(probability),
            "dados_candidato": candidate.dict(),
        }

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro ao realizar predição: {str(e)}"
        )
