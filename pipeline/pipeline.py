from pipeline.pre_processing import load_data
from pipeline.model_training import train_model
from pipeline.feature_engineering import feature_engineering


def run_pipeline(applicants_path, prospects_path):
    print("[1] Carregando dados...")
    df = load_data(applicants_path, prospects_path)

    print("[2] Criando features...")
    df = feature_engineering(df)

    print("[3] Treinando modelo e salvando...")
    model = train_model(df)

    print("[4] Pipeline finalizada com sucesso.")
    return model


if __name__ == "__main__":
    run_pipeline("raw_data/applicants.json", "raw_data/prospects.json")
