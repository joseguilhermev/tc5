import pandas as pd


def feature_engineering(df):
    df = df.copy()
    df["remuneracao"] = (
        df["remuneracao"].astype(str).str.replace("R$", "").str.replace(",", ".")
    )
    df["remuneracao"] = pd.to_numeric(df["remuneracao"], errors="coerce")
    df["y"] = df["status"].apply(lambda x: 1 if "Encaminhado" in x else 0)
    df = df.drop(columns=["codigo", "nome", "status"])
    return df
