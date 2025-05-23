import json
import pandas as pd


def load_data(applicants_path, prospects_path):
    with open(applicants_path, "r", encoding="utf-8") as f:
        applicants_raw = json.load(f)
    with open(prospects_path, "r", encoding="utf-8") as f:
        prospects_raw = json.load(f)

    applicants = []
    for k, d in applicants_raw.items():
        info = d.get("infos_basicas", {})
        form = d.get("formacao_e_idiomas", {})
        prof = d.get("informacoes_profissionais", {})
        applicants.append(
            {
                "codigo": info.get("codigo_profissional"),
                "nome": info.get("nome"),
                "local": info.get("local"),
                "objetivo": info.get("objetivo_profissional"),
                "nivel_academico": form.get("nivel_academico"),
                "ingles": form.get("nivel_ingles"),
                "espanhol": form.get("nivel_espanhol"),
                "remuneracao": prof.get("remuneracao"),
            }
        )
    df_applicants = pd.DataFrame(applicants)

    records = []
    for _, vaga in prospects_raw.items():
        for p in vaga.get("prospects", []):
            records.append(
                {"codigo": p.get("codigo"), "status": p.get("situacao_candidado")}
            )
    df_prospects = pd.DataFrame(records)

    df = pd.merge(df_applicants, df_prospects, on="codigo", how="inner")
    return df
