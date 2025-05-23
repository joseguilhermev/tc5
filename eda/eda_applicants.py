# eda_applicants.py
import json
import pandas as pd
import matplotlib.pyplot as plt

# Carrega os dados JSON (troque o caminho pelo seu arquivo real)
with open("raw_data/applicants.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extrai informações principais dos candidatos
candidatos = []
for id, d in data.items():
    infos = d.get("infos_basicas", {})
    formacao = d.get("formacao_e_idiomas", {})
    prof = d.get("informacoes_profissionais", {})

    candidatos.append(
        {
            "codigo": infos.get("codigo_profissional"),
            "nome": infos.get("nome"),
            "email": infos.get("email"),
            "local": infos.get("local"),
            "objetivo": infos.get("objetivo_profissional"),
            "nivel_academico": formacao.get("nivel_academico"),
            "ingles": formacao.get("nivel_ingles"),
            "espanhol": formacao.get("nivel_espanhol"),
            "remuneracao": prof.get("remuneracao"),
        }
    )

df = pd.DataFrame(candidatos)

# Limpeza de dados
df["remuneracao"] = (
    df["remuneracao"]
    .str.replace("R$", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
)
df["remuneracao"] = pd.to_numeric(df["remuneracao"], errors="coerce")

# Exibe resumo
print("Resumo dos Dados:\n")
print(df.info())
print("\nAmostra dos Dados:")
print(df.head())

# Exibe estatísticas de remuneração
print("\nEstatísticas de remuneração:")
print(df["remuneracao"].describe())

# Gera um gráfico (opcional)
df["nivel_academico"].value_counts().plot(
    kind="bar", title="Distribuição do Nível Acadêmico"
)
plt.xlabel("Nível Acadêmico")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()
