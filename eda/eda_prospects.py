# eda_prospects.py
import json
import pandas as pd
import matplotlib.pyplot as plt

# Carrega os dados JSON
with open("raw_data/prospects.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Lista para armazenar os prospects de todas as vagas
registros = []

# Itera pelas vagas
for vaga_id, vaga_data in data.items():
    titulo = vaga_data.get("titulo", "")
    for prospect in vaga_data.get("prospects", []):
        registros.append(
            {
                "vaga_id": vaga_id,
                "titulo_vaga": titulo,
                "nome": prospect.get("nome"),
                "codigo": prospect.get("codigo"),
                "status": prospect.get("situacao_candidado"),
                "data_candidatura": prospect.get("data_candidatura"),
                "recrutador": prospect.get("recrutador"),
            }
        )

# Cria DataFrame
df = pd.DataFrame(registros)

# Converte datas
df["data_candidatura"] = pd.to_datetime(
    df["data_candidatura"], errors="coerce", dayfirst=True
)

# Resumo geral
print("Total de registros:", len(df))
print("\nStatus mais frequentes:")
print(df["status"].value_counts())

print("\nTop recrutadores:")
print(df["recrutador"].value_counts().head(5))

# Gráficos
df["status"].value_counts().plot(
    kind="bar", title="Distribuição de Status dos Candidatos"
)
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df["recrutador"].value_counts().head(10).plot(
    kind="barh", title="Top 10 Recrutadores por Volume de Encaminhamentos"
)
plt.xlabel("Número de Candidatos")
plt.tight_layout()
plt.show()

# Candidaturas ao longo do tempo
df.dropna(subset=["data_candidatura"]).groupby(
    df["data_candidatura"].dt.to_period("M")
).size().plot(kind="line", title="Evolução de Candidaturas ao Longo do Tempo")
plt.ylabel("Candidaturas")
plt.xlabel("Data")
plt.tight_layout()
plt.show()
