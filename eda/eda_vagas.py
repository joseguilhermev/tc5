# eda_vagas.py
import json
import pandas as pd
import matplotlib.pyplot as plt

# Carrega os dados
with open("raw_data/vagas.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Processa os dados
vagas = []
for id_vaga, d in data.items():
    info = d.get("informacoes_basicas", {})
    perfil = d.get("perfil_vaga", {})

    vagas.append(
        {
            "id_vaga": id_vaga,
            "cliente": info.get("cliente", ""),
            "titulo": info.get("titulo_vaga", ""),
            "tipo_contratacao": info.get("tipo_contratacao", ""),
            "analista_responsavel": info.get("analista_responsavel", ""),
            "cidade": perfil.get("cidade", ""),
            "estado": perfil.get("estado", ""),
            "nivel_profissional": perfil.get("nivel profissional", ""),
            "nivel_academico": perfil.get("nivel_academico", ""),
            "ingles": perfil.get("nivel_ingles", ""),
            "espanhol": perfil.get("nivel_espanhol", ""),
            "area_atuacao": perfil.get("areas_atuacao", ""),
        }
    )

# Cria DataFrame
df = pd.DataFrame(vagas)

# Resumo
print("Total de vagas:", len(df))
print("\nTipos de contratação mais comuns:")
print(df["tipo_contratacao"].value_counts())

print("\nNíveis profissionais mais buscados:")
print(df["nivel_profissional"].value_counts())

# Gráficos
df["tipo_contratacao"].value_counts().plot(kind="bar", title="Tipo de Contratação")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df["nivel_profissional"].value_counts().plot(
    kind="bar", title="Nível Profissional Requerido"
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df["ingles"].value_counts().plot(kind="bar", title="Exigência de Inglês")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df["cliente"].value_counts().head(10).plot(
    kind="barh", title="Top 10 Clientes com Mais Vagas"
)
plt.xlabel("Número de Vagas")
plt.tight_layout()
plt.show()
