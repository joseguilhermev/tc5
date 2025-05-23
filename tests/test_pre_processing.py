import unittest
import json
import os
import tempfile
import pandas as pd
from pipeline.pre_processing import load_data


class TestPreProcessing(unittest.TestCase):
    def setUp(self):
        # Criar arquivos de teste temporários com dados mínimos
        self.temp_dir = tempfile.TemporaryDirectory()
        self.applicants_path = os.path.join(self.temp_dir.name, "test_applicants.json")
        self.prospects_path = os.path.join(self.temp_dir.name, "test_prospects.json")

        # Dados de teste para applicants
        test_applicants = {
            "1": {
                "infos_basicas": {
                    "codigo_profissional": "A001",
                    "nome": "João Silva",
                    "local": "São Paulo",
                    "objetivo_profissional": "Analista de Dados",
                },
                "formacao_e_idiomas": {
                    "nivel_academico": "Superior Completo",
                    "nivel_ingles": "Avançado",
                    "nivel_espanhol": "Básico",
                },
                "informacoes_profissionais": {"remuneracao": "R$5000,00"},
            },
            "2": {
                "infos_basicas": {
                    "codigo_profissional": "A002",
                    "nome": "Maria Souza",
                    "local": "Rio de Janeiro",
                    "objetivo_profissional": "Cientista de Dados",
                },
                "formacao_e_idiomas": {
                    "nivel_academico": "Mestrado",
                    "nivel_ingles": "Fluente",
                    "nivel_espanhol": "Intermediário",
                },
                "informacoes_profissionais": {"remuneracao": "R$8000,00"},
            },
        }

        # Dados de teste para prospects
        test_prospects = {
            "vaga1": {
                "prospects": [
                    {"codigo": "A001", "situacao_candidado": "Encaminhado"},
                    {"codigo": "A002", "situacao_candidado": "Reprovado"},
                ]
            }
        }

        # Salvar os arquivos temporários
        with open(self.applicants_path, "w", encoding="utf-8") as f:
            json.dump(test_applicants, f)

        with open(self.prospects_path, "w", encoding="utf-8") as f:
            json.dump(test_prospects, f)

    def tearDown(self):
        # Limpar arquivos temporários
        self.temp_dir.cleanup()

    def test_load_data(self):
        # Testar a função load_data
        df = load_data(self.applicants_path, self.prospects_path)

        # Verificar se o dataframe foi criado corretamente
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)  # Devemos ter 2 registros

        # Verificar se todas as colunas esperadas estão presentes
        expected_columns = [
            "codigo",
            "nome",
            "local",
            "objetivo",
            "nivel_academico",
            "ingles",
            "espanhol",
            "remuneracao",
            "status",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Verificar valores específicos
        self.assertEqual(
            df.loc[df["codigo"] == "A001", "status"].values[0], "Encaminhado"
        )
        self.assertEqual(
            df.loc[df["codigo"] == "A002", "status"].values[0], "Reprovado"
        )


if __name__ == "__main__":
    unittest.main()
