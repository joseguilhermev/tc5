import unittest
import os
import json
import tempfile
import pandas as pd
from pipeline.pre_processing import load_data
from pipeline.feature_engineering import feature_engineering


class TestIntegration(unittest.TestCase):
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

    def test_pre_processing_to_feature_engineering(self):
        """Testa a integração entre o pré-processamento e feature engineering"""
        # Carregar dados usando o pré-processamento
        df = load_data(self.applicants_path, self.prospects_path)

        # Verificar se o dataframe foi criado corretamente
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)  # Devemos ter 2 registros

        # Aplicar feature engineering sobre os dados pré-processados
        transformed_df = feature_engineering(df)

        # Verificar se o dataframe foi transformado corretamente
        self.assertIsInstance(transformed_df, pd.DataFrame)

        # Verificar se a coluna y foi criada corretamente
        self.assertIn("y", transformed_df.columns)

        # Verificar valores específicos da coluna y
        # Estes valores devem corresponder aos status "Encaminhado" (1) e "Reprovado" (0)
        y_values = transformed_df["y"].tolist()
        self.assertEqual(y_values, [1, 0])

        # Verificar se a remuneração foi convertida para numérico
        self.assertTrue(pd.api.types.is_numeric_dtype(transformed_df["remuneracao"]))
        self.assertEqual(transformed_df["remuneracao"].tolist(), [5000.0, 8000.0])

        # Verificar se as colunas removidas não estão mais presentes
        removed_columns = ["codigo", "nome", "status"]
        for col in removed_columns:
            self.assertNotIn(col, transformed_df.columns)


if __name__ == "__main__":
    unittest.main()
