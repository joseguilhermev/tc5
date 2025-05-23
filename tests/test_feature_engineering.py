import unittest
import pandas as pd
from pipeline.feature_engineering import feature_engineering


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Criar um dataframe de teste
        self.test_data = pd.DataFrame(
            {
                "codigo": ["A001", "A002", "A003"],
                "nome": ["João Silva", "Maria Souza", "Pedro Alves"],
                "local": ["São Paulo", "Rio de Janeiro", "Belo Horizonte"],
                "objetivo": ["Analista", "Cientista", "Engenheiro"],
                "nivel_academico": ["Superior", "Mestrado", "Doutorado"],
                "ingles": ["Avançado", "Fluente", "Intermediário"],
                "espanhol": ["Básico", "Intermediário", "Básico"],
                "remuneracao": ["R$5000,00", "R$8000,00", "R$7000,00"],
                "status": ["Encaminhado", "Reprovado", "Encaminhado"],
            }
        )

    def test_feature_engineering(self):
        # Aplicar a função feature_engineering
        transformed_df = feature_engineering(self.test_data)

        # Verificar se o dataframe foi transformado corretamente
        self.assertIsInstance(transformed_df, pd.DataFrame)

        # Verificar se as colunas foram removidas
        removed_columns = ["codigo", "nome", "status"]
        for col in removed_columns:
            self.assertNotIn(col, transformed_df.columns)

        # Verificar se a coluna y foi criada e tem os valores corretos
        self.assertIn("y", transformed_df.columns)
        self.assertEqual(transformed_df["y"].tolist(), [1, 0, 1])

        # Verificar se a remuneração foi convertida para numérico
        self.assertTrue(pd.api.types.is_numeric_dtype(transformed_df["remuneracao"]))
        self.assertEqual(
            transformed_df["remuneracao"].tolist(), [5000.0, 8000.0, 7000.0]
        )

        # Verificar se o dataframe original não foi modificado
        self.assertEqual(
            self.test_data["remuneracao"].tolist(),
            ["R$5000,00", "R$8000,00", "R$7000,00"],
        )


if __name__ == "__main__":
    unittest.main()
