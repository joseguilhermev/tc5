import unittest
import os
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock
from pipeline.model_training import train_model


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Criar um dataframe de teste para o treinamento do modelo
        # Criando um dataset sintético com mais registros para o treinamento
        data = []
        for i in range(50):
            # Criando 25 exemplos de cada classe
            status = "Encaminhado" if i < 25 else "Reprovado"
            data.append(
                {
                    "local": f"Cidade {i % 5}",
                    "objetivo": f"Cargo {i % 10}",
                    "nivel_academico": f"Nível {i % 3}",
                    "ingles": ["Básico", "Intermediário", "Avançado", "Fluente"][i % 4],
                    "espanhol": ["Básico", "Intermediário", "Avançado"][i % 3],
                    "remuneracao": float(3000 + i * 100),
                    "status": status,
                }
            )

        self.test_df = pd.DataFrame(data)
        # Aplicar as transformações de feature engineering
        self.test_df["y"] = self.test_df["status"].apply(
            lambda x: 1 if "Encaminhado" in x else 0
        )
        self.test_df = self.test_df.drop(columns=["status"])

    @patch("optuna.create_study")
    def test_model_training(self, mock_create_study):
        # Configurar o mock para retornar um estudo que executa apenas 1 trial
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study

        # Configurar o método optimize do estudo mock
        mock_study.optimize = MagicMock()
        mock_study.best_params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
        }

        # Treinar o modelo
        model = train_model(self.test_df)

        # Verificar se o modelo foi criado
        self.assertIsNotNone(model)

        # Verificar se optimize foi chamado
        mock_study.optimize.assert_called_once()

        # Verificar se o arquivo do modelo foi salvo
        model_path = "models/modelo_lgbm_optuna.pkl"
        self.assertTrue(os.path.exists(model_path))

        # Carregar o modelo e verificar a estrutura
        saved_model = joblib.load(model_path)
        self.assertIn("model", saved_model)
        self.assertIn("preprocessor", saved_model)


if __name__ == "__main__":
    unittest.main()
