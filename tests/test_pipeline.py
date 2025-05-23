import unittest
import os
import json
import tempfile
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock
from pipeline.pipeline import run_pipeline


class TestPipeline(unittest.TestCase):
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
            "3": {
                "infos_basicas": {
                    "codigo_profissional": "A003",
                    "nome": "Pedro Alves",
                    "local": "Belo Horizonte",
                    "objetivo_profissional": "Engenheiro de Dados",
                },
                "formacao_e_idiomas": {
                    "nivel_academico": "Doutorado",
                    "nivel_ingles": "Intermediário",
                    "nivel_espanhol": "Básico",
                },
                "informacoes_profissionais": {"remuneracao": "R$7000,00"},
            },
        }

        # Adicionar mais dados para treinamento adequado do modelo
        for i in range(4, 50):
            status = "Encaminhado" if i % 2 == 0 else "Reprovado"
            local = [
                "São Paulo",
                "Rio de Janeiro",
                "Belo Horizonte",
                "Brasília",
                "Porto Alegre",
            ][i % 5]
            objetivo = [
                "Analista de Dados",
                "Cientista de Dados",
                "Engenheiro de Dados",
                "Desenvolvedor Python",
                "Analista de BI",
            ][i % 5]
            nivel = ["Superior Completo", "Mestrado", "Doutorado", "Pós-graduação"][
                i % 4
            ]
            ingles = ["Avançado", "Intermediário", "Básico", "Fluente"][i % 4]
            espanhol = ["Avançado", "Intermediário", "Básico", "Fluente"][i % 4]
            remuneracao = f"R${3000 + i * 100},00"

            test_applicants[str(i)] = {
                "infos_basicas": {
                    "codigo_profissional": f"A{i:03d}",
                    "nome": f"Candidato {i}",
                    "local": local,
                    "objetivo_profissional": objetivo,
                },
                "formacao_e_idiomas": {
                    "nivel_academico": nivel,
                    "nivel_ingles": ingles,
                    "nivel_espanhol": espanhol,
                },
                "informacoes_profissionais": {"remuneracao": remuneracao},
            }

        # Dados de teste para prospects
        test_prospects = {
            "vaga1": {
                "prospects": [
                    {
                        "codigo": f"A{i:03d}",
                        "situacao_candidado": (
                            "Encaminhado" if i % 2 == 0 else "Reprovado"
                        ),
                    }
                    for i in range(1, 50)
                ]
            }
        }

        # Salvar os arquivos temporários
        with open(self.applicants_path, "w", encoding="utf-8") as f:
            json.dump(test_applicants, f)

        with open(self.prospects_path, "w", encoding="utf-8") as f:
            json.dump(test_prospects, f)

        # Vamos usar um patch para o treinamento ser mais rápido
        # O patch será aplicado no próprio teste

    def tearDown(self):
        # Limpar arquivos temporários
        self.temp_dir.cleanup()

    @patch("optuna.create_study")
    def test_run_pipeline(self, mock_create_study):
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

        # Executar a pipeline completa
        model = run_pipeline(self.applicants_path, self.prospects_path)

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
