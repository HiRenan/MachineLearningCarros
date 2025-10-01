import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json


class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.feature_info = None
        self.model_path = Path(__file__).parent.parent.parent.parent / "models" / "trained_models" / "lasso_modelo_campeao.pkl"
        self.feature_info_path = Path(__file__).parent.parent.parent.parent / "models" / "trained_models" / "feature_info.json"
        self._load_model()

    def _load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            with open(self.feature_info_path, 'r', encoding='utf-8') as f:
                self.feature_info = json.load(f)
            print(f"Modelo carregado com sucesso: {self.feature_info['modelo']}")
            print(f"Total de features: {self.feature_info['n_features']}")
        except Exception as e:
            raise Exception(f"Erro ao carregar modelo: {str(e)}")

    def _create_features(self, input_data: Dict) -> pd.DataFrame:
        ano_atual = 2024
        idade_veiculo = ano_atual - input_data['ano']
        km_por_ano = input_data['quilometragem'] / (idade_veiculo + 1)

        features_dict = {
            'Quilometragem': input_data['quilometragem'],
            'Portas': input_data['portas'],
            'Idade_Veiculo': idade_veiculo,
            'Km_por_Ano': km_por_ano
        }

        for feature in self.feature_info['features']:
            if feature.startswith('Marca_'):
                marca_name = feature.replace('Marca_', '')
                features_dict[feature] = 1 if input_data['marca'] == marca_name else 0

            elif feature.startswith('Modelo_'):
                modelo_name = feature.replace('Modelo_', '')
                features_dict[feature] = 1 if input_data['modelo'] == modelo_name else 0

            elif feature.startswith('Cor_'):
                cor_name = feature.replace('Cor_', '')
                features_dict[feature] = 1 if input_data['cor'] == cor_name else 0

            elif feature.startswith('Cambio_'):
                cambio_name = feature.replace('Cambio_', '')
                features_dict[feature] = 1 if input_data['cambio'] == cambio_name else 0

            elif feature.startswith('Combustivel_'):
                combustivel_name = feature.replace('Combustivel_', '')
                features_dict[feature] = 1 if input_data['combustivel'] == combustivel_name else 0

        df = pd.DataFrame([features_dict])

        for feature in self.feature_info['features']:
            if feature not in df.columns:
                df[feature] = 0

        df = df[self.feature_info['features']]

        return df

    def predict(self, input_data: Dict) -> Tuple[float, float, float, str]:
        try:
            features_df = self._create_features(input_data)

            prediction = self.model.predict(features_df)[0]

            mae = 2527.46
            valor_minimo = prediction - mae
            valor_maximo = prediction + mae

            if prediction < 20000:
                confianca = "média"
            elif prediction > 150000:
                confianca = "média"
            else:
                confianca = "alta"

            return (
                round(prediction, 2),
                round(valor_minimo, 2),
                round(valor_maximo, 2),
                confianca
            )

        except Exception as e:
            raise Exception(f"Erro ao realizar predição: {str(e)}")

    def get_model_info(self) -> Dict:
        return {
            "nome_modelo": self.feature_info['modelo'],
            "r2_score": 0.9803,
            "mae": 2527.46,
            "rmse": 2917.80,
            "total_features": self.feature_info['n_features'],
            "descricao": "Modelo campeão com regularização L1"
        }

    def is_loaded(self) -> bool:
        return self.model is not None


predictor = CarPricePredictor()
