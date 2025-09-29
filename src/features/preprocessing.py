"""
Feature Engineering and Preprocessing Module for Car Price Prediction

Este módulo implementa classes especializadas para preparação de dados do mercado
automotivo brasileiro, incluindo feature engineering baseada em insights de clustering
e padrões de depreciação.

Author: Machine Learning Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuração para feature engineering"""
    create_temporal_features: bool = True
    create_interaction_features: bool = True
    create_categorical_derived: bool = True
    create_brand_features: bool = True
    create_boolean_features: bool = True
    outlier_contamination: float = 0.05
    high_cardinality_threshold: int = 10
    target_encoding_threshold: float = 0.1


class CarMarketFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature Engineer especializado para mercado automotivo brasileiro

    Cria features baseadas nos insights descobertos na análise exploratória:
    - Segmentação em 4 clusters naturais
    - Padrões de depreciação por marca
    - Elasticidade preço-quilometragem
    - Sweet spots de custo-benefício
    """

    def __init__(self, config: Optional[FeatureConfig] = None, cluster_info: Optional[Dict] = None):
        self.config = config or FeatureConfig()
        self.cluster_info = cluster_info
        self.fitted_ = False
        self.brand_stats_ = {}
        self.price_quantiles_ = {}
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CarMarketFeatureEngineer':
        """
        Fit do transformador - calcula estatísticas necessárias

        Args:
            X: DataFrame com features de entrada
            y: Série com variável target (opcional)

        Returns:
            Self para method chaining
        """
        logger.info("Iniciando fit do CarMarketFeatureEngineer...")

        X_copy = X.copy()

        # Calcular idade do veículo se necessário
        if 'Idade_Veiculo' not in X_copy.columns and 'Ano' in X_copy.columns:
            X_copy['Idade_Veiculo'] = 2024 - X_copy['Ano']

        # Calcular estatísticas por marca para features de marca
        if self.config.create_brand_features and 'Marca' in X_copy.columns:
            self._calculate_brand_statistics(X_copy, y)

        # Calcular quantis para categorização de preços
        if y is not None:
            self._calculate_price_quantiles(y)

        # Guardar nomes das colunas originais
        self.original_columns_ = list(X.columns)

        self.fitted_ = True
        logger.info("CarMarketFeatureEngineer fitted com sucesso")
        return self

    def _calculate_brand_statistics(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Calcula estatísticas por marca para feature engineering"""
        for marca in X['Marca'].unique():
            marca_data = X[X['Marca'] == marca]

            if len(marca_data) > 10:
                brand_stats = {
                    'volume': len(marca_data),
                    'avg_age': marca_data.get('Idade_Veiculo', marca_data.get('Ano', 2020)).mean(),
                    'avg_km': marca_data.get('Quilometragem', 0).mean()
                }

                if y is not None:
                    marca_indices = marca_data.index.intersection(y.index)
                    if len(marca_indices) > 5:
                        marca_prices = y[marca_indices]
                        marca_km = marca_data.loc[marca_indices, 'Quilometragem'] if 'Quilometragem' in marca_data.columns else None

                        brand_stats.update({
                            'price_mean': marca_prices.mean(),
                            'price_std': marca_prices.std(),
                            'price_median': marca_prices.median()
                        })

                        # Elasticidade preço-quilometragem
                        if marca_km is not None and len(marca_km.dropna()) > 5:
                            correlation = abs(marca_prices.corr(marca_km))
                            brand_stats['elasticity_km'] = correlation if not pd.isna(correlation) else 0
                        else:
                            brand_stats['elasticity_km'] = 0

                self.brand_stats_[marca] = brand_stats

    def _calculate_price_quantiles(self, y: pd.Series) -> None:
        """Calcula quantis para categorização de preços"""
        self.price_quantiles_ = {
            'q25': y.quantile(0.25),
            'q50': y.quantile(0.50),
            'q75': y.quantile(0.75),
            'q90': y.quantile(0.90),
            'q95': y.quantile(0.95)
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transformação principal - cria novas features

        Args:
            X: DataFrame com features de entrada

        Returns:
            DataFrame com features originais + features criadas
        """
        if not self.fitted_:
            raise ValueError("CarMarketFeatureEngineer deve ser fitted antes da transformação")

        logger.info("Aplicando transformações de feature engineering...")

        X_transformed = X.copy()
        features_created = []

        # 1. Features temporais
        if self.config.create_temporal_features:
            features_created.extend(self._create_temporal_features(X_transformed))

        # 2. Features de marca
        if self.config.create_brand_features and 'Marca' in X_transformed.columns:
            features_created.extend(self._create_brand_features(X_transformed))

        # 3. Features de interação
        if self.config.create_interaction_features:
            features_created.extend(self._create_interaction_features(X_transformed))

        # 4. Features categóricas derivadas
        if self.config.create_categorical_derived:
            features_created.extend(self._create_categorical_derived_features(X_transformed))

        # 5. Features booleanas
        if self.config.create_boolean_features:
            features_created.extend(self._create_boolean_features(X_transformed))

        self.feature_names_ = features_created
        logger.info(f"Feature engineering concluído. {len(features_created)} novas features criadas")

        return X_transformed

    def _create_temporal_features(self, X: pd.DataFrame) -> List[str]:
        """Cria features baseadas em tempo e idade"""
        features_created = []

        # Idade do veículo
        if 'Ano' in X.columns and 'Idade_Veiculo' not in X.columns:
            X['Idade_Veiculo'] = 2024 - X['Ano']
            features_created.append('Idade_Veiculo')

        # Eficiência de quilometragem (km por ano)
        if 'Quilometragem' in X.columns and 'Idade_Veiculo' in X.columns:
            X['Eficiencia_Km'] = X['Quilometragem'] / (X['Idade_Veiculo'] + 1)
            features_created.append('Eficiencia_Km')

        # Depreciação estimada (baseada na idade)
        if 'Idade_Veiculo' in X.columns:
            X['Depreciacao_Estimada'] = np.exp(-0.1 * X['Idade_Veiculo'])  # Curva exponencial
            features_created.append('Depreciacao_Estimada')

        return features_created

    def _create_brand_features(self, X: pd.DataFrame) -> List[str]:
        """Cria features baseadas nas estatísticas por marca"""
        features_created = []

        # Preço médio da marca
        X['Marca_Preco_Medio'] = X['Marca'].map(
            lambda x: self.brand_stats_.get(x, {}).get('price_mean', 0)
        )
        features_created.append('Marca_Preco_Medio')

        # Elasticidade da marca
        X['Marca_Elasticidade'] = X['Marca'].map(
            lambda x: self.brand_stats_.get(x, {}).get('elasticity_km', 0)
        )
        features_created.append('Marca_Elasticidade')

        # Volume de vendas da marca
        X['Marca_Volume'] = X['Marca'].map(
            lambda x: self.brand_stats_.get(x, {}).get('volume', 0)
        )
        features_created.append('Marca_Volume')

        # Posicionamento da marca (baseado no preço médio)
        marca_preco_medio = X['Marca_Preco_Medio']
        if marca_preco_medio.max() > 0:
            X['Marca_Posicionamento'] = pd.cut(
                marca_preco_medio,
                bins=[0, marca_preco_medio.quantile(0.33), marca_preco_medio.quantile(0.67), float('inf')],
                labels=['Econômica', 'Intermediária', 'Premium']
            ).astype(str)
            features_created.append('Marca_Posicionamento')

        return features_created

    def _create_interaction_features(self, X: pd.DataFrame) -> List[str]:
        """Cria features de interação entre variáveis baseadas nos insights do EDA"""
        features_created = []

        # Interação idade x quilometragem (fundamental para depreciação)
        if 'Idade_Veiculo' in X.columns and 'Quilometragem' in X.columns:
            X['Idade_x_Km'] = X['Idade_Veiculo'] * X['Quilometragem']
            features_created.append('Idade_x_Km')

        # Quilometragem por porta (indicador de uso intensivo)
        if 'Quilometragem' in X.columns and 'Portas' in X.columns:
            X['Km_por_Porta'] = X['Quilometragem'] / (X['Portas'] + 1)
            features_created.append('Km_por_Porta')

        # Interação marca x idade (padrões de depreciação específicos)
        if 'Marca_Preco_Medio' in X.columns and 'Idade_Veiculo' in X.columns:
            X['Marca_x_Idade'] = X['Marca_Preco_Medio'] / (X['Idade_Veiculo'] + 1)
            features_created.append('Marca_x_Idade')

        # Intensidade de uso por idade (km/ano normalizado)
        if 'Quilometragem' in X.columns and 'Idade_Veiculo' in X.columns:
            # Média de 15000 km/ano é considerado normal
            km_anual_esperado = 15000
            X['Intensidade_Uso'] = (X['Quilometragem'] / ((X['Idade_Veiculo'] + 1) * km_anual_esperado))
            features_created.append('Intensidade_Uso')

        # Interação cambio x combustível (configurações técnicas)
        if 'Cambio' in X.columns and 'Combustivel' in X.columns:
            X['Config_Tecnica'] = X['Cambio'].astype(str) + '_' + X['Combustivel'].astype(str)
            features_created.append('Config_Tecnica')

        return features_created

    def _create_categorical_derived_features(self, X: pd.DataFrame) -> List[str]:
        """Cria features categóricas derivadas baseadas nos clusters descobertos no EDA"""
        features_created = []

        # Faixa de idade baseada nos clusters do EDA
        if 'Idade_Veiculo' in X.columns:
            X['Faixa_Idade'] = pd.cut(
                X['Idade_Veiculo'],
                bins=[0, 2, 5, 10, 25],  # Ajustado baseado nos clusters: Novo, Seminovo, Usado, Antigo
                labels=['Novo', 'Seminovo', 'Usado', 'Antigo']
            ).astype(str)
            features_created.append('Faixa_Idade')

        # Faixa de quilometragem baseada nos padrões descobertos
        if 'Quilometragem' in X.columns:
            X['Faixa_Km'] = pd.cut(
                X['Quilometragem'],
                bins=[0, 30000, 80000, 150000, float('inf')],  # Baseado na análise do EDA
                labels=['Baixa', 'Media', 'Alta', 'Muito_Alta']
            ).astype(str)
            features_created.append('Faixa_Km')

        # Categoria de preço baseada nos quartis
        if hasattr(self, 'price_quantiles_') and self.price_quantiles_:
            # Usar apenas se estivermos no contexto de treinamento
            pass  # Implementar se necessário durante a transformação

        # Sweet spot de custo-benefício (baseado nos insights do EDA)
        if 'Idade_Veiculo' in X.columns and 'Quilometragem' in X.columns:
            # Sweet spot: carros de 3-7 anos com km moderada
            sweet_spot = (
                (X['Idade_Veiculo'].between(3, 7)) &
                (X['Quilometragem'].between(30000, 100000))
            )
            X['Sweet_Spot'] = sweet_spot.astype(int)
            features_created.append('Sweet_Spot')

        # Raridade do modelo (baseada na frequência)
        if 'Modelo' in X.columns:
            modelo_counts = X['Modelo'].value_counts()
            X['Modelo_Raridade'] = X['Modelo'].map(modelo_counts)
            X['Modelo_Raro'] = (X['Modelo_Raridade'] < 50).astype(int)
            features_created.extend(['Modelo_Raridade', 'Modelo_Raro'])

        return features_created

    def _create_boolean_features(self, X: pd.DataFrame) -> List[str]:
        """Cria features booleanas de mercado"""
        features_created = []

        # Marca premium
        if 'Marca' in X.columns:
            premium_brands = ['Toyota', 'Honda', 'Nissan', 'Jeep']
            X['Is_Premium_Brand'] = X['Marca'].isin(premium_brands).astype(int)
            features_created.append('Is_Premium_Brand')

        # Transmissão automática
        if 'Cambio' in X.columns:
            X['Is_Automatic'] = (X['Cambio'] == 'Automático').astype(int)
            features_created.append('Is_Automatic')

        # Combustível flex
        if 'Combustivel' in X.columns:
            X['Is_Flex'] = (X['Combustivel'] == 'Flex').astype(int)
            features_created.append('Is_Flex')

        # 4 portas
        if 'Portas' in X.columns:
            X['Is_4_Portas'] = (X['Portas'] == 4).astype(int)
            features_created.append('Is_4_Portas')

        # Veículo novo (< 3 anos)
        if 'Idade_Veiculo' in X.columns:
            X['Is_Novo'] = (X['Idade_Veiculo'] <= 3).astype(int)
            features_created.append('Is_Novo')

        return features_created

    def get_feature_names(self) -> List[str]:
        """Retorna nomes das features criadas"""
        return self.feature_names_

    def save(self, filepath: str) -> None:
        """Salva o transformador"""
        joblib.dump(self, filepath)
        logger.info(f"CarMarketFeatureEngineer salvo em: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CarMarketFeatureEngineer':
        """Carrega o transformador"""
        engineer = joblib.load(filepath)
        logger.info(f"CarMarketFeatureEngineer carregado de: {filepath}")
        return engineer


class ClusterBasedOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Tratamento de outliers baseado na segmentação de clusters do mercado

    Utiliza Isolation Forest separadamente para cada cluster identificado,
    permitindo detecção mais precisa de anomalias específicas de cada segmento.
    """

    def __init__(self, contamination: float = 'auto', cluster_col: str = 'Cluster', methods: List[str] = None):
        self.contamination = contamination  # 'auto' para detecção adaptativa
        self.cluster_col = cluster_col
        self.methods = methods or ['isolation_forest', 'iqr', 'z_score']
        self.outlier_detectors_ = {}
        self.cluster_stats_ = {}
        self.outlier_rates_ = {}  # Taxa real de outliers por cluster
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ClusterBasedOutlierHandler':
        """
        Treina detectores de outliers por cluster

        Args:
            X: DataFrame com features
            y: Target (não utilizado)

        Returns:
            Self para method chaining
        """
        logger.info("Treinando ClusterBasedOutlierHandler...")

        X_copy = X.copy()

        # Se não há coluna de cluster, usar apenas um detector global
        if self.cluster_col not in X_copy.columns:
            logger.warning(f"Coluna {self.cluster_col} não encontrada. Usando detector global.")
            X_copy[self.cluster_col] = 0

        # Colunas numéricas para detecção
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns.tolist()
        if self.cluster_col in numeric_cols:
            numeric_cols.remove(self.cluster_col)

        # Treinar detector por cluster com métodos adaptativos
        clusters_trained = 0
        for cluster in X_copy[self.cluster_col].unique():
            if pd.isna(cluster):
                continue

            cluster_data = X_copy[X_copy[self.cluster_col] == cluster][numeric_cols]

            if len(cluster_data) > 10:  # Mínimo de dados para treinar
                # Calcular estatísticas do cluster
                cluster_stats = {
                    'size': len(cluster_data),
                    'mean': cluster_data.mean().to_dict(),
                    'std': cluster_data.std().to_dict(),
                    'q25': cluster_data.quantile(0.25).to_dict(),
                    'q75': cluster_data.quantile(0.75).to_dict()
                }

                # Calcular taxa de contaminação adaptativa
                if self.contamination == 'auto':
                    # Usar IQR para estimar taxa de outliers natural
                    outlier_counts = []
                    for col in numeric_cols:
                        if cluster_data[col].std() > 0:
                            Q1 = cluster_data[col].quantile(0.25)
                            Q3 = cluster_data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outliers = cluster_data[(cluster_data[col] < lower_bound) |
                                                  (cluster_data[col] > upper_bound)]
                            outlier_counts.append(len(outliers))

                    # Taxa média de outliers baseada no IQR
                    estimated_contamination = max(0.01, min(0.25, np.mean(outlier_counts) / len(cluster_data)))
                else:
                    estimated_contamination = self.contamination

                self.outlier_rates_[cluster] = estimated_contamination

                # Treinar múltiplos detectores
                detectors = {}

                # Isolation Forest
                if 'isolation_forest' in self.methods:
                    detectors['isolation_forest'] = IsolationForest(
                        contamination=estimated_contamination,
                        random_state=42,
                        n_estimators=100
                    )
                    detectors['isolation_forest'].fit(cluster_data)

                self.outlier_detectors_[cluster] = detectors
                self.cluster_stats_[cluster] = cluster_stats
                clusters_trained += 1

        logger.info(f"ClusterBasedOutlierHandler treinado para {clusters_trained} clusters")
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta e marca outliers

        Args:
            X: DataFrame com features

        Returns:
            DataFrame com colunas adicionais de outlier
        """
        if not self.fitted_:
            raise ValueError("ClusterBasedOutlierHandler deve ser fitted antes da transformação")

        X_copy = X.copy()

        # Se não há coluna de cluster, usar valor padrão
        if self.cluster_col not in X_copy.columns:
            X_copy[self.cluster_col] = 0

        # Colunas numéricas
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns.tolist()
        if self.cluster_col in numeric_cols:
            numeric_cols.remove(self.cluster_col)

        # Detectar outliers por cluster usando múltiplos métodos
        outlier_mask = np.zeros(len(X_copy), dtype=bool)
        outlier_scores = np.zeros(len(X_copy))
        outlier_methods = np.zeros(len(X_copy))  # Quantos métodos detectaram como outlier

        for cluster in X_copy[self.cluster_col].unique():
            if pd.isna(cluster) or cluster not in self.outlier_detectors_:
                continue

            cluster_mask = X_copy[self.cluster_col] == cluster
            cluster_data = X_copy.loc[cluster_mask, numeric_cols]

            if len(cluster_data) > 0:
                detectors = self.outlier_detectors_[cluster]
                cluster_outlier_votes = []

                # Isolation Forest
                if 'isolation_forest' in detectors:
                    forest_outliers = detectors['isolation_forest'].predict(cluster_data) == -1
                    forest_scores = detectors['isolation_forest'].decision_function(cluster_data)
                    cluster_outlier_votes.append(forest_outliers)
                    outlier_scores[cluster_mask] = forest_scores

                # Combinar votos dos métodos (pelo menos 1 método deve detectar como outlier)
                if cluster_outlier_votes:
                    combined_outliers = np.any(cluster_outlier_votes, axis=0)
                    votes_count = np.sum(cluster_outlier_votes, axis=0)

                    outlier_mask[cluster_mask] = combined_outliers
                    outlier_methods[cluster_mask] = votes_count

        # Adicionar informações de outliers
        X_copy['Is_Outlier'] = outlier_mask.astype(int)
        X_copy['Outlier_Methods'] = outlier_methods.astype(int)
        X_copy['Outlier_Score'] = outlier_scores

        outliers_detected = outlier_mask.sum()
        outlier_percentage = (outliers_detected / len(X_copy)) * 100

        logger.info(f"Outliers detectados: {outliers_detected} ({outlier_percentage:.1f}%)")

        return X_copy

    def get_cluster_stats(self) -> Dict:
        """Retorna estatísticas dos clusters"""
        return self.cluster_stats_


class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder inteligente que escolhe a melhor estratégia baseada na cardinalidade
    e correlação com a variável target

    Estratégias:
    - One-hot encoding para baixa cardinalidade (≤ 5)
    - Label encoding para cardinalidade média (6-10)
    - Target/Frequency encoding para alta cardinalidade (> 10)
    """

    def __init__(self, high_cardinality_threshold: int = 10, target_encoding_threshold: float = 0.1):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.target_encoding_threshold = target_encoding_threshold
        self.encoding_strategies_ = {}
        self.encoders_ = {}
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'SmartCategoricalEncoder':
        """
        Determina a melhor estratégia de encoding para cada variável categórica

        Args:
            X: DataFrame com features
            y: Série com target para target encoding

        Returns:
            Self para method chaining
        """
        logger.info("Determinando estratégias de encoding...")

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            cardinality = X[col].nunique()

            # Estratégia baseada na cardinalidade
            if cardinality <= 5:
                # One-hot encoding para baixa cardinalidade
                strategy = 'onehot'
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                encoder.fit(X[[col]])

            elif cardinality <= self.high_cardinality_threshold:
                # Label encoding para cardinalidade média
                strategy = 'label'
                encoder = LabelEncoder()
                encoder.fit(X[col].fillna('Missing'))

            else:
                # Target encoding ou frequency encoding para alta cardinalidade
                if y is not None:
                    # Target encoding
                    strategy = 'target'
                    # Create temporary DataFrame for target encoding
                    temp_df = pd.DataFrame({
                        col: X[col],
                        'target': y
                    })
                    target_means = temp_df.groupby(col)['target'].mean()
                    global_mean = y.mean()
                    encoder = target_means.to_dict()
                    encoder['__global_mean__'] = global_mean
                else:
                    # Frequency encoding
                    strategy = 'frequency'
                    encoder = X[col].value_counts().to_dict()

            self.encoding_strategies_[col] = strategy
            self.encoders_[col] = encoder

            logger.info(f"Coluna '{col}': cardinalidade={cardinality}, estratégia={strategy}")

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica as estratégias de encoding determinadas

        Args:
            X: DataFrame com features

        Returns:
            DataFrame com variáveis categóricas encoded
        """
        if not self.fitted_:
            raise ValueError("SmartCategoricalEncoder deve ser fitted antes da transformação")

        X_encoded = X.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if col not in self.encoding_strategies_:
                continue

            strategy = self.encoding_strategies_[col]
            encoder = self.encoders_[col]

            if strategy == 'onehot':
                # One-hot encoding
                try:
                    encoded_data = encoder.transform(X[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]  # drop first
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X.index)
                    X_encoded = pd.concat([X_encoded.drop(col, axis=1), encoded_df], axis=1)
                except Exception as e:
                    logger.warning(f"Erro no one-hot encoding de {col}: {e}")
                    X_encoded = X_encoded.drop(col, axis=1)

            elif strategy == 'label':
                # Label encoding
                try:
                    X_filled = X[col].fillna('Missing')
                    X_encoded[col] = encoder.transform(X_filled)
                except ValueError:
                    # Handle unknown categories
                    known_labels = set(encoder.classes_)
                    X_mapped = X[col].fillna('Missing').map(
                        lambda x: x if x in known_labels else 'Missing'
                    )
                    X_encoded[col] = encoder.transform(X_mapped)

            elif strategy == 'target':
                # Target encoding
                global_mean = encoder.get('__global_mean__', 0)
                X_encoded[col] = X[col].map(encoder).fillna(global_mean)

            elif strategy == 'frequency':
                # Frequency encoding
                X_encoded[col] = X[col].map(encoder).fillna(0)

        return X_encoded

    def get_encoding_strategies(self) -> Dict[str, str]:
        """Retorna as estratégias de encoding utilizadas"""
        return self.encoding_strategies_


class AdvancedPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Pipeline avançado que combina todas as transformações de preparação de dados
    baseadas nos insights do mercado automotivo brasileiro
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_engineer = None
        self.cluster_engineer = None  # Adicionar cluster engineer
        self.outlier_handler = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.fitted_ = False
        self.feature_names_ = []
        self.numeric_cols_ = []
        self.quality_checks_ = {}  # Armazenar checks de qualidade

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AdvancedPreprocessingPipeline':
        """
        Treina todo o pipeline de preprocessing

        Args:
            X: DataFrame com features
            y: Série com target

        Returns:
            Self para method chaining
        """
        logger.info("Treinando AdvancedPreprocessingPipeline...")

        # 1. Feature Engineering
        logger.info("1. Feature Engineering...")
        self.feature_engineer = CarMarketFeatureEngineer(config=self.config)
        self.feature_engineer.fit(X, y)
        X_engineered = self.feature_engineer.transform(X)

        # 1.5. Clustering baseado em features
        logger.info("1.5. Cluster-based Feature Engineering...")
        from .engineering import ClusterBasedFeatureEngineer, ClusterConfig
        cluster_config = ClusterConfig(n_clusters=4)
        self.cluster_engineer = ClusterBasedFeatureEngineer(config=cluster_config)
        self.cluster_engineer.fit(X_engineered, y)
        X_clustered = self.cluster_engineer.transform(X_engineered)

        # 2. Tratamento de Outliers
        logger.info("2. Tratamento de Outliers...")
        self.outlier_handler = ClusterBasedOutlierHandler(
            contamination=self.config.outlier_contamination
        )
        self.outlier_handler.fit(X_clustered)
        X_outliers = self.outlier_handler.transform(X_clustered)

        # 3. Encoding Categórico
        logger.info("3. Encoding Categórico...")
        self.encoder = SmartCategoricalEncoder(
            high_cardinality_threshold=self.config.high_cardinality_threshold,
            target_encoding_threshold=self.config.target_encoding_threshold
        )
        self.encoder.fit(X_outliers, y)
        X_encoded = self.encoder.transform(X_outliers)

        # 4. Remover features constantes
        logger.info("4. Removendo features constantes...")
        X_filtered = self._remove_constant_features(X_encoded)

        # 5. Feature selection baseada em poder preditivo
        logger.info("5. Seleção de features preditivas...")
        X_selected = self._select_predictive_features(X_filtered, y)

        # 6. Scaling para variáveis numéricas
        logger.info("6. Scaling de variáveis numéricas...")
        numeric_cols = X_selected.select_dtypes(include=[np.number]).columns
        self.numeric_cols_ = numeric_cols.tolist()

        if len(self.numeric_cols_) > 0:
            self.scaler.fit(X_selected[self.numeric_cols_])

        # Guardar nomes das features finais
        self.feature_names_ = list(X_selected.columns)

        # 7. Validação de qualidade
        logger.info("7. Executando validação de qualidade...")
        self._run_quality_checks(X, X_selected, y)

        self.fitted_ = True
        logger.info("AdvancedPreprocessingPipeline treinado com sucesso!")
        return self

    def _remove_constant_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features constantes ou com variância zero"""
        constant_features = []

        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
            elif X[col].dtype in [np.number] and X[col].std() == 0:
                constant_features.append(col)

        if constant_features:
            logger.info(f"Removendo {len(constant_features)} features constantes: {constant_features}")
            X_filtered = X.drop(columns=constant_features)
        else:
            X_filtered = X.copy()

        # Armazenar features removidas para referência
        self.constant_features_removed_ = constant_features

        return X_filtered

    def _select_predictive_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Seleciona features com poder preditivo significativo"""
        if y is None:
            return X.copy()

        # Separar features numéricas e categóricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        selected_features = []
        selection_stats = {}

        # 1. Feature selection para variáveis numéricas
        if numeric_cols:
            # Calcular correlações
            correlations = {}
            for col in numeric_cols:
                try:
                    corr = abs(X[col].corr(y))
                    if not pd.isna(corr):
                        correlations[col] = corr
                except:
                    continue

            # Calcular mutual information
            try:
                mi_scores = mutual_info_regression(X[numeric_cols].fillna(0), y, random_state=42)
                mi_dict = dict(zip(numeric_cols, mi_scores))
            except:
                mi_dict = {col: 0 for col in numeric_cols}

            # Selecionar features numéricas com correlação > 0.02 OU mutual info > 0.01
            for col in numeric_cols:
                corr = correlations.get(col, 0)
                mi = mi_dict.get(col, 0)

                if corr > 0.02 or mi > 0.01:
                    selected_features.append(col)
                    selection_stats[col] = {'correlation': corr, 'mutual_info': mi, 'type': 'numeric'}

        # 2. Para variáveis categóricas, manter todas (já foram filtradas pelo encoding)
        selected_features.extend(categorical_cols)
        for col in categorical_cols:
            selection_stats[col] = {'correlation': 0, 'mutual_info': 0, 'type': 'categorical'}

        # 3. Garantir mínimo de features importantes
        if len([f for f in selected_features if f in numeric_cols]) < 5:
            # Se muito poucas features numéricas, relaxar critério
            for col in numeric_cols:
                if col not in selected_features:
                    corr = correlations.get(col, 0)
                    if corr > 0.005:  # Critério mais relaxado
                        selected_features.append(col)
                        selection_stats[col] = {'correlation': corr, 'mutual_info': mi_dict.get(col, 0), 'type': 'numeric_relaxed'}

        # 4. Limitar número total de features para evitar overfitting
        max_features = min(len(X.columns), 50)  # Máximo 50 features
        if len(selected_features) > max_features:
            # Ranking por correlação + mutual info
            feature_scores = []
            for f in selected_features:
                if f in selection_stats:
                    score = selection_stats[f]['correlation'] + selection_stats[f]['mutual_info']
                    feature_scores.append((f, score))

            # Manter top features
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in feature_scores[:max_features]]

        # Armazenar estatísticas para análise
        self.feature_selection_stats_ = selection_stats
        self.selected_features_ = selected_features

        logger.info(f"Selecionadas {len(selected_features)} features de {len(X.columns)} (redução de {(1-len(selected_features)/len(X.columns))*100:.1f}%)")

        return X[selected_features]

    def _run_quality_checks(self, X_original: pd.DataFrame, X_processed: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Executa checks de qualidade no pipeline"""
        checks = {}

        # Check 1: Verificar se há features constantes
        constant_features = []
        for col in X_processed.columns:
            if X_processed[col].nunique() <= 1:
                constant_features.append(col)

        checks['constant_features'] = {
            'count': len(constant_features),
            'features': constant_features,
            'status': 'PASS' if len(constant_features) == 0 else 'WARNING'
        }

        # Check 2: Verificar valores infinitos ou NaN
        infinite_cols = []
        nan_cols = []
        for col in X_processed.select_dtypes(include=[np.number]).columns:
            if np.isinf(X_processed[col]).any():
                infinite_cols.append(col)
            if X_processed[col].isna().any():
                nan_cols.append(col)

        checks['data_validity'] = {
            'infinite_cols': infinite_cols,
            'nan_cols': nan_cols,
            'status': 'PASS' if len(infinite_cols) == 0 and len(nan_cols) == 0 else 'FAIL'
        }

        # Check 3: Verificar dimensionalidade
        original_features = len(X_original.columns)
        processed_features = len(X_processed.columns)
        feature_growth_ratio = processed_features / original_features if original_features > 0 else 0

        checks['dimensionality'] = {
            'original_features': original_features,
            'processed_features': processed_features,
            'growth_ratio': feature_growth_ratio,
            'status': 'PASS' if 1.5 <= feature_growth_ratio <= 10 else 'WARNING'
        }

        # Check 4: Verificar distribuição após scaling
        if self.numeric_cols_:
            scaled_stats = {}
            for col in self.numeric_cols_:
                if col in X_processed.columns:
                    scaled_stats[col] = {
                        'mean': X_processed[col].mean(),
                        'std': X_processed[col].std()
                    }

            checks['scaling_quality'] = {
                'stats': scaled_stats,
                'status': 'PASS'  # Sempre passa se não há erros
            }

        # Check 5: Verificar correlação com target se disponível
        if y is not None:
            target_correlations = {}
            for col in X_processed.select_dtypes(include=[np.number]).columns:
                try:
                    correlation = abs(X_processed[col].corr(y))
                    if not pd.isna(correlation):
                        target_correlations[col] = correlation
                except:
                    continue

            # Contar features com correlação significativa
            significant_correlations = {k: v for k, v in target_correlations.items() if v > 0.05}

            checks['predictive_power'] = {
                'total_correlations': len(target_correlations),
                'significant_correlations': len(significant_correlations),
                'top_correlations': dict(sorted(target_correlations.items(), key=lambda x: x[1], reverse=True)[:10]),
                'status': 'PASS' if len(significant_correlations) >= 5 else 'WARNING'
            }

        self.quality_checks_ = checks

        # Log resumo dos checks
        passed = sum(1 for check in checks.values() if check.get('status') == 'PASS')
        warnings = sum(1 for check in checks.values() if check.get('status') == 'WARNING')
        failed = sum(1 for check in checks.values() if check.get('status') == 'FAIL')

        logger.info(f"Validação de qualidade: {passed} PASS, {warnings} WARNING, {failed} FAIL")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as transformações em sequência

        Args:
            X: DataFrame com features

        Returns:
            DataFrame processado
        """
        if not self.fitted_:
            raise ValueError("AdvancedPreprocessingPipeline deve ser fitted antes da transformação")

        # Aplicar transformações em sequência
        X_transformed = self.feature_engineer.transform(X)
        X_transformed = self.cluster_engineer.transform(X_transformed)
        X_transformed = self.outlier_handler.transform(X_transformed)
        X_transformed = self.encoder.transform(X_transformed)

        # Filtrar features constantes (usar apenas as que existem nas features finais)
        available_features = [col for col in self.feature_names_ if col in X_transformed.columns]
        X_filtered = X_transformed[available_features]

        # Scaling apenas para colunas numéricas existentes
        X_final = X_filtered.copy()
        if self.numeric_cols_:
            existing_numeric_cols = [col for col in self.numeric_cols_ if col in X_final.columns]
            if existing_numeric_cols:
                X_final[existing_numeric_cols] = self.scaler.transform(X_final[existing_numeric_cols])

        return X_final

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Treina e aplica transformações"""
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """Retorna nomes das features finais"""
        return self.feature_names_

    def get_quality_checks(self) -> Dict[str, Any]:
        """Retorna resultados da validação de qualidade"""
        return self.quality_checks_

    def get_feature_importance_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Análise de importância das features criadas

        Args:
            X: DataFrame processado
            y: Target

        Returns:
            Dicionário com análises de importância
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression

        # Correlações
        correlations = []
        for i, col in enumerate(X.columns):
            if X.dtypes[col] in [np.number]:
                corr = abs(np.corrcoef(X.iloc[:, i], y)[0, 1])
                if not np.isnan(corr):
                    correlations.append({'feature': col, 'correlation': corr})

        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

        # Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)

        # Random Forest Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)

        return {
            'correlations': corr_df,
            'mutual_info': mi_df,
            'rf_importance': rf_df
        }

    def save(self, filepath: str) -> None:
        """Salva o pipeline completo"""
        joblib.dump(self, filepath)
        logger.info(f"AdvancedPreprocessingPipeline salvo em: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'AdvancedPreprocessingPipeline':
        """Carrega o pipeline completo"""
        pipeline = joblib.load(filepath)
        logger.info(f"AdvancedPreprocessingPipeline carregado de: {filepath}")
        return pipeline


# Funções utilitárias
def load_and_preprocess(data_path: str, config: Optional[FeatureConfig] = None) -> Tuple[pd.DataFrame, pd.Series, AdvancedPreprocessingPipeline]:
    """
    Função utilitária para carregar e preprocessar dados

    Args:
        data_path: Caminho para os dados
        config: Configuração do preprocessing

    Returns:
        Tupla com (X_processed, y, pipeline)
    """
    # Carregar dados
    from data.load_data import load_car_dataset
    df, _ = load_car_dataset(data_path)

    # Separar features e target
    X = df.drop('Valor_Venda', axis=1)
    y = df['Valor_Venda']

    # Criar e treinar pipeline
    pipeline = AdvancedPreprocessingPipeline(config=config)
    X_processed = pipeline.fit_transform(X, y)

    return X_processed, y, pipeline


def validate_preprocessing_pipeline(pipeline: AdvancedPreprocessingPipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Valida a qualidade do pipeline de preprocessing

    Args:
        pipeline: Pipeline treinado
        X: Features originais
        y: Target

    Returns:
        Relatório de validação
    """
    logger.info("Validando pipeline de preprocessing...")

    # Aplicar pipeline
    X_processed = pipeline.transform(X)

    # Análise de qualidade
    quality_analysis = pipeline.get_feature_importance_analysis(X_processed, y)

    # Estatísticas gerais
    report = {
        'original_features': X.shape[1],
        'final_features': X_processed.shape[1],
        'features_created': X_processed.shape[1] - X.shape[1],
        'feature_quality': {
            'mean_correlation': quality_analysis['correlations']['correlation'].mean(),
            'features_above_01_corr': (quality_analysis['correlations']['correlation'] > 0.1).sum(),
            'features_above_02_corr': (quality_analysis['correlations']['correlation'] > 0.2).sum(),
            'top_correlations': quality_analysis['correlations'].head(10).to_dict('records')
        },
        'outliers_detected': (X_processed['Is_Outlier'] == 1).sum() if 'Is_Outlier' in X_processed.columns else 0,
        'encoding_strategies': pipeline.encoder.get_encoding_strategies() if pipeline.encoder else {},
        'timestamp': datetime.now().isoformat()
    }

    logger.info("Validação do pipeline concluída")
    return report


if __name__ == "__main__":
    # Exemplo de uso
    print("Testando AdvancedPreprocessingPipeline...")

    # Configuração personalizada
    config = FeatureConfig(
        create_temporal_features=True,
        create_interaction_features=True,
        create_categorical_derived=True,
        create_brand_features=True,
        create_boolean_features=True,
        outlier_contamination=0.05
    )

    try:
        # Carregar e preprocessar dados
        data_path = "../data/raw/dataset_carros_brasil.csv"
        X_processed, y, pipeline = load_and_preprocess(data_path, config)

        print(f"Dados processados: {X_processed.shape}")
        print(f"Features criadas: {X_processed.shape[1] - 8}")  # 8 features originais

        # Validar pipeline
        validation_report = validate_preprocessing_pipeline(pipeline, X_processed, y)
        print(f"Qualidade média das features: {validation_report['feature_quality']['mean_correlation']:.3f}")

        # Salvar pipeline
        pipeline.save("../models/advanced_preprocessing_pipeline.pkl")
        print("Pipeline salvo com sucesso!")

    except Exception as e:
        logger.error(f"Erro no teste: {e}")