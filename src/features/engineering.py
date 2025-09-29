"""
Advanced Feature Engineering Module for Brazilian Car Market

Este módulo implementa feature engineering avançada baseada nos 4 clusters descobertos
na análise exploratória e nos padrões de depreciação específicos do mercado brasileiro.

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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.spatial.distance import cdist

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuração para clustering e features baseadas em clusters"""
    n_clusters: int = 4
    cluster_features: List[str] = None
    min_cluster_size: int = 50
    silhouette_threshold: float = 0.3

    def __post_init__(self):
        if self.cluster_features is None:
            self.cluster_features = ['Valor_Venda', 'Idade_Veiculo', 'Quilometragem']


class ClusterBasedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature Engineer baseado na segmentação natural do mercado em 4 clusters:
    1. Econômico Usado (36.6%) - baixo preço, alta idade
    2. Médio Seminovo (61.4% combinado) - preços médios, idades variadas
    3. Luxury (2%) - preços altíssimos, outliers
    """

    def __init__(self, config: Optional[ClusterConfig] = None):
        self.config = config or ClusterConfig()
        self.fitted_ = False
        self.kmeans_ = None
        self.scaler_ = None
        self.cluster_stats_ = {}
        self.cluster_names_ = {}
        self.depreciation_models_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ClusterBasedFeatureEngineer':
        """
        Treina o clustering e calcula estatísticas para feature engineering

        Args:
            X: DataFrame com features
            y: Target para análise de depreciação

        Returns:
            Self para method chaining
        """
        logger.info("Treinando ClusterBasedFeatureEngineer...")

        X_work = X.copy()

        # Calcular idade do veículo se necessário
        if 'Idade_Veiculo' not in X_work.columns and 'Ano' in X_work.columns:
            X_work['Idade_Veiculo'] = 2024 - X_work['Ano']

        # Preparar dados para clustering
        cluster_data, cluster_df = self._prepare_cluster_data(X_work, y)

        # Treinar clustering
        self._fit_clustering(cluster_data)

        # Calcular estatísticas dos clusters
        cluster_labels = self.kmeans_.predict(cluster_data)
        self._calculate_cluster_statistics(cluster_df, y, cluster_labels)

        # Nomear clusters baseado nas características
        self._name_clusters()

        # Treinar modelos de depreciação por cluster
        if y is not None:
            self._fit_depreciation_models(X_work, y, cluster_labels)

        self.fitted_ = True
        logger.info("ClusterBasedFeatureEngineer treinado com sucesso!")
        return self

    def _prepare_cluster_data(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[np.ndarray, pd.DataFrame]:
        """Prepara dados para clustering"""
        cluster_df = X.copy()

        # Features que estarão disponíveis tanto no fit quanto no transform
        prediction_features = ['Idade_Veiculo', 'Quilometragem']

        # Verificar se features essenciais existem
        missing_features = [f for f in prediction_features if f not in cluster_df.columns]
        if missing_features:
            raise ValueError(f"Features essenciais faltando: {missing_features}")

        # Para clustering durante treinamento, usar também o target se disponível
        clustering_features = prediction_features.copy()
        if y is not None:
            cluster_df['Valor_Venda'] = y
            clustering_features.append('Valor_Venda')

        logger.info(f"Features para clustering: {clustering_features}")

        # Extrair dados para clustering e tratar missing values
        cluster_data = cluster_df[clustering_features].fillna(cluster_df[clustering_features].mean())

        # Treinar dois scalers: um para clustering (com target) e outro para predição (sem target)
        self.clustering_scaler_ = StandardScaler()
        cluster_data_scaled = self.clustering_scaler_.fit_transform(cluster_data)

        # Scaler para predição (apenas features disponíveis durante transform)
        prediction_data = cluster_df[prediction_features].fillna(cluster_df[prediction_features].mean())
        self.prediction_scaler_ = StandardScaler()
        prediction_data_scaled = self.prediction_scaler_.fit_transform(prediction_data)

        # Armazenar features de predição para uso no transform
        self.prediction_features_ = prediction_features
        self.prediction_data_scaled_ = prediction_data_scaled  # Para treinar KMeans de predição

        return cluster_data_scaled, cluster_df

    def _fit_clustering(self, data: np.ndarray) -> None:
        """Treina o modelo de clustering"""
        # Usar 4 clusters como descoberto no EDA
        n_clusters = 4

        # Verificar se há dados suficientes
        if len(data) < n_clusters * self.config.min_cluster_size:
            n_clusters = max(2, len(data) // self.config.min_cluster_size)
            logger.warning(f"Dados insuficientes para 4 clusters. Usando {n_clusters} clusters.")

        # Treinar modelo principal (com todas as features incluindo target)
        self.kmeans_ = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans_.fit(data)

        # Treinar modelo para predição (apenas com features de predição)
        if hasattr(self, 'prediction_data_scaled_'):
            self.prediction_kmeans_ = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            prediction_labels = self.prediction_kmeans_.fit_predict(self.prediction_data_scaled_)

            # Mapear labels entre os dois modelos para manter consistência
            # Usar labels do modelo principal como referência
            main_labels = self.kmeans_.labels_
            self._create_label_mapping(main_labels, prediction_labels)

        # Calcular silhouette score para validação
        if len(set(self.kmeans_.labels_)) > 1:
            score = silhouette_score(data, self.kmeans_.labels_)
            logger.info(f"Clustering treinado com {n_clusters} clusters (silhouette score: {score:.3f})")
        else:
            logger.warning("Clustering resultou em apenas 1 cluster")

    def _create_label_mapping(self, main_labels: np.ndarray, pred_labels: np.ndarray) -> None:
        """Cria mapeamento entre labels do modelo principal e de predição"""
        from scipy.stats import mode
        from collections import defaultdict

        # Para cada cluster de predição, encontrar o cluster principal mais comum
        pred_to_main = {}
        for pred_cluster in np.unique(pred_labels):
            pred_mask = pred_labels == pred_cluster
            main_clusters_in_pred = main_labels[pred_mask]
            if len(main_clusters_in_pred) > 0:
                most_common = mode(main_clusters_in_pred, keepdims=True)[0][0]
                pred_to_main[pred_cluster] = most_common

        self.prediction_label_mapping_ = pred_to_main

    def _calculate_cluster_statistics(self, X: pd.DataFrame, y: Optional[pd.Series], labels: np.ndarray) -> None:
        """Calcula estatísticas detalhadas por cluster"""
        for cluster_id in range(self.kmeans_.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]

            stats = {
                'size': cluster_mask.sum(),
                'percentage': (cluster_mask.sum() / len(labels)) * 100
            }

            # Estatísticas numéricas
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in cluster_data.columns:
                    stats[f'{col}_mean'] = cluster_data[col].mean()
                    stats[f'{col}_std'] = cluster_data[col].std()
                    stats[f'{col}_median'] = cluster_data[col].median()

            # Estatísticas do target se disponível
            if y is not None:
                cluster_target = y[cluster_mask]
                stats.update({
                    'target_mean': cluster_target.mean(),
                    'target_std': cluster_target.std(),
                    'target_median': cluster_target.median(),
                    'target_min': cluster_target.min(),
                    'target_max': cluster_target.max()
                })

            # Estatísticas categóricas mais frequentes
            categorical_cols = cluster_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in cluster_data.columns and not cluster_data[col].empty:
                    most_frequent = cluster_data[col].mode()
                    if len(most_frequent) > 0:
                        stats[f'{col}_most_frequent'] = most_frequent.iloc[0]

            self.cluster_stats_[cluster_id] = stats

    def _name_clusters(self) -> None:
        """Nomeia clusters baseado nas características"""
        for cluster_id, stats in self.cluster_stats_.items():
            price_mean = stats.get('target_mean', 0)
            age_mean = stats.get('Idade_Veiculo_mean', 0)

            if price_mean < 50000 and age_mean > 10:
                name = 'Econômico Usado'
            elif price_mean > 100000:
                name = 'Luxury'
            elif price_mean > 80000 and age_mean < 5:
                name = 'Premium Novo'
            else:
                name = 'Médio Seminovo'

            self.cluster_names_[cluster_id] = name

        logger.info(f"Clusters nomeados: {self.cluster_names_}")

    def _fit_depreciation_models(self, X: pd.DataFrame, y: pd.Series, labels: np.ndarray) -> None:
        """Treina modelos de depreciação por cluster"""
        for cluster_id in range(self.kmeans_.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            cluster_target = y[cluster_mask]

            if 'Idade_Veiculo' in cluster_data.columns and len(cluster_data) > 10:
                # Regressão linear simples: preço vs idade
                age = cluster_data['Idade_Veiculo']
                price = cluster_target

                # Calcular coeficientes da regressão
                slope, intercept, r_value, p_value, std_err = stats.linregress(age, price)

                # Taxa de depreciação anual
                depreciation_rate = -slope / intercept * 100 if intercept != 0 else 0

                self.depreciation_models_[cluster_id] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'depreciation_rate': abs(depreciation_rate),
                    'base_price': intercept
                }

        logger.info(f"Modelos de depreciação treinados para {len(self.depreciation_models_)} clusters")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica feature engineering baseada nos clusters

        Args:
            X: DataFrame com features

        Returns:
            DataFrame com features baseadas em clusters
        """
        if not self.fitted_:
            raise ValueError("ClusterBasedFeatureEngineer deve ser fitted antes da transformação")

        X_transformed = X.copy()

        # Calcular idade se necessário
        if 'Idade_Veiculo' not in X_transformed.columns and 'Ano' in X_transformed.columns:
            X_transformed['Idade_Veiculo'] = 2024 - X_transformed['Ano']

        # Predizer clusters
        cluster_labels = self._predict_clusters(X_transformed)
        X_transformed['Cluster'] = cluster_labels
        X_transformed['Cluster_Nome'] = X_transformed['Cluster'].map(self.cluster_names_)

        # Features baseadas em clusters
        X_transformed = self._create_cluster_features(X_transformed, cluster_labels)

        # Features de depreciação
        X_transformed = self._create_depreciation_features(X_transformed, cluster_labels)

        # Features de posicionamento no cluster
        X_transformed = self._create_positioning_features(X_transformed, cluster_labels)

        return X_transformed

    def _predict_clusters(self, X: pd.DataFrame) -> np.ndarray:
        """Prediz clusters para novos dados"""
        if not hasattr(self, 'prediction_features_') or not hasattr(self, 'prediction_kmeans_'):
            logger.warning("Modelo de predição não foi treinado. Usando cluster padrão.")
            return np.zeros(len(X))

        # Verificar se features de predição estão disponíveis
        missing_features = [f for f in self.prediction_features_ if f not in X.columns]
        if missing_features:
            logger.warning(f"Features de predição faltando: {missing_features}. Usando cluster padrão.")
            return np.zeros(len(X))

        # Usar apenas features de predição
        cluster_data = X[self.prediction_features_].fillna(X[self.prediction_features_].mean())

        # Usar scaler de predição
        cluster_data_scaled = self.prediction_scaler_.transform(cluster_data)

        # Usar modelo de predição (treinado com features de predição)
        pred_labels = self.prediction_kmeans_.predict(cluster_data_scaled)

        # Mapear labels para manter consistência com modelo principal
        if hasattr(self, 'prediction_label_mapping_'):
            mapped_labels = np.array([self.prediction_label_mapping_.get(label, label) for label in pred_labels])
            return mapped_labels
        else:
            return pred_labels

    def _create_cluster_features(self, X: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        """Cria features baseadas nas características dos clusters"""

        # Distância ao centroide do cluster
        if (hasattr(self, 'prediction_features_') and hasattr(self, 'prediction_scaler_') and
            hasattr(self, 'prediction_kmeans_')):
            # Usar features de predição
            available_features = [f for f in self.prediction_features_ if f in X.columns]
            if len(available_features) == len(self.prediction_features_):
                cluster_data = X[self.prediction_features_].fillna(X[self.prediction_features_].mean())
                cluster_data_scaled = self.prediction_scaler_.transform(cluster_data)

                distances_to_centroid = []
                for i, label in enumerate(cluster_labels):
                    # Usar centroide do modelo de predição para compatibilidade de dimensões
                    if hasattr(self, 'prediction_label_mapping_'):
                        # Encontrar cluster correspondente no modelo de predição
                        pred_cluster = None
                        for pred_id, main_id in self.prediction_label_mapping_.items():
                            if main_id == label:
                                pred_cluster = pred_id
                                break

                        if pred_cluster is not None:
                            centroid = self.prediction_kmeans_.cluster_centers_[pred_cluster]
                        else:
                            # Fallback: usar primeiro centroide disponível
                            centroid = self.prediction_kmeans_.cluster_centers_[0]
                    else:
                        # Se não há mapeamento, usar label diretamente
                        centroid = self.prediction_kmeans_.cluster_centers_[label]

                    distance = np.linalg.norm(cluster_data_scaled[i] - centroid)
                    distances_to_centroid.append(distance)

                X['Distancia_Centroide'] = distances_to_centroid

        # Features baseadas nas estatísticas do cluster
        for cluster_id in range(self.kmeans_.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.any():
                stats = self.cluster_stats_[cluster_id]

                # Features de comparação com a média do cluster
                if 'Idade_Veiculo' in X.columns:
                    age_mean = stats.get('Idade_Veiculo_mean', 0)
                    X.loc[cluster_mask, 'Idade_vs_Cluster'] = X.loc[cluster_mask, 'Idade_Veiculo'] - age_mean

                if 'Quilometragem' in X.columns:
                    km_mean = stats.get('Quilometragem_mean', 0)
                    X.loc[cluster_mask, 'Km_vs_Cluster'] = X.loc[cluster_mask, 'Quilometragem'] - km_mean

        # Features de tamanho e representatividade do cluster
        cluster_sizes = pd.Series(cluster_labels).value_counts()
        X['Cluster_Size'] = X['Cluster'].map(cluster_sizes)
        X['Cluster_Percentage'] = X['Cluster'].map(lambda x: self.cluster_stats_[x]['percentage'])

        return X

    def _create_depreciation_features(self, X: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        """Cria features baseadas nos modelos de depreciação"""

        # Depreciação estimada baseada no modelo do cluster
        depreciation_estimates = []
        depreciation_rates = []

        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id in self.depreciation_models_ and 'Idade_Veiculo' in X.columns:
                model = self.depreciation_models_[cluster_id]
                age = X.iloc[i]['Idade_Veiculo']

                # Valor estimado baseado no modelo de depreciação
                estimated_value = model['intercept'] + model['slope'] * age
                depreciation_estimates.append(max(0, estimated_value))
                depreciation_rates.append(model['depreciation_rate'])
            else:
                depreciation_estimates.append(0)
                depreciation_rates.append(0)

        X['Valor_Estimado_Depreciacao'] = depreciation_estimates
        X['Taxa_Depreciacao_Cluster'] = depreciation_rates

        # Valor residual (percentual do valor original estimado)
        if 'Valor_Estimado_Depreciacao' in X.columns:
            base_values = [self.depreciation_models_.get(cluster_id, {}).get('base_price', 1)
                          for cluster_id in cluster_labels]
            X['Valor_Residual_Pct'] = X['Valor_Estimado_Depreciacao'] / np.maximum(base_values, 1)

        return X

    def _create_positioning_features(self, X: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        """Cria features de posicionamento dentro do cluster"""

        # Ranking dentro do cluster (para features numéricas)
        numeric_cols = ['Idade_Veiculo', 'Quilometragem']

        for col in numeric_cols:
            if col in X.columns:
                rankings = []
                for cluster_id in range(self.kmeans_.n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    if cluster_mask.any():
                        cluster_data = X.loc[cluster_mask, col]
                        cluster_ranks = cluster_data.rank(pct=True)
                        rankings.extend(cluster_ranks.tolist())

                # Garantir que as ordenações correspondem aos índices originais
                rank_series = pd.Series(rankings, index=X.index)
                X[f'{col}_Rank_Cluster'] = rank_series

        # Features de outlier dentro do cluster
        for cluster_id in range(self.kmeans_.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.any():
                for col in numeric_cols:
                    if col in X.columns:
                        cluster_data = X.loc[cluster_mask, col]
                        if len(cluster_data) > 3:
                            # Z-score dentro do cluster
                            mean_val = cluster_data.mean()
                            std_val = cluster_data.std()
                            if std_val > 0:
                                z_scores = np.abs((cluster_data - mean_val) / std_val)
                                X.loc[cluster_mask, f'{col}_ZScore_Cluster'] = z_scores
                                X.loc[cluster_mask, f'{col}_Outlier_Cluster'] = (z_scores > 2).astype(int)

        return X

    def get_cluster_info(self) -> Dict[str, Any]:
        """Retorna informações detalhadas dos clusters"""
        return {
            'cluster_stats': self.cluster_stats_,
            'cluster_names': self.cluster_names_,
            'depreciation_models': self.depreciation_models_,
            'n_clusters': self.kmeans_.n_clusters if self.kmeans_ else 0
        }

    def plot_cluster_analysis(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Gera visualizações da análise de clusters"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if not self.fitted_:
                raise ValueError("Modelo deve ser fitted antes da visualização")

            # Preparar dados
            X_viz = X.copy()
            if 'Idade_Veiculo' not in X_viz.columns and 'Ano' in X_viz.columns:
                X_viz['Idade_Veiculo'] = 2024 - X_viz['Ano']

            cluster_labels = self._predict_clusters(X_viz)

            # Configurar plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Análise de Clusters do Mercado Automotivo', fontsize=16)

            # 1. Distribuição dos clusters
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            cluster_names = [self.cluster_names_.get(i, f'Cluster {i}') for i in cluster_counts.index]

            axes[0, 0].pie(cluster_counts.values, labels=cluster_names, autopct='%1.1f%%')
            axes[0, 0].set_title('Distribuição dos Clusters')

            # 2. Scatter plot principal
            if 'Idade_Veiculo' in X_viz.columns and 'Quilometragem' in X_viz.columns:
                scatter = axes[0, 1].scatter(X_viz['Idade_Veiculo'], X_viz['Quilometragem'],
                                           c=cluster_labels, cmap='tab10', alpha=0.6)
                axes[0, 1].set_xlabel('Idade do Veículo')
                axes[0, 1].set_ylabel('Quilometragem')
                axes[0, 1].set_title('Clusters: Idade vs Quilometragem')
                plt.colorbar(scatter, ax=axes[0, 1])

            # 3. Boxplot de preços por cluster (se target disponível)
            if y is not None:
                cluster_df = pd.DataFrame({'Cluster': cluster_labels, 'Preco': y})
                cluster_df['Cluster_Nome'] = cluster_df['Cluster'].map(self.cluster_names_)

                sns.boxplot(data=cluster_df, x='Cluster_Nome', y='Preco', ax=axes[1, 0])
                axes[1, 0].set_title('Distribuição de Preços por Cluster')
                axes[1, 0].tick_params(axis='x', rotation=45)

            # 4. Características dos clusters
            stats_data = []
            for cluster_id, stats in self.cluster_stats_.items():
                stats_data.append({
                    'Cluster': self.cluster_names_.get(cluster_id, f'Cluster {cluster_id}'),
                    'Tamanho': stats['size'],
                    'Idade_Média': stats.get('Idade_Veiculo_mean', 0),
                    'Preço_Médio': stats.get('target_mean', 0)
                })

            stats_df = pd.DataFrame(stats_data)
            if not stats_df.empty:
                axes[1, 1].barh(stats_df['Cluster'], stats_df['Preço_Médio'])
                axes[1, 1].set_title('Preço Médio por Cluster')
                axes[1, 1].set_xlabel('Preço Médio (R$)')

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib/Seaborn não disponível para visualizações")

    def save(self, filepath: str) -> None:
        """Salva o modelo treinado"""
        joblib.dump(self, filepath)
        logger.info(f"ClusterBasedFeatureEngineer salvo em: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ClusterBasedFeatureEngineer':
        """Carrega modelo salvo"""
        model = joblib.load(filepath)
        logger.info(f"ClusterBasedFeatureEngineer carregado de: {filepath}")
        return model


class DepreciationPatternAnalyzer(BaseEstimator, TransformerMixin):
    """
    Analisador de padrões de depreciação específicos do mercado brasileiro
    """

    def __init__(self, min_observations: int = 20):
        self.min_observations = min_observations
        self.fitted_ = False
        self.brand_depreciation_ = {}
        self.model_depreciation_ = {}
        self.market_depreciation_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DepreciationPatternAnalyzer':
        """Analisa padrões de depreciação por marca e modelo"""
        logger.info("Analisando padrões de depreciação...")

        X_work = X.copy()
        if 'Idade_Veiculo' not in X_work.columns and 'Ano' in X_work.columns:
            X_work['Idade_Veiculo'] = 2024 - X_work['Ano']

        # Depreciação geral do mercado
        self._analyze_market_depreciation(X_work, y)

        # Depreciação por marca
        self._analyze_brand_depreciation(X_work, y)

        # Depreciação por modelo (top modelos)
        self._analyze_model_depreciation(X_work, y)

        self.fitted_ = True
        logger.info("Análise de depreciação concluída")
        return self

    def _analyze_market_depreciation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Analisa depreciação geral do mercado"""
        if 'Idade_Veiculo' in X.columns and len(X) > 1:
            # Verificar se há variação na idade
            if X['Idade_Veiculo'].std() > 0:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    X['Idade_Veiculo'], y
                )

                # Debug
                logger.debug(f"Market depreciation: slope={slope}, intercept={intercept}, mean_price={y.mean()}")

                # Calcular taxa de depreciação anual
                # Se slope é negativo (preço diminui com idade), taxa é positiva
                if not np.isnan(slope) and slope != 0:
                    # Taxa de depreciação como % do preço médio por ano
                    depreciation_rate = abs(slope / y.mean() * 100) if y.mean() > 0 else 0

                    # Alternativa: usar diferença entre veículo novo (idade 0) e usado (idade 10)
                    price_age_0 = intercept
                    price_age_10 = intercept + (slope * 10)
                    if price_age_0 > 0:
                        alt_depreciation_rate = abs((price_age_0 - price_age_10) / price_age_0 * 100 / 10)
                        # Usar a maior das duas como medida mais conservadora
                        depreciation_rate = max(depreciation_rate, alt_depreciation_rate)
                else:
                    depreciation_rate = 0

                self.market_depreciation_ = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'annual_depreciation_rate': depreciation_rate
                }

                logger.info(f"Market depreciation rate: {depreciation_rate:.1f}% per year")
            else:
                # Se não há variação na idade, usar valores padrão
                self.market_depreciation_ = {
                    'slope': 0,
                    'intercept': y.mean(),
                    'r_squared': 0,
                    'p_value': 1.0,
                    'annual_depreciation_rate': 0
                }

    def _analyze_brand_depreciation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Analisa depreciação por marca"""
        if 'Marca' in X.columns and 'Idade_Veiculo' in X.columns:
            for brand in X['Marca'].unique():
                brand_data = X[X['Marca'] == brand]
                brand_prices = y[brand_data.index]

                if len(brand_data) >= self.min_observations:
                    # Verificar se há variação na idade para esta marca
                    if brand_data['Idade_Veiculo'].std() > 0:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            brand_data['Idade_Veiculo'], brand_prices
                        )

                        # Calcular taxa de depreciação para a marca
                        if not np.isnan(slope) and slope != 0:
                            # Taxa baseada na média de preços
                            depreciation_rate = abs(slope / brand_prices.mean() * 100) if brand_prices.mean() > 0 else 0

                            # Taxa alternativa baseada em diferença de 10 anos
                            price_age_0 = intercept
                            price_age_10 = intercept + (slope * 10)
                            if price_age_0 > 0:
                                alt_rate = abs((price_age_0 - price_age_10) / price_age_0 * 100 / 10)
                                depreciation_rate = max(depreciation_rate, alt_rate)
                        else:
                            depreciation_rate = 0

                        self.brand_depreciation_[brand] = {
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'annual_depreciation_rate': depreciation_rate,
                            'observations': len(brand_data)
                        }
                    else:
                        # Se não há variação na idade, usar valores padrão
                        self.brand_depreciation_[brand] = {
                            'slope': 0,
                            'intercept': brand_prices.mean(),
                            'r_squared': 0,
                            'p_value': 1.0,
                            'annual_depreciation_rate': 0,
                            'observations': len(brand_data)
                        }

    def _analyze_model_depreciation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Analisa depreciação por modelo"""
        if 'Modelo' in X.columns and 'Idade_Veiculo' in X.columns:
            # Analisar apenas modelos com observações suficientes
            model_counts = X['Modelo'].value_counts()
            top_models = model_counts[model_counts >= self.min_observations].index

            for model in top_models:
                model_data = X[X['Modelo'] == model]
                model_prices = y[model_data.index]

                # Verificar se há variação na idade para este modelo
                if model_data['Idade_Veiculo'].std() > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        model_data['Idade_Veiculo'], model_prices
                    )

                    # Calcular taxa de depreciação para o modelo
                    if not np.isnan(slope) and slope != 0:
                        # Taxa baseada na média de preços
                        depreciation_rate = abs(slope / model_prices.mean() * 100) if model_prices.mean() > 0 else 0

                        # Taxa alternativa baseada em diferença de 10 anos
                        price_age_0 = intercept
                        price_age_10 = intercept + (slope * 10)
                        if price_age_0 > 0:
                            alt_rate = abs((price_age_0 - price_age_10) / price_age_0 * 100 / 10)
                            depreciation_rate = max(depreciation_rate, alt_rate)
                    else:
                        depreciation_rate = 0

                    self.model_depreciation_[model] = {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'annual_depreciation_rate': depreciation_rate,
                        'observations': len(model_data)
                    }
                else:
                    # Se não há variação na idade, usar valores padrão
                    self.model_depreciation_[model] = {
                        'slope': 0,
                        'intercept': model_prices.mean(),
                        'r_squared': 0,
                        'p_value': 1.0,
                        'annual_depreciation_rate': 0,
                        'observations': len(model_data)
                    }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cria features baseadas nos padrões de depreciação"""
        if not self.fitted_:
            raise ValueError("DepreciationPatternAnalyzer deve ser fitted antes da transformação")

        X_transformed = X.copy()

        if 'Idade_Veiculo' not in X_transformed.columns and 'Ano' in X_transformed.columns:
            X_transformed['Idade_Veiculo'] = 2024 - X_transformed['Ano']

        # Features de depreciação por marca
        if 'Marca' in X_transformed.columns:
            X_transformed['Marca_Taxa_Depreciacao'] = X_transformed['Marca'].map(
                lambda x: self.brand_depreciation_.get(x, {}).get('annual_depreciation_rate', 0)
            )
            X_transformed['Marca_R2_Depreciacao'] = X_transformed['Marca'].map(
                lambda x: self.brand_depreciation_.get(x, {}).get('r_squared', 0)
            )

        # Features de depreciação por modelo
        if 'Modelo' in X_transformed.columns:
            X_transformed['Modelo_Taxa_Depreciacao'] = X_transformed['Modelo'].map(
                lambda x: self.model_depreciation_.get(x, {}).get('annual_depreciation_rate', 0)
            )

        # Comparação com depreciação do mercado
        if self.market_depreciation_ and 'Idade_Veiculo' in X_transformed.columns:
            market_rate = self.market_depreciation_['annual_depreciation_rate']

            if 'Marca_Taxa_Depreciacao' in X_transformed.columns:
                X_transformed['Depreciacao_vs_Mercado'] = (
                    X_transformed['Marca_Taxa_Depreciacao'] - market_rate
                )

        # Valor estimado baseado na depreciação
        if 'Marca' in X_transformed.columns and 'Idade_Veiculo' in X_transformed.columns:
            estimated_values = []
            for _, row in X_transformed.iterrows():
                brand = row['Marca']
                age = row['Idade_Veiculo']

                if brand in self.brand_depreciation_:
                    model = self.brand_depreciation_[brand]
                    estimated_value = model['intercept'] + model['slope'] * age
                    estimated_values.append(max(0, estimated_value))
                else:
                    estimated_values.append(0)

            X_transformed['Valor_Estimado_Depreciacao'] = estimated_values

        return X_transformed

    def get_depreciation_report(self) -> Dict[str, Any]:
        """Gera relatório detalhado de depreciação"""
        # Ordenar marcas por taxa de depreciação
        brand_rates = {brand: info['annual_depreciation_rate']
                      for brand, info in self.brand_depreciation_.items()}
        sorted_brands = sorted(brand_rates.items(), key=lambda x: x[1])

        return {
            'market_depreciation': self.market_depreciation_,
            'best_retention_brands': sorted_brands[:5],  # Menor depreciação
            'worst_retention_brands': sorted_brands[-5:],  # Maior depreciação
            'brand_analysis': self.brand_depreciation_,
            'model_analysis': self.model_depreciation_,
            'total_brands_analyzed': len(self.brand_depreciation_),
            'total_models_analyzed': len(self.model_depreciation_)
        }


# Função utilitária para feature engineering completa
def create_advanced_features(X: pd.DataFrame, y: Optional[pd.Series] = None,
                           cluster_config: Optional[ClusterConfig] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cria features avançadas baseadas em clustering e depreciação

    Args:
        X: DataFrame com features originais
        y: Target para análise de depreciação
        cluster_config: Configuração do clustering

    Returns:
        Tupla com (DataFrame com features, relatório de análise)
    """
    logger.info("Iniciando feature engineering avançada...")

    # Feature engineering baseada em clusters
    cluster_engineer = ClusterBasedFeatureEngineer(config=cluster_config)
    cluster_engineer.fit(X, y)
    X_clustered = cluster_engineer.transform(X)

    # Análise de depreciação
    depreciation_analyzer = DepreciationPatternAnalyzer()
    if y is not None:
        depreciation_analyzer.fit(X, y)
        X_final = depreciation_analyzer.transform(X_clustered)
    else:
        X_final = X_clustered

    # Relatório
    report = {
        'original_features': X.shape[1],
        'final_features': X_final.shape[1],
        'features_created': X_final.shape[1] - X.shape[1],
        'cluster_info': cluster_engineer.get_cluster_info(),
        'depreciation_info': depreciation_analyzer.get_depreciation_report() if y is not None else None,
        'timestamp': datetime.now().isoformat()
    }

    logger.info(f"Feature engineering concluída. {report['features_created']} features criadas.")

    return X_final, report


if __name__ == "__main__":
    # Teste do módulo
    print("Testando ClusterBasedFeatureEngineer...")

    # Dados de exemplo
    np.random.seed(42)
    n_samples = 1000

    X_test = pd.DataFrame({
        'Ano': np.random.randint(2000, 2024, n_samples),
        'Quilometragem': np.random.exponential(80000, n_samples),
        'Marca': np.random.choice(['Toyota', 'Honda', 'Ford', 'Volkswagen'], n_samples),
        'Modelo': np.random.choice(['Civic', 'Corolla', 'Focus', 'Golf'], n_samples),
        'Portas': np.random.choice([2, 4], n_samples)
    })

    y_test = pd.Series(np.random.exponential(70000, n_samples))

    try:
        # Teste completo
        X_featured, report = create_advanced_features(X_test, y_test)

        print(f"Features originais: {report['original_features']}")
        print(f"Features finais: {report['final_features']}")
        print(f"Features criadas: {report['features_created']}")
        print(f"Clusters identificados: {report['cluster_info']['n_clusters']}")

        print("Teste concluído com sucesso!")

    except Exception as e:
        logger.error(f"Erro no teste: {e}")
        raise