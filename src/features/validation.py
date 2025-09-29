"""
Feature Validation and Cross-Validation Module

Este módulo implementa validação cruzada e análise de qualidade para features criadas
no processo de preparação de dados do mercado automotivo brasileiro.

Author: Machine Learning Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit, cross_validate
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import stats

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuração para validação de features"""
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    scoring_metrics: List[str] = None
    feature_importance_threshold: float = 0.01
    correlation_threshold: float = 0.1
    mutual_info_threshold: float = 0.1

    def __post_init__(self):
        if self.scoring_metrics is None:
            self.scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']


class FeatureQualityAnalyzer:
    """
    Analisador de qualidade de features com métricas estatísticas avançadas
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.quality_report_ = {}

    def analyze_feature_quality(self, X: pd.DataFrame, y: pd.Series,
                              feature_groups: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Análise completa da qualidade das features

        Args:
            X: DataFrame com features
            y: Target
            feature_groups: Grupos de features para análise separada

        Returns:
            Relatório detalhado de qualidade
        """
        logger.info("Iniciando análise de qualidade das features...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'target_mean': y.mean(),
                'target_std': y.std()
            },
            'correlation_analysis': self._analyze_correlations(X, y),
            'mutual_information': self._analyze_mutual_information(X, y),
            'statistical_tests': self._perform_statistical_tests(X, y),
            'feature_stability': self._analyze_feature_stability(X, y),
            'redundancy_analysis': self._analyze_redundancy(X),
            'feature_importance': self._analyze_feature_importance(X, y)
        }

        # Análise por grupos se fornecido
        if feature_groups:
            report['group_analysis'] = self._analyze_feature_groups(X, y, feature_groups)

        # Recomendações
        report['recommendations'] = self._generate_recommendations(report)

        self.quality_report_ = report
        logger.info("Análise de qualidade concluída")

        return report

    def _analyze_correlations(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Análise de correlações com múltiplos métodos"""
        correlations = {}

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in X.columns and not X[col].isna().all():
                try:
                    # Pearson
                    pearson_corr, pearson_p = pearsonr(X[col].fillna(0), y)

                    # Spearman
                    spearman_corr, spearman_p = spearmanr(X[col].fillna(0), y)

                    # Kendall
                    kendall_corr, kendall_p = kendalltau(X[col].fillna(0), y)

                    correlations[col] = {
                        'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
                        'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
                        'kendall': {'correlation': kendall_corr, 'p_value': kendall_p},
                        'abs_pearson': abs(pearson_corr),
                        'significant': min(pearson_p, spearman_p, kendall_p) < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Erro ao calcular correlação para {col}: {e}")

        # Ranking por correlação absoluta
        if correlations:
            sorted_correlations = sorted(
                correlations.items(),
                key=lambda x: x[1]['abs_pearson'],
                reverse=True
            )

            return {
                'individual_correlations': correlations,
                'top_correlations': sorted_correlations[:20],
                'strong_correlations': [
                    item for item in sorted_correlations
                    if item[1]['abs_pearson'] > self.config.correlation_threshold
                ],
                'significant_correlations': [
                    item for item in sorted_correlations
                    if item[1]['significant']
                ]
            }

        return {}

    def _analyze_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Análise de informação mútua"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            X_numeric = X[numeric_cols].fillna(X[numeric_cols].mean())

            try:
                mi_scores = mutual_info_regression(X_numeric, y, random_state=self.config.random_state)

                mi_results = pd.DataFrame({
                    'feature': numeric_cols,
                    'mutual_info': mi_scores
                }).sort_values('mutual_info', ascending=False)

                return {
                    'mi_scores': mi_results.to_dict('records'),
                    'high_mi_features': mi_results[
                        mi_results['mutual_info'] > self.config.mutual_info_threshold
                    ].to_dict('records'),
                    'mi_mean': mi_scores.mean(),
                    'mi_std': mi_scores.std()
                }
            except Exception as e:
                logger.warning(f"Erro no cálculo de mutual information: {e}")

        return {}

    def _perform_statistical_tests(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Testes estatísticos para validação de features"""
        tests_results = {}

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in X.columns and not X[col].isna().all():
                try:
                    feature_data = X[col].fillna(X[col].mean())

                    # Teste de normalidade
                    shapiro_stat, shapiro_p = stats.shapiro(
                        feature_data.sample(min(5000, len(feature_data)))
                    )

                    # Teste F para regressão
                    f_stat, f_p = f_regression(feature_data.values.reshape(-1, 1), y)

                    tests_results[col] = {
                        'normality_test': {
                            'shapiro_stat': shapiro_stat,
                            'shapiro_p': shapiro_p,
                            'is_normal': shapiro_p > 0.05
                        },
                        'f_regression': {
                            'f_stat': f_stat[0],
                            'f_p': f_p[0],
                            'significant': f_p[0] < 0.05
                        }
                    }
                except Exception as e:
                    logger.warning(f"Erro em testes estatísticos para {col}: {e}")

        return tests_results

    def _analyze_feature_stability(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Análise de estabilidade das features através de validação cruzada"""
        stability_results = {}

        # Usar uma amostra menor se o dataset for muito grande
        if len(X) > 5000:
            sample_idx = np.random.choice(len(X), 5000, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y

        numeric_cols = X_sample.select_dtypes(include=[np.number]).columns

        # Validação cruzada para cada feature individualmente
        cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)

        for col in numeric_cols[:20]:  # Limitar a 20 features para eficiência
            if col in X_sample.columns and not X_sample[col].isna().all():
                try:
                    X_single = X_sample[[col]].fillna(X_sample[[col]].mean())

                    # Modelo simples para teste
                    model = Ridge(alpha=1.0)

                    # Validação cruzada
                    cv_scores = cross_val_score(
                        model, X_single, y_sample,
                        cv=cv, scoring='r2'
                    )

                    stability_results[col] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'cv_min': cv_scores.min(),
                        'cv_max': cv_scores.max(),
                        'stability_score': 1 - (cv_scores.std() / (abs(cv_scores.mean()) + 1e-8))
                    }

                except Exception as e:
                    logger.warning(f"Erro na análise de estabilidade para {col}: {e}")

        return stability_results

    def _analyze_redundancy(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Análise de redundância entre features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {}

        # Matriz de correlação entre features
        X_numeric = X[numeric_cols].fillna(X[numeric_cols].mean())
        correlation_matrix = X_numeric.corr()

        # Encontrar features altamente correlacionadas
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.8:  # Threshold para alta correlação
                    high_corr_pairs.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i, j]
                    })

        # Identificar features candidatas para remoção
        features_to_remove = set()
        for pair in high_corr_pairs:
            # Remover a feature com nome "menos informativo" (heurística simples)
            if len(pair['feature_1']) > len(pair['feature_2']):
                features_to_remove.add(pair['feature_1'])
            else:
                features_to_remove.add(pair['feature_2'])

        return {
            'correlation_matrix_shape': correlation_matrix.shape,
            'high_correlation_pairs': high_corr_pairs,
            'redundant_features': list(features_to_remove),
            'avg_correlation': correlation_matrix.abs().mean().mean()
        }

    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Análise de importância usando múltiplos métodos"""
        importance_results = {}

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {}

        X_numeric = X[numeric_cols].fillna(X[numeric_cols].mean())

        try:
            # Random Forest Importance
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            rf_model.fit(X_numeric, y)

            rf_importance = pd.DataFrame({
                'feature': numeric_cols,
                'rf_importance': rf_model.feature_importances_
            }).sort_values('rf_importance', ascending=False)

            # Recursive Feature Elimination
            rfe_model = RFE(
                estimator=LinearRegression(),
                n_features_to_select=min(10, len(numeric_cols))
            )
            rfe_model.fit(X_numeric, y)

            rfe_ranking = pd.DataFrame({
                'feature': numeric_cols,
                'rfe_ranking': rfe_model.ranking_,
                'rfe_selected': rfe_model.support_
            }).sort_values('rfe_ranking')

            importance_results = {
                'rf_importance': rf_importance.to_dict('records'),
                'rfe_ranking': rfe_ranking.to_dict('records'),
                'top_rf_features': rf_importance.head(10).to_dict('records'),
                'rfe_selected_features': rfe_ranking[rfe_ranking['rfe_selected']]['feature'].tolist()
            }

        except Exception as e:
            logger.warning(f"Erro na análise de importância: {e}")

        return importance_results

    def _analyze_feature_groups(self, X: pd.DataFrame, y: pd.Series,
                              feature_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Análise de qualidade por grupos de features"""
        group_analysis = {}

        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in X.columns]

            if available_features:
                X_group = X[available_features]

                # Análise básica do grupo
                group_stats = {
                    'n_features': len(available_features),
                    'missing_features': [f for f in features if f not in X.columns]
                }

                # Correlação média com target
                numeric_features = X_group.select_dtypes(include=[np.number]).columns
                if len(numeric_features) > 0:
                    correlations = []
                    for col in numeric_features:
                        if not X_group[col].isna().all():
                            try:
                                corr = abs(pearsonr(X_group[col].fillna(0), y)[0])
                                if not np.isnan(corr):
                                    correlations.append(corr)
                            except:
                                pass

                    if correlations:
                        group_stats.update({
                            'avg_correlation': np.mean(correlations),
                            'max_correlation': np.max(correlations),
                            'min_correlation': np.min(correlations)
                        })

                group_analysis[group_name] = group_stats

        return group_analysis

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []

        # Recomendações sobre correlações
        if 'correlation_analysis' in report:
            strong_corr = report['correlation_analysis'].get('strong_correlations', [])
            if len(strong_corr) == 0:
                recommendations.append(
                    "ATENÇÃO: Nenhuma feature apresenta correlação forte com o target. "
                    "Considere feature engineering adicional."
                )
            elif len(strong_corr) < 5:
                recommendations.append(
                    f"Apenas {len(strong_corr)} features apresentam correlação forte. "
                    "Pode ser necessário criar features adicionais."
                )

        # Recomendações sobre redundância
        if 'redundancy_analysis' in report:
            redundant = report['redundancy_analysis'].get('redundant_features', [])
            if redundant:
                recommendations.append(
                    f"Encontradas {len(redundant)} features redundantes que podem ser removidas: "
                    f"{', '.join(redundant[:5])}{'...' if len(redundant) > 5 else ''}"
                )

        # Recomendações sobre estabilidade
        if 'feature_stability' in report:
            unstable_features = [
                feat for feat, stats in report['feature_stability'].items()
                if stats.get('stability_score', 0) < 0.5
            ]
            if unstable_features:
                recommendations.append(
                    f"Features com baixa estabilidade detectadas: {len(unstable_features)}. "
                    "Considere regularização ou remoção."
                )

        if not recommendations:
            recommendations.append("Features apresentam qualidade adequada para modelagem.")

        return recommendations


class CrossValidationAnalyzer:
    """
    Analisador especializado em validação cruzada para features
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def comprehensive_cv_analysis(self, X: pd.DataFrame, y: pd.Series,
                                models: Optional[List[BaseEstimator]] = None) -> Dict[str, Any]:
        """
        Análise abrangente usando validação cruzada

        Args:
            X: Features
            y: Target
            models: Lista de modelos para teste (opcional)

        Returns:
            Relatório de validação cruzada
        """
        logger.info("Iniciando análise de validação cruzada...")

        if models is None:
            models = [
                LinearRegression(),
                Ridge(alpha=1.0),
                RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            ]

        # Preparar dados
        X_clean = self._prepare_data_for_cv(X)

        results = {
            'model_performance': {},
            'feature_importance_cv': {},
            'stability_analysis': {},
            'learning_curves': {}
        }

        # Validação cruzada para cada modelo
        for model in models:
            model_name = model.__class__.__name__
            logger.info(f"Testando modelo: {model_name}")

            # Performance do modelo
            cv_results = self._evaluate_model_cv(model, X_clean, y)
            results['model_performance'][model_name] = cv_results

            # Importância das features com CV
            if hasattr(model, 'feature_importances_'):
                importance_cv = self._feature_importance_cv(model, X_clean, y)
                results['feature_importance_cv'][model_name] = importance_cv

        # Análise de estabilidade das features
        results['stability_analysis'] = self._stability_analysis_cv(X_clean, y)

        # Análise de curvas de aprendizado
        results['learning_curves'] = self._learning_curves_analysis(
            models[0], X_clean, y  # Usar primeiro modelo
        )

        logger.info("Análise de validação cruzada concluída")
        return results

    def _prepare_data_for_cv(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados para validação cruzada"""
        # Selecionar apenas colunas numéricas
        X_numeric = X.select_dtypes(include=[np.number])

        # Preencher valores faltantes
        X_clean = X_numeric.fillna(X_numeric.mean())

        # Remover colunas com variância zero
        constant_cols = X_clean.columns[X_clean.var() == 0]
        if len(constant_cols) > 0:
            logger.info(f"Removendo colunas com variância zero: {list(constant_cols)}")
            X_clean = X_clean.drop(columns=constant_cols)

        return X_clean

    def _evaluate_model_cv(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Avalia modelo usando validação cruzada"""
        cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)

        # Métricas múltiplas
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

        try:
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )

            return {
                'test_mse': {
                    'mean': -cv_results['test_neg_mean_squared_error'].mean(),
                    'std': cv_results['test_neg_mean_squared_error'].std()
                },
                'test_mae': {
                    'mean': -cv_results['test_neg_mean_absolute_error'].mean(),
                    'std': cv_results['test_neg_mean_absolute_error'].std()
                },
                'test_r2': {
                    'mean': cv_results['test_r2'].mean(),
                    'std': cv_results['test_r2'].std()
                },
                'train_r2': {
                    'mean': cv_results['train_r2'].mean(),
                    'std': cv_results['train_r2'].std()
                },
                'overfitting_score': cv_results['train_r2'].mean() - cv_results['test_r2'].mean()
            }

        except Exception as e:
            logger.error(f"Erro na validação cruzada: {e}")
            return {}

    def _feature_importance_cv(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Análise de importância com validação cruzada"""
        cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)

        importance_scores = []

        for train_idx, val_idx in cv.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)

            if hasattr(model_copy, 'feature_importances_'):
                importance_scores.append(model_copy.feature_importances_)

        if importance_scores:
            importance_array = np.array(importance_scores)

            importance_results = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': importance_array.mean(axis=0),
                'importance_std': importance_array.std(axis=0),
                'importance_min': importance_array.min(axis=0),
                'importance_max': importance_array.max(axis=0)
            }).sort_values('importance_mean', ascending=False)

            return {
                'importance_stats': importance_results.to_dict('records'),
                'stable_important_features': importance_results[
                    (importance_results['importance_mean'] > self.config.feature_importance_threshold) &
                    (importance_results['importance_std'] < importance_results['importance_mean'] * 0.5)
                ]['feature'].tolist()
            }

        return {}

    def _stability_analysis_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Análise de estabilidade das predictions"""
        cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)

        model = Ridge(alpha=1.0)
        predictions = []

        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)

            predictions.extend(val_pred)

        # Análise da estabilidade das predições
        pred_array = np.array(predictions)

        return {
            'prediction_stability': {
                'mean': pred_array.mean(),
                'std': pred_array.std(),
                'coefficient_of_variation': pred_array.std() / (pred_array.mean() + 1e-8)
            },
            'prediction_range': {
                'min': pred_array.min(),
                'max': pred_array.max(),
                'range': pred_array.max() - pred_array.min()
            }
        }

    def _learning_curves_analysis(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Análise de curvas de aprendizado"""
        from sklearn.model_selection import learning_curve

        try:
            train_sizes = np.linspace(0.1, 1.0, 10)

            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=3,  # Reduzir para eficiência
                scoring='r2',
                n_jobs=-1,
                random_state=self.config.random_state
            )

            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist(),
                'final_gap': train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
            }

        except Exception as e:
            logger.warning(f"Erro na análise de curvas de aprendizado: {e}")
            return {}


def validate_preprocessing_pipeline(pipeline, X: pd.DataFrame, y: pd.Series,
                                  config: Optional[ValidationConfig] = None) -> Dict[str, Any]:
    """
    Validação completa de pipeline de preprocessing

    Args:
        pipeline: Pipeline de preprocessing treinado
        X: Features originais
        y: Target
        config: Configuração de validação

    Returns:
        Relatório completo de validação
    """
    logger.info("Iniciando validação completa do pipeline...")

    config = config or ValidationConfig()

    # Aplicar pipeline
    X_processed = pipeline.transform(X)

    # Feature groups baseados nos tipos de features criadas
    feature_groups = {
        'temporal': [col for col in X_processed.columns if any(keyword in col.lower()
                    for keyword in ['idade', 'ano', 'eficiencia', 'depreciacao'])],
        'brand': [col for col in X_processed.columns if 'marca' in col.lower()],
        'interaction': [col for col in X_processed.columns if '_x_' in col.lower() or '_por_' in col.lower()],
        'categorical': [col for col in X_processed.columns if any(keyword in col.lower()
                       for keyword in ['faixa', 'is_', 'categoria'])],
        'cluster': [col for col in X_processed.columns if 'cluster' in col.lower()],
        'outlier': [col for col in X_processed.columns if 'outlier' in col.lower()]
    }

    # Remover grupos vazios
    feature_groups = {k: v for k, v in feature_groups.items() if v}

    # Análise de qualidade
    quality_analyzer = FeatureQualityAnalyzer(config)
    quality_report = quality_analyzer.analyze_feature_quality(X_processed, y, feature_groups)

    # Análise de validação cruzada
    cv_analyzer = CrossValidationAnalyzer(config)
    cv_report = cv_analyzer.comprehensive_cv_analysis(X_processed, y)

    # Relatório consolidado
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_info': {
            'original_features': X.shape[1],
            'processed_features': X_processed.shape[1],
            'features_created': X_processed.shape[1] - X.shape[1],
            'feature_groups': {k: len(v) for k, v in feature_groups.items()}
        },
        'quality_analysis': quality_report,
        'cv_analysis': cv_report,
        'overall_assessment': _generate_overall_assessment(quality_report, cv_report)
    }

    logger.info("Validação completa concluída")
    return final_report


def _generate_overall_assessment(quality_report: Dict[str, Any], cv_report: Dict[str, Any]) -> Dict[str, Any]:
    """Gera avaliação geral da qualidade do preprocessing"""
    assessment = {
        'quality_score': 0,
        'strengths': [],
        'weaknesses': [],
        'recommendations': []
    }

    # Avaliar qualidade das correlações
    if 'correlation_analysis' in quality_report:
        strong_corr = len(quality_report['correlation_analysis'].get('strong_correlations', []))
        if strong_corr > 10:
            assessment['strengths'].append("Muitas features com correlação forte")
            assessment['quality_score'] += 2
        elif strong_corr > 5:
            assessment['strengths'].append("Correlações adequadas")
            assessment['quality_score'] += 1
        else:
            assessment['weaknesses'].append("Poucas features com correlação forte")

    # Avaliar performance dos modelos
    if 'model_performance' in cv_report:
        best_r2 = 0
        for model_name, metrics in cv_report['model_performance'].items():
            r2_score = metrics.get('test_r2', {}).get('mean', 0)
            best_r2 = max(best_r2, r2_score)

        if best_r2 > 0.7:
            assessment['strengths'].append("Excelente performance preditiva")
            assessment['quality_score'] += 3
        elif best_r2 > 0.5:
            assessment['strengths'].append("Boa performance preditiva")
            assessment['quality_score'] += 2
        elif best_r2 > 0.3:
            assessment['strengths'].append("Performance moderada")
            assessment['quality_score'] += 1
        else:
            assessment['weaknesses'].append("Performance preditiva baixa")

    # Avaliar redundância
    if 'redundancy_analysis' in quality_report:
        redundant = len(quality_report['redundancy_analysis'].get('redundant_features', []))
        if redundant == 0:
            assessment['strengths'].append("Sem features redundantes")
            assessment['quality_score'] += 1
        elif redundant > 10:
            assessment['weaknesses'].append("Muitas features redundantes")
            assessment['recommendations'].append("Remover features redundantes")

    # Classificação final
    if assessment['quality_score'] >= 5:
        assessment['overall_grade'] = 'Excelente'
    elif assessment['quality_score'] >= 3:
        assessment['overall_grade'] = 'Bom'
    elif assessment['quality_score'] >= 2:
        assessment['overall_grade'] = 'Adequado'
    else:
        assessment['overall_grade'] = 'Requer melhorias'

    return assessment


if __name__ == "__main__":
    # Teste do módulo
    print("Testando FeatureQualityAnalyzer...")

    # Dados de exemplo
    np.random.seed(42)
    n_samples = 1000

    X_test = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.exponential(1, n_samples),
        'feature_3': np.random.choice([0, 1], n_samples),
        'feature_4': np.random.normal(0, 1, n_samples),
    })

    # Target correlacionado
    y_test = pd.Series(X_test['feature_1'] * 2 + X_test['feature_2'] + np.random.normal(0, 0.5, n_samples))

    try:
        # Teste de análise de qualidade
        analyzer = FeatureQualityAnalyzer()
        report = analyzer.analyze_feature_quality(X_test, y_test)

        print(f"Features analisadas: {len(X_test.columns)}")
        print(f"Correlações fortes: {len(report['correlation_analysis']['strong_correlations'])}")
        print(f"Features significativas: {len(report['correlation_analysis']['significant_correlations'])}")

        # Teste de validação cruzada
        cv_analyzer = CrossValidationAnalyzer()
        cv_report = cv_analyzer.comprehensive_cv_analysis(X_test, y_test)

        print(f"Modelos testados: {len(cv_report['model_performance'])}")

        print("Teste concluído com sucesso!")

    except Exception as e:
        logger.error(f"Erro no teste: {e}")
        raise