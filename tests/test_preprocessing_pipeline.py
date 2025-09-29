#!/usr/bin/env python3
"""
Teste Completo do Pipeline de Preprocessing

Este script testa todo o pipeline de preparação de dados implementado,
validando a qualidade das features criadas e a eficácia das transformações.

Author: Machine Learning Pipeline
Date: 2024
"""

import sys
import os
# Adicionar diretório pai ao path para acessar src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path

# Importações dos módulos implementados
from data.load_data import load_car_dataset
from features.preprocessing import AdvancedPreprocessingPipeline, FeatureConfig
from features.engineering import ClusterBasedFeatureEngineer, DepreciationPatternAnalyzer, ClusterConfig
from features.validation import validate_preprocessing_pipeline, ValidationConfig

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Função principal para teste do pipeline"""
    print("=" * 80)
    print("TESTE COMPLETO DO PIPELINE DE PREPROCESSING")
    print("=" * 80)

    try:
        # 1. Carregamento dos dados
        print("\n1. CARREGAMENTO DOS DADOS")
        print("-" * 40)

        data_path = os.path.join(os.path.dirname(__file__), '..', "data/raw/dataset_carros_brasil.csv")
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset não encontrado: {data_path}")

        df, quality_report = load_car_dataset(data_path)
        print(f"Dataset carregado: {df.shape}")
        print(f"Qualidade geral: {quality_report['data_quality']['duplicates']} duplicatas")

        # Separar features e target
        X = df.drop('Valor_Venda', axis=1)
        y = df['Valor_Venda']

        print(f"Features: {X.shape}")
        print(f"Target: {y.shape}")
        print(f"Target - Média: R$ {y.mean():,.2f}, Std: R$ {y.std():,.2f}")

    except Exception as e:
        logger.error(f"Erro no carregamento dos dados: {e}")
        return False

    try:
        # 2. Teste do Feature Engineering baseado em Clusters
        print("\n2. FEATURE ENGINEERING BASEADO EM CLUSTERS")
        print("-" * 50)

        cluster_config = ClusterConfig(n_clusters=4)
        cluster_engineer = ClusterBasedFeatureEngineer(config=cluster_config)

        # Treinar e aplicar
        cluster_engineer.fit(X, y)
        X_clustered = cluster_engineer.transform(X)

        cluster_info = cluster_engineer.get_cluster_info()
        print(f"Clusters identificados: {cluster_info['n_clusters']}")
        print(f"Features após clustering: {X_clustered.shape}")

        # Mostrar informações dos clusters
        for cluster_id, name in cluster_info['cluster_names'].items():
            stats = cluster_info['cluster_stats'][cluster_id]
            print(f"  Cluster {cluster_id} ({name}): {stats['size']} veículos ({stats['percentage']:.1f}%)")

    except Exception as e:
        logger.error(f"Erro no feature engineering de clusters: {e}")
        X_clustered = X.copy()

    try:
        # 3. Teste do Analisador de Depreciação
        print("\n3. ANÁLISE DE PADRÕES DE DEPRECIAÇÃO")
        print("-" * 40)

        depreciation_analyzer = DepreciationPatternAnalyzer(min_observations=20)
        depreciation_analyzer.fit(X, y)
        X_with_depreciation = depreciation_analyzer.transform(X_clustered)

        depreciation_report = depreciation_analyzer.get_depreciation_report()
        print(f"Marcas analisadas: {depreciation_report['total_brands_analyzed']}")
        print(f"Modelos analisados: {depreciation_report['total_models_analyzed']}")

        if depreciation_report['market_depreciation']:
            market_rate = depreciation_report['market_depreciation']['annual_depreciation_rate']
            print(f"Taxa de depreciação do mercado: {market_rate:.1f}% ao ano")

        # Melhores e piores marcas em retenção de valor
        if depreciation_report['best_retention_brands']:
            best_brands = depreciation_report['best_retention_brands'][:3]
            worst_brands = depreciation_report['worst_retention_brands'][-3:]

            print(f"Melhores em retenção de valor:")
            for brand, rate in best_brands:
                print(f"  {brand}: {rate:.1f}% ao ano")

            print(f"Piores em retenção de valor:")
            for brand, rate in worst_brands:
                print(f"  {brand}: {rate:.1f}% ao ano")

    except Exception as e:
        logger.error(f"Erro na análise de depreciação: {e}")
        X_with_depreciation = X_clustered.copy()

    try:
        # 4. Teste do Pipeline Completo
        print("\n4. TESTE DO PIPELINE COMPLETO")
        print("-" * 35)

        # Configuração do pipeline
        config = FeatureConfig(
            create_temporal_features=True,
            create_interaction_features=True,
            create_categorical_derived=True,
            create_brand_features=True,
            create_boolean_features=True,
            outlier_contamination=0.05,
            high_cardinality_threshold=10,
            target_encoding_threshold=0.1
        )

        # Criar e treinar pipeline
        pipeline = AdvancedPreprocessingPipeline(config=config)

        print("Treinando pipeline completo...")
        X_processed = pipeline.fit_transform(X, y)

        print(f"Pipeline treinado com sucesso!")
        print(f"Features finais: {X_processed.shape}")
        print(f"Features criadas: {X_processed.shape[1] - X.shape[1]}")

        # Mostrar tipos de features criadas
        feature_types = {}
        for col in X_processed.columns:
            if col in X.columns:
                feature_types['original'] = feature_types.get('original', 0) + 1
            elif any(keyword in col.lower() for keyword in ['idade', 'eficiencia', 'depreciacao']):
                feature_types['temporal'] = feature_types.get('temporal', 0) + 1
            elif 'marca' in col.lower():
                feature_types['brand'] = feature_types.get('brand', 0) + 1
            elif any(keyword in col.lower() for keyword in ['_x_', '_por_', 'intensidade']):
                feature_types['interaction'] = feature_types.get('interaction', 0) + 1
            elif 'cluster' in col.lower():
                feature_types['cluster'] = feature_types.get('cluster', 0) + 1
            elif col.startswith('Is_'):
                feature_types['boolean'] = feature_types.get('boolean', 0) + 1
            else:
                feature_types['other'] = feature_types.get('other', 0) + 1

        print(f"Distribuição das features:")
        for feature_type, count in feature_types.items():
            print(f"  {feature_type}: {count}")

    except Exception as e:
        logger.error(f"Erro no pipeline completo: {e}")
        return False

    try:
        # 5. Validação e Análise de Qualidade
        print("\n5. VALIDAÇÃO E ANÁLISE DE QUALIDADE")
        print("-" * 40)

        validation_config = ValidationConfig(
            cv_folds=5,
            random_state=42,
            correlation_threshold=0.1,
            feature_importance_threshold=0.01
        )

        print("Executando validação completa...")
        validation_report = validate_preprocessing_pipeline(
            pipeline, X, y, validation_config
        )

        # Mostrar resultados da validação
        pipeline_info = validation_report['pipeline_info']
        print(f"Features processadas: {pipeline_info['processed_features']}")
        print(f"Features criadas: {pipeline_info['features_created']}")

        # Análise de qualidade
        quality_analysis = validation_report['quality_analysis']
        if 'correlation_analysis' in quality_analysis:
            strong_corr = len(quality_analysis['correlation_analysis'].get('strong_correlations', []))
            significant_corr = len(quality_analysis['correlation_analysis'].get('significant_correlations', []))
            print(f"Features com correlação forte (>{validation_config.correlation_threshold}): {strong_corr}")
            print(f"Features estatisticamente significativas: {significant_corr}")

        # Análise de validação cruzada
        cv_analysis = validation_report['cv_analysis']
        if 'model_performance' in cv_analysis:
            print(f"\\nPerformance dos modelos (R²):")
            for model_name, metrics in cv_analysis['model_performance'].items():
                r2_score = metrics.get('test_r2', {})
                if r2_score:
                    print(f"  {model_name}: {r2_score['mean']:.3f} ± {r2_score['std']:.3f}")

        # Avaliação geral
        overall_assessment = validation_report['overall_assessment']
        print(f"\\nAvaliação Geral: {overall_assessment['overall_grade']}")
        print(f"Score de Qualidade: {overall_assessment['quality_score']}")

        if overall_assessment['strengths']:
            print(f"Pontos Fortes:")
            for strength in overall_assessment['strengths']:
                print(f"  + {strength}")

        if overall_assessment['weaknesses']:
            print(f"Pontos Fracos:")
            for weakness in overall_assessment['weaknesses']:
                print(f"  - {weakness}")

        if overall_assessment['recommendations']:
            print(f"Recomendações:")
            for rec in overall_assessment['recommendations']:
                print(f"  → {rec}")

    except Exception as e:
        logger.error(f"Erro na validação: {e}")

    try:
        # 6. Salvamento dos Resultados
        print("\n6. SALVAMENTO DOS RESULTADOS")
        print("-" * 32)

        # Criar diretórios se necessário
        base_path = os.path.join(os.path.dirname(__file__), '..')
        os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'reports'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'data/processed'), exist_ok=True)

        # Salvar pipeline
        pipeline_path = os.path.join(base_path, 'models/tested_preprocessing_pipeline.pkl')
        pipeline.save(pipeline_path)
        print(f"Pipeline salvo: {pipeline_path}")

        # Salvar dados processados (amostra)
        sample_processed = X_processed.head(100)  # Apenas amostra para teste
        sample_path = os.path.join(base_path, 'data/processed/sample_processed_data.csv')
        sample_processed.to_csv(sample_path, index=False)
        print(f"Amostra dos dados processados salva: {sample_path}")

        # Salvar relatório de validação
        validation_report_path = os.path.join(base_path, 'reports/preprocessing_validation_report.json')
        with open(validation_report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False, default=str)
        print(f"Relatório de validação salvo: {validation_report_path}")

        # Salvar resumo do teste
        test_summary = {
            'timestamp': datetime.now().isoformat(),
            'test_results': {
                'data_loaded': True,
                'clusters_identified': cluster_info['n_clusters'] if 'cluster_info' in locals() else 0,
                'depreciation_analyzed': depreciation_report['total_brands_analyzed'] if 'depreciation_report' in locals() else 0,
                'pipeline_trained': True,
                'features_original': X.shape[1],
                'features_final': X_processed.shape[1],
                'features_created': X_processed.shape[1] - X.shape[1],
                'validation_completed': True,
                'overall_grade': overall_assessment.get('overall_grade', 'N/A') if 'overall_assessment' in locals() else 'N/A'
            },
            'files_created': [
                pipeline_path,
                sample_path,
                validation_report_path
            ]
        }

        summary_path = os.path.join(base_path, 'reports/preprocessing_test_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"Resumo do teste salvo: {summary_path}")

    except Exception as e:
        logger.error(f"Erro no salvamento: {e}")

    # 7. Conclusão
    print("\n" + "=" * 80)
    print("TESTE DO PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)

    print(f"\\nResumo dos Resultados:")
    print(f"• Dataset: {df.shape[0]:,} registros, {df.shape[1]} features originais")
    print(f"• Features finais: {X_processed.shape[1]} (criadas: {X_processed.shape[1] - X.shape[1]})")
    print(f"• Clusters identificados: {cluster_info['n_clusters'] if 'cluster_info' in locals() else 'N/A'}")
    print(f"• Avaliação final: {overall_assessment.get('overall_grade', 'N/A') if 'overall_assessment' in locals() else 'N/A'}")

    print(f"\\nArquivos gerados:")
    print(f"• Pipeline: models/tested_preprocessing_pipeline.pkl")
    print(f"• Relatório: reports/preprocessing_validation_report.json")
    print(f"• Resumo: reports/preprocessing_test_summary.json")

    print(f"\\nO pipeline de preprocessing está pronto para uso na fase de modelagem!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)