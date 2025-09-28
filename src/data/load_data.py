"""
Data Loading Module for Car Price Prediction Project

Este módulo implementa uma classe DataLoader robusta para carregamento,
validação e análise inicial de dados do mercado automotivo brasileiro.

Author: Machine Learning Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataSchema:
    """Esquema esperado para o dataset de carros brasileiros"""
    required_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    target_column: str
    expected_dtypes: Dict[str, str]
    valid_ranges: Dict[str, Tuple[float, float]]
    valid_categories: Dict[str, List[str]]


class DataQualityMetrics:
    """Métricas de qualidade dos dados"""

    def __init__(self):
        self.metrics = {}

    def calculate_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula a completude dos dados por coluna"""
        completeness = {}
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            completeness[column] = (non_null_count / len(df)) * 100
        return completeness

    def calculate_uniqueness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula a unicidade dos dados por coluna"""
        uniqueness = {}
        for column in df.columns:
            unique_count = df[column].nunique()
            uniqueness[column] = (unique_count / len(df)) * 100
        return uniqueness

    def detect_outliers_iqr(self, series: pd.Series, factor: float = 1.5) -> Dict[str, Any]:
        """Detecta outliers usando método IQR"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outliers_mask = (series < lower_bound) | (series > upper_bound)
        outliers_count = outliers_mask.sum()

        return {
            'count': outliers_count,
            'percentage': (outliers_count / len(series)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_values': series[outliers_mask].tolist()
        }


class DataLoader:
    """
    Classe principal para carregamento e validação de dados do mercado automotivo brasileiro.

    Esta classe implementa funcionalidades robustas para:
    - Carregamento seguro de dados
    - Validação de schema e integridade
    - Detecção de anomalias
    - Geração de relatórios de qualidade
    - Cache inteligente
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o DataLoader

        Args:
            config_path: Caminho para arquivo de configuração (opcional)
        """
        self.config_path = config_path
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Schema do dataset de carros brasileiros
        self.schema = DataSchema(
            required_columns=[
                'Marca', 'Modelo', 'Ano', 'Quilometragem', 'Cor',
                'Cambio', 'Combustivel', 'Portas', 'Valor_Venda'
            ],
            numeric_columns=['Ano', 'Quilometragem', 'Portas', 'Valor_Venda'],
            categorical_columns=['Marca', 'Modelo', 'Cor', 'Cambio', 'Combustivel'],
            target_column='Valor_Venda',
            expected_dtypes={
                'Marca': 'object',
                'Modelo': 'object',
                'Ano': 'int64',
                'Quilometragem': 'float64',
                'Cor': 'object',
                'Cambio': 'object',
                'Combustivel': 'object',
                'Portas': 'int64',
                'Valor_Venda': 'float64'
            },
            valid_ranges={
                'Ano': (1990, 2024),
                'Quilometragem': (0, 500000),
                'Portas': (2, 4),
                'Valor_Venda': (5000, 500000)
            },
            valid_categories={
                'Cambio': ['Manual', 'Automático'],
                'Combustivel': ['Flex', 'Gasolina', 'Diesel'],
                'Portas': [2, 4]
            }
        )

        self.quality_metrics = DataQualityMetrics()
        logger.info("DataLoader inicializado com sucesso")

    def _generate_file_hash(self, file_path: str) -> str:
        """Gera hash MD5 do arquivo para controle de cache"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Erro ao gerar hash do arquivo: {e}")
            return None

    def _detect_encoding(self, file_path: str) -> str:
        """Detecta o encoding do arquivo automaticamente"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Tenta ler uma pequena parte
                logger.info(f"Encoding detectado: {encoding}")
                return encoding
            except UnicodeDecodeError:
                continue

        logger.warning("Não foi possível detectar encoding, usando utf-8 como fallback")
        return 'utf-8'

    def load_dataset(self, file_path: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Carrega o dataset com validações e cache inteligente

        Args:
            file_path: Caminho para o arquivo CSV
            use_cache: Se deve usar cache (default: True)

        Returns:
            pd.DataFrame: Dataset carregado e validado

        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            ValueError: Se o dataset não atender aos critérios de qualidade
        """
        logger.info(f"Iniciando carregamento do dataset: {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        # Verificar cache
        file_hash = self._generate_file_hash(file_path)
        cache_file = self.cache_dir / f"dataset_{file_hash}.pkl"

        if use_cache and cache_file.exists():
            logger.info("Dataset encontrado no cache, carregando...")
            try:
                df = pd.read_pickle(cache_file)
                logger.info(f"Dataset carregado do cache: {df.shape}")
                return df
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}")

        # Detectar encoding e carregar
        encoding = self._detect_encoding(file_path)

        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"Dataset carregado: {df.shape}")
        except Exception as e:
            logger.error(f"Erro ao carregar dataset: {e}")
            raise

        # Validar schema
        if not self.validate_schema(df):
            raise ValueError("Dataset não atende ao schema esperado")

        # Aplicar transformações básicas
        df = self._apply_basic_transformations(df)

        # Salvar no cache
        if use_cache:
            try:
                df.to_pickle(cache_file)
                logger.info("Dataset salvo no cache")
            except Exception as e:
                logger.warning(f"Erro ao salvar cache: {e}")

        logger.info("Dataset carregado e validado com sucesso")
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Valida se o DataFrame atende ao schema esperado

        Args:
            df: DataFrame a ser validado

        Returns:
            bool: True se válido, False caso contrário
        """
        logger.info("Validando schema do dataset...")

        # Verificar colunas obrigatórias
        missing_columns = set(self.schema.required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Colunas obrigatórias ausentes: {missing_columns}")
            return False

        # Verificar tipos de dados
        for column, expected_dtype in self.schema.expected_dtypes.items():
            if column in df.columns:
                current_dtype = str(df[column].dtype)
                if not current_dtype.startswith(expected_dtype.split('64')[0]):  # Flexibilidade para int32/int64
                    logger.warning(f"Tipo de dados inesperado em {column}: {current_dtype} (esperado: {expected_dtype})")

        # Verificar ranges válidos
        for column, (min_val, max_val) in self.schema.valid_ranges.items():
            if column in df.columns and df[column].dtype in ['int64', 'float64']:
                if df[column].min() < min_val or df[column].max() > max_val:
                    logger.warning(f"Valores fora do range esperado em {column}: [{df[column].min()}, {df[column].max()}] (esperado: [{min_val}, {max_val}])")

        logger.info("Validação de schema concluída")
        return True

    def _apply_basic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformações básicas no dataset"""
        df_transformed = df.copy()

        # Converter tipos de dados
        for column in self.schema.numeric_columns:
            if column in df_transformed.columns:
                df_transformed[column] = pd.to_numeric(df_transformed[column], errors='coerce')

        # Limpar espaços em strings
        for column in self.schema.categorical_columns:
            if column in df_transformed.columns:
                df_transformed[column] = df_transformed[column].astype(str).str.strip()

        logger.info("Transformações básicas aplicadas")
        return df_transformed

    def generate_data_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera relatório completo de qualidade dos dados

        Args:
            df: DataFrame a ser analisado

        Returns:
            Dict: Relatório detalhado de qualidade
        """
        logger.info("Gerando relatório de qualidade dos dados...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'columns': list(df.columns)
            },
            'data_quality': {
                'completeness': self.quality_metrics.calculate_completeness(df),
                'uniqueness': self.quality_metrics.calculate_uniqueness(df),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': df.duplicated().sum()
            },
            'statistical_summary': {},
            'outliers_analysis': {},
            'categorical_analysis': {}
        }

        # Análise estatística para variáveis numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            report['statistical_summary'][column] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'skewness': df[column].skew(),
                'kurtosis': df[column].kurtosis()
            }

            # Análise de outliers
            outliers_info = self.quality_metrics.detect_outliers_iqr(df[column])
            report['outliers_analysis'][column] = outliers_info

        # Análise de variáveis categóricas
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            value_counts = df[column].value_counts()
            report['categorical_analysis'][column] = {
                'unique_values': df[column].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'distribution': value_counts.head(10).to_dict()
            }

        logger.info("Relatório de qualidade gerado com sucesso")
        return report

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detecta anomalias específicas do mercado automotivo

        Args:
            df: DataFrame a ser analisado

        Returns:
            Dict: Dicionário com tipos de anomalias detectadas
        """
        logger.info("Detectando anomalias específicas do mercado automotivo...")

        anomalies = {
            'temporal_anomalies': [],
            'business_logic_anomalies': [],
            'statistical_anomalies': [],
            'data_quality_issues': []
        }

        # Anomalias temporais
        current_year = datetime.now().year
        if 'Ano' in df.columns:
            future_cars = df[df['Ano'] > current_year]
            if len(future_cars) > 0:
                anomalies['temporal_anomalies'].append(f"{len(future_cars)} carros com ano futuro detectados")

        # Anomalias de lógica de negócio
        if 'Quilometragem' in df.columns and 'Ano' in df.columns:
            # Carros novos com alta quilometragem
            new_cars_high_km = df[(current_year - df['Ano'] <= 1) & (df['Quilometragem'] > 50000)]
            if len(new_cars_high_km) > 0:
                anomalies['business_logic_anomalies'].append(f"{len(new_cars_high_km)} carros novos com alta quilometragem")

            # Carros muito antigos com baixa quilometragem
            old_cars_low_km = df[(current_year - df['Ano'] > 15) & (df['Quilometragem'] < 50000)]
            if len(old_cars_low_km) > 0:
                anomalies['business_logic_anomalies'].append(f"{len(old_cars_low_km)} carros antigos com baixa quilometragem")

        # Preços inconsistentes
        if 'Valor_Venda' in df.columns:
            very_cheap_cars = df[df['Valor_Venda'] < 10000]
            very_expensive_cars = df[df['Valor_Venda'] > 300000]

            if len(very_cheap_cars) > 0:
                anomalies['business_logic_anomalies'].append(f"{len(very_cheap_cars)} carros com preços muito baixos (<R$10k)")

            if len(very_expensive_cars) > 0:
                anomalies['business_logic_anomalies'].append(f"{len(very_expensive_cars)} carros com preços muito altos (>R$300k)")

        # Inconsistências de dados
        for column in df.columns:
            if df[column].dtype == 'object':
                # Verificar valores em branco ou apenas espaços
                blank_values = df[df[column].str.strip() == ''].shape[0] if df[column].dtype == 'object' else 0
                if blank_values > 0:
                    anomalies['data_quality_issues'].append(f"{blank_values} valores em branco em {column}")

        logger.info(f"Detecção de anomalias concluída: {sum(len(v) for v in anomalies.values())} anomalias encontradas")
        return anomalies

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Salva o relatório de qualidade em arquivo JSON

        Args:
            report: Relatório a ser salvo
            output_path: Caminho do arquivo de saída
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Relatório salvo em: {output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")
            raise


# Função utilitária para uso rápido
def load_car_dataset(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Função utilitária para carregamento rápido do dataset com relatório

    Args:
        file_path: Caminho para o arquivo CSV

    Returns:
        Tuple: (DataFrame carregado, Relatório de qualidade)
    """
    loader = DataLoader()
    df = loader.load_dataset(file_path)
    report = loader.generate_data_report(df)
    anomalies = loader.detect_anomalies(df)

    # Adicionar anomalias ao relatório
    report['anomalies'] = anomalies

    return df, report


if __name__ == "__main__":
    # Exemplo de uso
    file_path = "data/raw/dataset_carros_brasil.csv"

    try:
        df, report = load_car_dataset(file_path)
        print(f"Dataset carregado: {df.shape}")
        print(f"Qualidade geral: {len(report['anomalies'])} tipos de anomalias detectadas")

        # Salvar relatório
        loader = DataLoader()
        loader.save_report(report, "reports/data_quality_report.json")

    except Exception as e:
        logger.error(f"Erro na execução: {e}")