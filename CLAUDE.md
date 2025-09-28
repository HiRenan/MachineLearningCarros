# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete Machine Learning pipeline project for predicting car sale values in Brazil. The project follows the CRISP-DM methodology and includes:

- Exploratory Data Analysis (EDA) on Brazilian car market data
- Data preparation and transformation
- Model training and comparison
- Experiment tracking (MLflow)
- Model deployment with web interface
- Required educational disclaimer: "Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais."

## Dataset

**File**: `dataset_carros_brasil.csv`

**Target Variable**: `Valor_Venda` (sale price)

**Features**:

- `Marca` - car manufacturer
- `Modelo` - car model
- `Ano` - manufacturing year
- `Quilometragem` - mileage/distance traveled
- `Cor` - car color
- `Cambio` - transmission type (manual/automatic)
- `Combustivel` - fuel type (flex/gasoline/diesel)
- `Portas` - number of doors

## Project Requirements

**Must Include**:

- Public repository (GitHub/GitLab) with organized source code
- README.md explaining the project, how to run, and how to use the model
- requirements.txt with dependencies
- Working deployment link on free platform (Hugging Face Spaces, Render, Railway, Deta Space)
- Web page must contain the educational disclaimer text

**Evaluation Criteria**:

- Quality of data analysis and preparation
- Clear model training and comparison
- Basic MLOps usage (experiment tracking)
- Working deployment
- Organization and documentation clarity

## Development Approach

Follow CRISP-DM methodology:

1. Business Understanding
2. Data Understanding (EDA)
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

The project should demonstrate innovation beyond basic examples and serve as a professional showcase.

## Algoritmos de Machine Learning

**Implementar e comparar os seguintes algoritmos**:

- **Regressão Linear**: Baseline e interpretabilidade
- **Random Forest**: Robustez e feature importance
- **XGBoost**: Performance e otimização avançada
- **LightGBM**: Eficiência computacional
- **Support Vector Regression**: Robustez a outliers

## Estrutura Técnica do Projeto

```
DesafioFinalML/
├── data/
│   ├── raw/                    # dataset_carros_brasil.csv
│   ├── processed/              # Dados limpos e transformados
│   └── external/               # Dados auxiliares (se necessário)
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment_preparation.ipynb
├── src/
│   ├── data/
│   │   ├── load_data.py        # Carregamento de dados
│   │   └── preprocess.py       # Preprocessamento
│   ├── features/
│   │   └── engineering.py      # Feature engineering
│   ├── models/
│   │   ├── train.py            # Treinamento de modelos
│   │   └── predict.py          # Predições
│   └── utils/
│       └── helpers.py          # Funções auxiliares
├── models/
│   └── trained_models/         # Modelos serializados
├── reports/
│   ├── figures/                # Gráficos e visualizações
│   └── metrics/                # Métricas de performance
├── deployment/
│   ├── app.py                  # Aplicação Streamlit
│   ├── model_handler.py        # Carregamento de modelos
│   └── utils.py                # Utilitários para deploy
└── mlruns/                     # Experimentos MLflow
```

## Comandos de Execução

### Setup Inicial
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Iniciar MLflow UI
mlflow ui --port 5000
```

### Execução do Pipeline
```bash
# 1. Executar notebooks na sequência
jupyter notebook notebooks/01_business_understanding.ipynb
jupyter notebook notebooks/02_data_understanding.ipynb
jupyter notebook notebooks/03_data_preparation.ipynb
jupyter notebook notebooks/04_modeling.ipynb
jupyter notebook notebooks/05_evaluation.ipynb

# 2. Treinamento via scripts
python src/models/train.py

# 3. Deploy local
streamlit run deployment/app.py
```

### Métricas de Avaliação

**Métricas obrigatórias**:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coeficiente de Determinação)
- MAPE (Mean Absolute Percentage Error)

## MLflow Integration

**Tracking obrigatório**:
- Hiperparâmetros de cada modelo
- Métricas de performance
- Tempo de treinamento
- Versionamento de dados
- Artefatos do modelo (pickle/joblib)

## Deploy Requirements

**Hugging Face Spaces**:
- Interface Streamlit responsiva
- Formulário intuitivo para entrada de dados
- Visualizações das predições
- Disclaimer educacional visível
- Performance < 200ms por predição

## Feature Engineering

**Features derivadas obrigatórias**:
- Idade do veículo (2024 - Ano)
- Depreciação estimada
- Quilometragem por ano
- Categorização de faixas de preço
- Encoding de variáveis categóricas

## Qualidade de Código

**Padrões obrigatórios**:
- PEP 8 compliance
- Docstrings em funções principais
- Type hints nas funções críticas
- Tratamento de exceções
- Logging estruturado

##Faça tudo sempre bem profissional, sem emojis
