# Predição de Preços de Veículos no Mercado Brasileiro

## Aviso Educacional

Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.

## Resumo Executivo

Este projeto implementa um pipeline completo de Machine Learning para predição de valores de venda de veículos automotores no mercado brasileiro. O desenvolvimento segue rigorosamente a metodologia CRISP-DM (Cross-Industry Standard Process for Data Mining), abrangendo desde a análise exploratória dos dados até o deployment de um sistema de predição em produção.

O objetivo principal é desenvolver modelos preditivos capazes de estimar com precisão o valor de venda de veículos com base em suas características técnicas e de mercado, utilizando técnicas avançadas de ciência de dados e práticas de MLOps para garantir reprodutibilidade e rastreabilidade dos experimentos.

## Fundamentação Teórica

O projeto baseia-se na metodologia CRISP-DM, um framework padrão da indústria para projetos de mineração de dados que organiza o processo em seis fases distintas:

1. **Business Understanding**: Compreensão do domínio automotivo brasileiro
2. **Data Understanding**: Análise exploratória e caracterização do dataset
3. **Data Preparation**: Preprocessamento e engenharia de features
4. **Modeling**: Desenvolvimento e treinamento de algoritmos de ML
5. **Evaluation**: Avaliação comparativa dos modelos
6. **Deployment**: Implementação em ambiente de produção

## Dataset

### Características Técnicas

- **Tamanho**: 10.000 observações
- **Formato**: CSV (Comma-Separated Values)
- **Arquivo**: `dataset_carros_brasil.csv`

### Variáveis do Sistema

| Variável        | Tipo       | Descrição                               | Domínio                           |
| --------------- | ---------- | --------------------------------------- | --------------------------------- |
| `Marca`         | Categórica | Fabricante do veículo                   | Ford, Hyundai, Jeep, Toyota, etc. |
| `Modelo`        | Categórica | Denominação do modelo                   | EcoSport, HB20S, Renegade, etc.   |
| `Ano`           | Numérica   | Ano de fabricação                       | 2000-2021                         |
| `Quilometragem` | Numérica   | Distância percorrida (km)               | Contínua                          |
| `Cor`           | Categórica | Coloração do veículo                    | Azul, Prata, Preto, etc.          |
| `Cambio`        | Categórica | Sistema de transmissão                  | Manual, Automático                |
| `Combustivel`   | Categórica | Tipo de combustível                     | Flex, Gasolina, Diesel            |
| `Portas`        | Numérica   | Quantidade de portas                    | 2, 4                              |
| `Valor_Venda`   | Numérica   | **Variável alvo** - Preço de venda (R$) | Contínua                          |

## Arquitetura do Projeto

```
projeto-carros-brasil/
├── data/
│   ├── raw/                    # Dados brutos originais
│   ├── processed/              # Dados processados
│   └── external/               # Dados externos auxiliares
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data/                   # Scripts de processamento
│   ├── features/               # Engenharia de features
│   ├── models/                 # Algoritmos de ML
│   └── visualization/          # Funções de plotagem
├── models/                     # Modelos treinados serializados
├── reports/
│   ├── figures/                # Visualizações geradas
│   └── metrics/                # Relatórios de performance
├── deployment/                 # Arquivos para deploy
├── mlruns/                     # Experimentos MLflow
├── requirements.txt            # Dependências Python
├── README.md                   # Documentação principal
└── CLAUDE.md                   # Orientações para desenvolvimento
```

## Pré-requisitos Técnicos

### Ambiente de Desenvolvimento

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git (controle de versão)
- Jupyter Notebook ou JupyterLab

### Dependências Principais

- pandas: Manipulação de dados estruturados
- numpy: Computação numérica
- scikit-learn: Algoritmos de machine learning
- matplotlib/seaborn: Visualização de dados
- mlflow: Rastreamento de experimentos
- streamlit: Interface web para deploy

## Instalação e Configuração

### 1. Clonagem do Repositório

```bash
git clone https://github.com/seu-usuario/projeto-carros-brasil.git
cd projeto-carros-brasil
```

### 2. Ambiente Virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 4. Configuração MLflow

```bash
mlflow ui
```

## Execução dos Experimentos

### Pipeline Completo

```bash
# 1. Análise exploratória
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# 2. Preparação dos dados
python src/data/prepare_data.py

# 3. Treinamento dos modelos
python src/models/train_models.py

# 4. Avaliação comparativa
python src/models/evaluate_models.py

# 5. Geração de relatórios
python src/reports/generate_report.py
```

### Execução Individual

```bash
# Apenas preprocessamento
python src/data/preprocess.py

# Treinamento de modelo específico
python src/models/train_model.py --algorithm random_forest

# Predição em novos dados
python src/models/predict.py --input data/new_cars.csv
```

## Metodologia CRISP-DM Aplicada

### 1. Business Understanding

Análise do mercado automotivo brasileiro, identificação de fatores determinantes no preço de veículos usados e definição de métricas de sucesso para o projeto.

### 2. Data Understanding

Caracterização estatística do dataset, identificação de padrões, detecção de valores ausentes e outliers, análise de correlações entre variáveis.

### 3. Data Preparation

- Tratamento de valores ausentes
- Codificação de variáveis categóricas
- Normalização/padronização de features numéricas
- Engenharia de features (idade do veículo, depreciação)
- Divisão treino/validação/teste

### 4. Modeling

Implementação e treinamento de múltiplos algoritmos:

- Regressão Linear
- Random Forest Regressor
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Regression
- Redes Neurais (MLPRegressor)

### 5. Evaluation

Avaliação quantitativa utilizando métricas padrão:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (Coeficiente de Determinação)
- Mean Absolute Percentage Error (MAPE)

### 6. Deployment

Implementação de interface web responsiva para predições em tempo real, com documentação de API e monitoramento de performance.

## MLOps e Rastreamento de Experimentos

### MLflow Integration

Todos os experimentos são automaticamente registrados com:

- Hiperparâmetros utilizados
- Métricas de performance
- Artefatos do modelo
- Código fonte versionado
- Ambiente computacional

### Reprodutibilidade

```bash
# Executar experimento específico
mlflow run . -P algorithm=random_forest -P n_estimators=100

# Comparar experimentos
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Deploy em Produção

### Plataformas Suportadas

- Hugging Face Spaces
- Render
- Railway
- Deta Space
- Heroku

### Configuração de Deploy

```bash
# Streamlit local
streamlit run deployment/app.py

# Docker containerization
docker build -t car-price-predictor .
docker run -p 8501:8501 car-price-predictor
```

## Resultados Esperados

O sistema final deve ser capaz de:

- Predizer preços com erro médio inferior a 15%
- Processar predições em tempo real (<200ms)
- Explicar fatores determinantes no preço
- Fornecer intervalos de confiança nas predições

## Estrutura de Código

### Padrões de Desenvolvimento

- PEP 8 (Style Guide for Python Code)
- Docstrings seguindo padrão NumPy
- Type hints para maior robustez
- Testes unitários com pytest
- Logging estruturado

### Organização Modular

```python
# Exemplo de estrutura de classe
class CarPricePredictor:
    def __init__(self, model_type: str):
        self.model = self._load_model(model_type)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features)

    def predict_with_confidence(self, features: pd.DataFrame) -> tuple:
        predictions = self.predict(features)
        confidence_intervals = self._calculate_confidence(features)
        return predictions, confidence_intervals
```

## Contribuição ao Projeto

### Diretrizes para Colaboradores

1. Fork do repositório principal
2. Criação de branch específica para feature
3. Implementação com testes unitários
4. Documentação das modificações
5. Pull request com descrição detalhada

### Padrões de Commit

```
feat: implementa novo algoritmo de ensemble
fix: corrige bug na validação cruzada
docs: atualiza documentação da API
test: adiciona testes para preprocessamento
```

## Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo LICENSE para detalhes.

## Referências Acadêmicas

1. Chapman, P., et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.
