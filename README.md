# Predição de Preços de Veículos no Mercado Brasileiro

## Aviso Educacional

Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.

## Resumo Executivo

Este projeto implementa um pipeline completo de Machine Learning para predição de valores de venda de veículos automotores no mercado brasileiro, consolidado em um único notebook Jupyter que segue a metodologia CRISP-DM.

O modelo campeão (Lasso Regression) alcançou **R² de 98,03%**, com erro médio absoluto de apenas R$ 2.527, demonstrando alta precisão na predição de preços de veículos usados.

## Dataset

### Características Técnicas

- **Tamanho**: 10.000 observações (9.603 após limpeza)
- **Formato**: CSV (Comma-Separated Values)
- **Arquivo**: `dataset_carros_brasil.csv`

### Variáveis do Sistema

| Variável        | Tipo       | Descrição                               | Domínio                           |
| --------------- | ---------- | --------------------------------------- | --------------------------------- |
| `Marca`         | Categórica | Fabricante do veículo                   | Ford, Hyundai, Jeep, Toyota, etc. |
| `Modelo`        | Categórica | Denominação do modelo                   | EcoSport, HB20S, Renegade, etc.   |
| `Ano`           | Numérica   | Ano de fabricação                       | 2000-2023                         |
| `Quilometragem` | Numérica   | Distância percorrida (km)               | Contínua                          |
| `Cor`           | Categórica | Coloração do veículo                    | Azul, Prata, Preto, etc.          |
| `Cambio`        | Categórica | Sistema de transmissão                  | Manual, Automático                |
| `Combustivel`   | Categórica | Tipo de combustível                     | Flex, Gasolina, Diesel            |
| `Portas`        | Numérica   | Quantidade de portas                    | 2, 4                              |
| `Valor_Venda`   | Numérica   | **Variável alvo** - Preço de venda (R$) | Contínua                          |

## Arquitetura do Projeto

```
DesafioFinalML/
├── data/
│   ├── raw/
│   │   └── dataset_carros_brasil.csv    # Dataset original
│   └── processed/
│       ├── dataset_limpo.csv            # Dados processados (9.603 registros)
│       └── dataset_limpo.pkl            # Dados em formato pickle
├── notebooks/
│   └── modelo_carros.ipynb              # Notebook consolidado CRISP-DM
├── models/
│   └── trained_models/
│       ├── lasso_modelo_campeao.pkl     # Modelo campeão treinado
│       └── feature_info.json            # Informações das features
├── reports/
│   ├── figures/
│   │   └── importancia_features_*.png   # Gráficos de importância (6 modelos)
│   └── metrics/
│       ├── resultados_modelos.csv       # Métricas em CSV
│       └── resultados_modelos.json      # Métricas em JSON
├── src/                                 # Estrutura para desenvolvimento futuro
├── requirements.txt                     # Dependências Python
├── runtime.txt                          # Versão Python
└── README.md                            # Esta documentação
```

## Pré-requisitos Técnicos

### Ambiente de Desenvolvimento

- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)
- Git (controle de versão)
- Jupyter Notebook ou JupyterLab

### Dependências Principais

- pandas: Manipulação de dados estruturados
- numpy: Computação numérica
- scikit-learn: Algoritmos de machine learning
- matplotlib/seaborn: Visualização de dados
- xgboost: Gradient boosting otimizado
- lightgbm: Gradient boosting eficiente
- joblib: Serialização de modelos

## Instalação e Configuração

### 1. Clonagem do Repositório

```bash
git clone https://github.com/HiRenan/MachineLearningCarros.git
cd MachineLearningCarros
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

## Execução do Projeto

### Notebook Consolidado (Recomendado)

O projeto completo está consolidado em um único notebook que executa todas as fases CRISP-DM:

```bash
jupyter notebook notebooks/modelo_carros.ipynb
```

O notebook executa automaticamente:

1. Carregamento e análise exploratória dos dados
2. Limpeza e tratamento de outliers
3. Feature engineering
4. Treinamento de 6 modelos de ML
5. Otimização de hiperparâmetros com GridSearchCV
6. Geração de gráficos de importância de features
7. Salvamento de métricas e modelos

### Estrutura do Notebook

O notebook `modelo_carros.ipynb` está organizado nas seguintes seções:

1. **Imports e Upload**: Carregamento de bibliotecas e dataset
2. **Análise Exploratória**: Inspeção inicial dos dados
3. **Limpeza de Dados**: Remoção de nulos e erros de medição
4. **Tratamento de Outliers**: Remoção por IQR agrupado por marca
5. **Feature Engineering**: Criação de features derivadas
6. **Salvamento de Dados Processados**: Exportação para CSV e pickle
7. **Treinamento de Modelos**: Comparação de 8 algoritmos
8. **Otimização de Hiperparâmetros**: GridSearchCV em 6 modelos
9. **Geração de Gráficos**: Importância de features
10. **Salvamento de Métricas**: Exportação de resultados
11. **Salvamento do Modelo Campeão**: Serialização do Lasso

## Resultados Alcançados

### Comparação de Modelos

| Modelo            | R²     | MAE (R$) | RMSE (R$) | Hiperparâmetros                    |
| ----------------- | ------ | -------- | --------- | ---------------------------------- |
| **Lasso**         | 0.9803 | 2.527,46 | 2.917,80  | alpha=1.0                          |
| Ridge             | 0.9803 | 2.528,62 | 2.919,74  | alpha=1.0                          |
| XGBoost           | 0.9784 | 2.600,40 | 3.058,37  | lr=0.1, depth=3, n_est=200         |
| Gradient Boosting | 0.9783 | 2.609,82 | 3.063,99  | lr=0.1, depth=3, n_est=200         |
| LightGBM          | 0.9780 | 2.611,80 | 3.085,47  | lr=0.05, n_est=200, leaves=31      |
| Random Forest     | 0.9749 | 2.746,92 | 3.293,29  | depth=None, min_split=5, n_est=200 |

### Modelo Campeão: Lasso Regression

- **R² Score**: 0.9803 (98,03% da variância explicada)
- **MAE**: R$ 2.527,46 (erro médio absoluto)
- **RMSE**: R$ 2.917,80 (raiz do erro quadrático médio)
- **Hiperparâmetro**: alpha = 1.0
- **Features utilizadas**: 52 (após one-hot encoding)

## Pipeline de Machine Learning

### 1. Data Understanding

- **Dataset**: 10.000 registros, 9 variáveis
- **Marcas**: 10 fabricantes (Ford, Hyundai, Jeep, Toyota, Nissan, Fiat, Honda, Chevrolet, Volkswagen, Renault)
- **Modelos**: 32 modelos diferentes
- **Faixa de preços**: R$ 10.000 - R$ 1.241.962

### 2. Data Preparation

**Limpeza de Dados:**

- Remoção de 100 registros com valores nulos na coluna `Ano`
- Remoção de 99 registros com "erro_medicao" em `Quilometragem`
- Conversão de tipos: Ano → int, Quilometragem → float

**Tratamento de Outliers:**

- Aplicação de método IQR (Interquartile Range) agrupado por marca
- Remoção de 198 outliers (9.801 → 9.603 registros)
- Preservação da distribuição natural de preços por marca

**Feature Engineering:**

- `Idade_Veiculo`: Ano atual - Ano de fabricação
- `Km_por_Ano`: Quilometragem / (Idade_Veiculo + 1)

**Encoding:**

- One-Hot Encoding para variáveis categóricas (Marca, Modelo, Cor, Cambio, Combustivel)
- 52 features finais após encoding

### 3. Modeling

**Algoritmos Testados:**

1. Linear Regression (baseline)
2. Ridge Regression (regularização L2)
3. **Lasso Regression (regularização L1)**
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. LightGBM

**Otimização:**

- GridSearchCV com validação cruzada (cv=5)
- Métrica de otimização: R² Score
- Paralelização (n_jobs=-1)

### 4. Evaluation

**Métricas Utilizadas:**

- **R² Score**: Coeficiente de determinação
- **MAE**: Mean Absolute Error (erro médio em R$)
- **RMSE**: Root Mean Squared Error (penaliza erros grandes)

**Análise de Importância:**

- Gráficos gerados para 6 modelos
- Salvos em `reports/figures/`
- Principais features: Modelo, Marca, Quilometragem, Idade do Veículo

### 5. Deployment

**Artefatos Gerados:**

- Modelo serializado: `models/trained_models/lasso_modelo_campeao.pkl`
- Informações de features: `models/trained_models/feature_info.json`
- Métricas: `reports/metrics/resultados_modelos.{csv,json}`
- Gráficos: `reports/figures/importancia_features_*.png`

## Rastreamento de Experimentos (MLflow)

Este projeto utiliza **MLflow** para rastreamento completo de experimentos de Machine Learning.

### Visualizar Experimentos

Após executar o notebook `modelo_carros.ipynb`, todos os experimentos são registrados automaticamente.

Para visualizar a interface MLflow:

```bash
cd notebooks
mlflow ui --port 5000
```

Acesse: **http://localhost:5000**

### O que é Rastreado

**Para cada modelo treinado:**
- ✅ Hiperparâmetros otimizados (GridSearchCV)
- ✅ Métricas de performance (MAE, RMSE, R²)
- ✅ Informações do dataset (tamanho, features)
- ✅ Modelo serializado
- ✅ Configurações de treinamento (CV folds, random_state)

**Experimentos disponíveis:**
- Ridge Regression (GridSearch com 4 alphas)
- Lasso Regression (GridSearch com 4 alphas)
- Random Forest (GridSearch)
- Gradient Boosting (GridSearch)
- XGBoost (GridSearch)
- LightGBM (GridSearch)

### Model Registry

O modelo campeão (Lasso) está registrado no MLflow Model Registry:

- **Nome**: `PredicaoCarros_Lasso_Campeao`
- **Versão**: 1.0
- **Status**: Produção
- **Métricas**: R²=0.9803, MAE=R$2.527, RMSE=R$2.918

### Comparação de Modelos no MLflow UI

No MLflow UI você pode:
1. ✅ Comparar métricas lado a lado
2. ✅ Visualizar gráficos de performance
3. ✅ Exportar resultados em CSV
4. ✅ Baixar modelos específicos
5. ✅ Filtrar por tags (framework, model_family)

### Estrutura de Experimentos

```
notebooks/mlruns/
└── 0/  (Experimento: Predicao_Precos_Carros)
    ├── Ridge_GridSearch/
    ├── Lasso_GridSearch/
    ├── Random_Forest_GridSearch/
    ├── Gradient_Boosting_GridSearch/
    ├── XGBoost_GridSearch/
    ├── LightGBM_GridSearch/
    └── Modelo_Campeao_Lasso/
```

## Como Usar o Modelo Treinado

### Opção 1: Carregar do MLflow Model Registry

```python
import mlflow
import mlflow.sklearn

# Carregar modelo do MLflow
model_uri = "models:/PredicaoCarros_Lasso_Campeao/1"
modelo = mlflow.sklearn.load_model(model_uri)

# Fazer predição
preco_predito = modelo.predict(dados_preparados)
print(f"Preço estimado: R$ {preco_predito[0]:,.2f}")
```

### Opção 2: Carregar do Arquivo Local

```python
import joblib
import pandas as pd

# Carregar modelo
modelo = joblib.load('models/trained_models/lasso_modelo_campeao.pkl')

# Preparar dados de entrada (deve seguir o mesmo preprocessing)
# Exemplo de estrutura esperada:
# - One-hot encoding para Marca, Modelo, Cor, Cambio, Combustivel
# - Features numéricas: Quilometragem, Portas, Idade_Veiculo, Km_por_Ano
# - Total: 52 features

# Fazer predição
preco_predito = modelo.predict(dados_preparados)
print(f"Preço estimado: R$ {preco_predito[0]:,.2f}")
```

## Estrutura de Código

### Padrões de Desenvolvimento

- Código organizado e comentado no notebook
- Uso de bibliotecas padrão da indústria
- Reprodutibilidade garantida (random_state=42)
- Artefatos salvos automaticamente

## Contribuição ao Projeto

### Diretrizes para Colaboradores

1. Fork do repositório principal
2. Criação de branch específica para feature
3. Desenvolvimento com código limpo e comentado
4. Pull request com descrição detalhada

### Padrões de Commit

```
feat: adiciona novo modelo de ensemble
fix: corrige encoding de variáveis categóricas
docs: atualiza documentação do README
refactor: reorganiza estrutura do projeto
```

## Links Importantes

- **Repositório GitHub**: [MachineLearningCarros](https://github.com/HiRenan/MachineLearningCarros)
- **Branch Principal**: master
- **Branch de Desenvolvimento**: develop

## Status do Projeto

![Status](https://img.shields.io/badge/Status-Completo-green)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Best Model](https://img.shields.io/badge/R²-98.03%25-success)

## Próximos Passos

- [ ] Implementar interface web com Streamlit
- [ ] Deploy em plataforma cloud (Hugging Face Spaces/Render)
- [ ] Adicionar intervalos de confiança nas predições
- [ ] Implementar sistema de monitoramento do modelo
- [ ] Criar API REST para predições

## Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo LICENSE para detalhes.

## Referências

1. Chapman, P., et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
5. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.

---

**Desenvolvido por**: Renan & Nathan
**Instituição**: Projeto Educacional de Machine Learning
**Ano**: 2025
