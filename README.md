# Predição de Preços de Veículos no Mercado Brasileiro

![Status](https://img.shields.io/badge/Status-Em%20Produção-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688)
![React](https://img.shields.io/badge/React-19.1.1-61DAFB)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Best Model](https://img.shields.io/badge/R²-98.03%25-success)

## Aviso Educacional

Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.

## Demonstração

**Acesse a aplicação em produção**: [https://carros-frontend.onrender.com/](https://carros-frontend.onrender.com/)

## Resumo Executivo

Este projeto implementa um pipeline completo de Machine Learning para predição de valores de venda de veículos automotores no mercado brasileiro. O sistema é composto por:

- **Notebook Jupyter** consolidado seguindo metodologia CRISP-DM
- **API REST** desenvolvida em FastAPI para servir predições
- **Interface Web** moderna em React para interação com usuários
- **Deploy em produção** no Render com arquitetura full-stack

O modelo campeão (Lasso Regression) alcançou **R² de 98,03%**, com erro médio absoluto de apenas R$ 2.527, demonstrando alta precisão na predição de preços de veículos usados.

## Arquitetura do Sistema

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  React Frontend │ ──────> │  FastAPI Backend │ ──────> │  Modelo ML      │
│  (Vite + Axios) │  HTTPS  │  (REST API)      │  Joblib │  (Lasso .pkl)   │
└─────────────────┘         └──────────────────┘         └─────────────────┘
       ↓                             ↓                            ↓
   Render Deploy              Render Deploy                  Scikit-Learn
```

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

## Estrutura do Projeto

```
DesafioFinalML/
├── backend/                           # API REST FastAPI
│   ├── app/
│   │   ├── main.py                    # Endpoints da API
│   │   ├── models/schemas.py          # Schemas Pydantic
│   │   └── services/prediction.py    # Lógica de predição
│   └── requirements.txt               # Dependências backend
├── frontend/                          # Interface Web React
│   ├── src/
│   │   ├── components/                # Componentes React
│   │   ├── services/api.js            # Cliente Axios
│   │   └── App.jsx                    # Componente principal
│   ├── package.json                   # Dependências frontend
│   └── vite.config.js                 # Configuração Vite
├── data/
│   ├── raw/
│   │   └── dataset_carros_brasil.csv  # Dataset original
│   └── processed/
│       ├── dataset_limpo.csv          # Dados processados (9.603 registros)
│       └── dataset_limpo.pkl          # Dados em formato pickle
├── notebooks/
│   └── modelo_carros.ipynb            # Notebook consolidado CRISP-DM
├── mlruns/                            # Experimentos MLflow
├── models/
│   └── trained_models/
│       ├── lasso_modelo_campeao.pkl   # Modelo campeão treinado
│       └── feature_info.json          # Informações das features
├── reports/
│   ├── figures/
│   │   ├── importancia_features_*.png # Gráficos de importância (6 modelos)
│   │   └── mlflow_comparacao_modelos.png
│   └── metrics/
│       ├── resultados_modelos.csv     # Métricas em CSV
│       └── resultados_modelos.json    # Métricas em JSON
├── src/                               # Estrutura modular Python
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── requirements.txt                   # Dependências ML pipeline
├── runtime.txt                        # Python 3.9.18
└── README.md                          # Esta documentação
```

## Tecnologias Utilizadas

### Machine Learning & Data Science

- **Python 3.9+** (testado em 3.9.18 e 3.13): Linguagem base
- **Pandas ≥2.2.0**: Manipulação de dados
- **NumPy ≥2.0.0**: Computação numérica
- **Scikit-learn ≥1.3.0**: Algoritmos de ML
- **XGBoost ≥2.0.0**: Gradient boosting
- **LightGBM ≥4.1.0**: Gradient boosting eficiente
- **MLflow ≥2.8.0**: Rastreamento de experimentos

### Backend

- **FastAPI 0.115.0**: Framework web moderno
- **Uvicorn 0.32.0**: Servidor ASGI
- **Pydantic 2.9.0**: Validação de dados
- **Scikit-learn 1.5.2**: Modelo ML (backend)
- **Joblib 1.4.2**: Serialização de modelos

### Frontend

- **React 19.1.1**: Biblioteca UI
- **Vite 7.1.7**: Build tool
- **Axios 1.12.2**: Cliente HTTP
- **React Icons 5.5.0**: Ícones

### DevOps & Deploy

- **Render**: Plataforma de deploy
- **Git**: Controle de versão
- **Jupyter Notebook**: Desenvolvimento interativo

## Instalação e Configuração

### 1. Clonagem do Repositório

```bash
git clone https://github.com/HiRenan/MachineLearningCarros.git
cd MachineLearningCarros
```

### 2. Configuração do Ambiente ML (Notebooks)

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar notebook
jupyter notebook notebooks/modelo_carros.ipynb
```

### 3. Configuração do Backend

```bash
cd backend

# Instalar dependências
pip install -r requirements.txt

# Executar servidor
python -m uvicorn app.main:app --reload --port 8000
```

**Windows:** Se o comando `uvicorn` não funcionar, use `python -m uvicorn app.main:app --reload`

Acesse: http://localhost:8000/docs (Swagger UI)

### 4. Configuração do Frontend

```bash
cd frontend

# Instalar dependências
npm install

# Executar em desenvolvimento
npm run dev
```

Acesse: http://localhost:5173

## API REST - Endpoints

### Base URL (Produção)

```
https://carros-backend.onrender.com
```

### Endpoints Disponíveis

#### 1. Health Check

```http
GET /api/health
```

**Resposta:**

```json
{
  "status": "healthy",
  "modelo_carregado": true,
  "versao_api": "1.0.0"
}
```

#### 2. Informações do Modelo

```http
GET /api/model-info
```

**Resposta:**

```json
{
  "nome_modelo": "Lasso Regression",
  "r2_score": 0.9803,
  "mae": 2527.46,
  "rmse": 2917.8,
  "total_features": 52,
  "descricao": "Modelo campeão com regularização L1"
}
```

#### 3. Predição de Preço

```http
POST /api/predict
Content-Type: application/json

{
  "marca": "Toyota",
  "modelo": "Corolla",
  "ano": 2020,
  "quilometragem": 30000,
  "cor": "Prata",
  "cambio": "Automático",
  "combustivel": "Flex",
  "portas": 4
}
```

**Resposta:**

```json
{
  "valor_predito": 95000.5,
  "valor_minimo": 92473.04,
  "valor_maximo": 97527.96,
  "confianca": "alta"
}
```

#### 4. Opções de Entrada

```http
GET /api/options/marcas      # Lista de marcas disponíveis
GET /api/options/modelos     # Modelos por marca
GET /api/options/cores       # Cores disponíveis
GET /api/options/cambios     # Tipos de câmbio
GET /api/options/combustiveis # Tipos de combustível
```

### Documentação Interativa

- **Swagger UI**: https://carros-backend.onrender.com/docs
- **ReDoc**: https://carros-backend.onrender.com/redoc

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

## Pipeline de Machine Learning (CRISP-DM)

### 1. Business Understanding

- Objetivo: Predizer valor de venda de veículos usados
- Métrica de sucesso: R² > 0.95
- Aplicação: Auxiliar compradores e vendedores na precificação

### 2. Data Understanding

- **Dataset**: 10.000 registros, 9 variáveis
- **Marcas**: 10 fabricantes (Ford, Hyundai, Jeep, Toyota, Nissan, Fiat, Honda, Chevrolet, Volkswagen, Renault)
- **Modelos**: 32 modelos diferentes
- **Faixa de preços**: R$ 10.000 - R$ 1.241.962

### 3. Data Preparation

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

### 4. Modeling

**Algoritmos Testados:**

1. Linear Regression (baseline)
2. Ridge Regression (regularização L2)
3. **Lasso Regression (regularização L1) - CAMPEÃO**
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. LightGBM

**Otimização:**

- GridSearchCV com validação cruzada (cv=5)
- Métrica de otimização: R² Score
- Paralelização (n_jobs=-1)

### 5. Evaluation

**Métricas Utilizadas:**

- **R² Score**: Coeficiente de determinação
- **MAE**: Mean Absolute Error (erro médio em R$)
- **RMSE**: Root Mean Squared Error (penaliza erros grandes)

**Análise de Importância:**

- Gráficos gerados para 6 modelos
- Salvos em `reports/figures/`
- Principais features: Modelo, Marca, Quilometragem, Idade do Veículo

### 6. Deployment

**Arquitetura em Produção:**

- Frontend React hospedado no Render
- Backend FastAPI hospedado no Render
- Comunicação via API REST (HTTPS)
- Modelo serializado carregado em memória

**Artefatos Gerados:**

- Modelo serializado: `models/trained_models/lasso_modelo_campeao.pkl`
- Informações de features: `models/trained_models/feature_info.json`
- Métricas: `reports/metrics/resultados_modelos.{csv,json}`
- Gráficos: `reports/figures/importancia_features_*.png`

## Rastreamento de Experimentos (MLflow)

### Visualizar Experimentos

Após executar o notebook `modelo_carros.ipynb`, todos os experimentos são registrados automaticamente.

Para visualizar a interface MLflow:

```bash
mlflow ui --port 5000
```

Acesse: **http://localhost:5000**

### O que é Rastreado

**Para cada modelo treinado:**

- Hiperparâmetros otimizados (GridSearchCV)
- Métricas de performance (MAE, RMSE, R²)
- Informações do dataset (tamanho, features)
- Modelo serializado
- Configurações de treinamento (CV folds, random_state)

**Experimentos disponíveis:**

- Ridge Regression (GridSearch)
- Lasso Regression (GridSearch)
- Random Forest (GridSearch)
- Gradient Boosting (GridSearch)
- XGBoost (GridSearch)
- LightGBM (GridSearch)

## Interface Web

### Funcionalidades

- Formulário intuitivo para entrada de dados do veículo
- Validação de campos em tempo real
- Predição instantânea ao submeter formulário
- Exibição de valor estimado com intervalo de confiança
- Design responsivo (mobile-friendly)
- Disclaimer educacional visível

### Tecnologias Frontend

- **React 19.1.1**: Componentes funcionais com hooks
- **Vite**: Build ultrarrápido e HMR
- **Axios**: Requisições HTTP para API
- **React Icons**: Ícones modernos
- **CSS Modules**: Estilização isolada

## Como Usar o Modelo Treinado

### Via API REST (Recomendado)

```python
import requests

url = "https://carros-backend.onrender.com/api/predict"
payload = {
    "marca": "Toyota",
    "modelo": "Corolla",
    "ano": 2020,
    "quilometragem": 30000,
    "cor": "Prata",
    "cambio": "Automático",
    "combustivel": "Flex",
    "portas": 4
}

response = requests.post(url, json=payload)
resultado = response.json()
print(f"Preço estimado: R$ {resultado['valor_predito']:,.2f}")
```

### Via Arquivo Local

```python
import joblib
import pandas as pd

# Carregar modelo
modelo = joblib.load('models/trained_models/lasso_modelo_campeao.pkl')

# Preparar dados de entrada (deve seguir o mesmo preprocessing)
# - One-hot encoding para Marca, Modelo, Cor, Cambio, Combustivel
# - Features numéricas: Quilometragem, Portas, Idade_Veiculo, Km_por_Ano
# - Total: 52 features

# Fazer predição
preco_predito = modelo.predict(dados_preparados)
print(f"Preço estimado: R$ {preco_predito[0]:,.2f}")
```

## Deploy no Render

### Configurações de Deploy

**Frontend:**

- Build Command: `npm install && npm run build`
- Publish Directory: `dist`
- Node Version: 20.x

**Backend:**

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Python Version: 3.9.18 (definido em runtime.txt)

### Variáveis de Ambiente

**Frontend:**

```env
VITE_API_URL=https://carros-backend.onrender.com
```

**Backend:**

```env
MODEL_PATH=/opt/render/project/src/models/trained_models/lasso_modelo_campeao.pkl
```

## Estrutura do Notebook

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
12. **Rastreamento MLflow**: Registro de experimentos

## Contribuição ao Projeto

### Diretrizes para Colaboradores

1. Fork do repositório principal
2. Criação de branch específica para feature
3. Desenvolvimento com código limpo e comentado
4. Testes locais completos
5. Pull request com descrição detalhada

### Padrões de Commit

```
feat: adiciona novo modelo de ensemble
fix: corrige encoding de variáveis categóricas
docs: atualiza documentação do README
refactor: reorganiza estrutura do projeto
style: formata código Python com black
test: adiciona testes unitários para API
```

## Links Importantes

- **Aplicação em Produção**: [https://carros-frontend.onrender.com/](https://carros-frontend.onrender.com/)
- **API Backend**: [https://carros-backend.onrender.com/docs](https://carros-backend.onrender.com/docs)
- **Repositório GitHub**: [MachineLearningCarros](https://github.com/HiRenan/MachineLearningCarros)
- **Branch Principal**: master
- **Branch de Desenvolvimento**: develop

## Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo LICENSE para detalhes.

## Referências

1. Chapman, P., et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
5. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
6. FastAPI Documentation. (2024). https://fastapi.tiangolo.com/
7. React Documentation. (2024). https://react.dev/

---

**Desenvolvido por**: Renan Mocelin, Nathan Arrais, Lucas Porfilio, João Casemiro
**Instituição**: Projeto Educacional de Machine Learning
**Ano**: 2025
**Status**: Em Produção
**Última Atualização**: Outubro 2025

## Troubleshooting

### Problemas com Instalação de Dependências (Windows)

Se você encontrar erros ao instalar dependências no Windows (especialmente com Python 3.13), o `requirements.txt` foi atualizado para usar versões flexíveis (≥) que são compatíveis com versões mais recentes do Python.

**Erro com numpy/pandas:**

```bash
# Solução: O requirements.txt já usa versões flexíveis
pip install -r requirements.txt
```

**Erro "uvicorn: command not found":**

```bash
# Use o Python para executar o uvicorn
python -m uvicorn app.main:app --reload
```

**Erro no Jupyter "ModuleNotFoundError":**

```bash
# Instale no ambiente correto do Jupyter
python -m pip install numpy pandas matplotlib seaborn scikit-learn
# Depois reinicie o kernel do Jupyter
```
