# Backend - API de Predição de Preços de Veículos

API REST desenvolvida com FastAPI para predição de valores de venda de veículos no mercado brasileiro.

## Tecnologias

- FastAPI 0.115.0
- Uvicorn 0.32.0 (servidor ASGI)
- Pydantic 2.9.0 (validação de dados)
- Scikit-learn 1.5.2 (modelo ML)
- Pandas 2.2.3
- NumPy 2.1.2
- Python 3.9+ (testado em 3.13)

## Instalação

```bash
cd backend
pip install -r requirements.txt
```

## Execução

### Modo Desenvolvimento

```bash
python -m uvicorn app.main:app --reload --port 8000
```

**Windows (se uvicorn não estiver no PATH):**
```bash
python -m uvicorn app.main:app --reload
```

### Modo Produção

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Erro: `uvicorn: command not found`

Se o comando `uvicorn` não for encontrado (comum no Windows), use:
```bash
python -m uvicorn app.main:app --reload
```

### Erro: `Could not import module "main"`

Certifique-se de estar no diretório `backend/` e use:
```bash
python -m uvicorn app.main:app --reload
```

## Endpoints

### Health Check
- **GET** `/api/health`
- Verifica status da API e se o modelo está carregado

### Informações do Modelo
- **GET** `/api/model-info`
- Retorna métricas e informações do modelo treinado

### Predição
- **POST** `/api/predict`
- Realiza predição de preço do veículo

**Request Body:**
```json
{
  "marca": "Toyota",
  "modelo": "Corolla",
  "ano": 2020,
  "quilometragem": 45000.0,
  "cor": "Prata",
  "cambio": "Automático",
  "combustivel": "Flex",
  "portas": 4
}
```

**Response:**
```json
{
  "valor_predito": 78500.00,
  "valor_minimo": 75582.20,
  "valor_maximo": 81417.80,
  "confianca": "alta"
}
```

### Opções de Formulário
- **GET** `/api/options/marcas` - Lista de marcas disponíveis
- **GET** `/api/options/modelos` - Modelos por marca
- **GET** `/api/options/cores` - Cores disponíveis
- **GET** `/api/options/cambios` - Tipos de câmbio
- **GET** `/api/options/combustiveis` - Tipos de combustível

## Documentação Interativa

Após iniciar o servidor:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Estrutura

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # Endpoints da API
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Modelos Pydantic
│   └── services/
│       ├── __init__.py
│       └── prediction.py    # Lógica de predição
├── requirements.txt
└── README.md
```

## Modelo ML

- Algoritmo: Lasso Regression
- R² Score: 0.9803 (98.03%)
- MAE: R$ 2.527,46
- RMSE: R$ 2.917,80
- Features: 52 (após one-hot encoding)
