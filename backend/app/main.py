from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import CarInput, PredictionOutput, ModelInfo, HealthCheck
from app.services.prediction import predictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Predição de Preços de Veículos",
    description="API para predição de valores de venda de veículos no mercado brasileiro",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "API de Predição de Preços de Veículos",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "model_info": "/api/model-info",
            "predict": "/api/predict"
        }
    }


@app.get("/api/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "modelo_carregado": predictor.is_loaded(),
        "versao_api": "1.0.0"
    }


@app.get("/api/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    try:
        return predictor.get_model_info()
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações do modelo: {str(e)}")


@app.post("/api/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_price(car_data: CarInput):
    try:
        logger.info(f"Recebendo predição para: {car_data.marca} {car_data.modelo} {car_data.ano}")

        input_dict = {
            'marca': car_data.marca,
            'modelo': car_data.modelo,
            'ano': car_data.ano,
            'quilometragem': car_data.quilometragem,
            'cor': car_data.cor,
            'cambio': car_data.cambio,
            'combustivel': car_data.combustivel,
            'portas': car_data.portas
        }

        valor_predito, valor_minimo, valor_maximo, confianca = predictor.predict(input_dict)

        logger.info(f"Predição realizada: R$ {valor_predito:,.2f}")

        return {
            "valor_predito": valor_predito,
            "valor_minimo": valor_minimo,
            "valor_maximo": valor_maximo,
            "confianca": confianca
        }

    except Exception as e:
        logger.error(f"Erro ao realizar predição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao realizar predição: {str(e)}")


@app.get("/api/options/marcas", tags=["Options"])
async def get_marcas():
    return {
        "marcas": ["Chevrolet", "Fiat", "Ford", "Honda", "Hyundai", "Jeep", "Nissan", "Renault", "Toyota", "Volkswagen"]
    }


@app.get("/api/options/modelos", tags=["Options"])
async def get_modelos():
    modelos_por_marca = {
        "Chevrolet": ["Onix", "Onix Plus", "S10", "Tracker"],
        "Fiat": ["Cronos", "Mobi"],
        "Ford": ["EcoSport", "Ka", "Ranger"],
        "Honda": ["Civic", "Fit", "HR-V"],
        "Hyundai": ["Creta", "HB20", "HB20S"],
        "Jeep": ["Compass", "Renegade"],
        "Nissan": ["Frontier", "Kicks", "Versa"],
        "Renault": ["Duster", "Kwid", "Sandero"],
        "Toyota": ["Corolla", "Hilux", "Yaris"],
        "Volkswagen": ["Gol", "Polo", "T-Cross", "Toro", "Virtus"]
    }
    return {"modelos": modelos_por_marca}


@app.get("/api/options/cores", tags=["Options"])
async def get_cores():
    return {
        "cores": ["Azul", "Branco", "Cinza", "Prata", "Preto", "Vermelho"]
    }


@app.get("/api/options/cambios", tags=["Options"])
async def get_cambios():
    return {
        "cambios": ["Manual", "Automático"]
    }


@app.get("/api/options/combustiveis", tags=["Options"])
async def get_combustiveis():
    return {
        "combustiveis": ["Flex", "Gasolina", "Diesel"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
