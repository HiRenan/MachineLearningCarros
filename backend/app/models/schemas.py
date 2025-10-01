from pydantic import BaseModel, Field, validator
from typing import Optional


class CarInput(BaseModel):
    marca: str = Field(..., description="Fabricante do veículo")
    modelo: str = Field(..., description="Modelo do veículo")
    ano: int = Field(..., ge=2000, le=2024, description="Ano de fabricação")
    quilometragem: float = Field(..., ge=0, description="Quilometragem do veículo")
    cor: str = Field(..., description="Cor do veículo")
    cambio: str = Field(..., description="Tipo de câmbio (Manual/Automático)")
    combustivel: str = Field(..., description="Tipo de combustível (Flex/Gasolina/Diesel)")
    portas: int = Field(..., ge=2, le=4, description="Número de portas")

    @validator('cambio')
    def validate_cambio(cls, v):
        valid_options = ['Manual', 'Automático']
        if v not in valid_options:
            raise ValueError(f'Câmbio deve ser um dos valores: {valid_options}')
        return v

    @validator('combustivel')
    def validate_combustivel(cls, v):
        valid_options = ['Flex', 'Gasolina', 'Diesel']
        if v not in valid_options:
            raise ValueError(f'Combustível deve ser um dos valores: {valid_options}')
        return v

    @validator('portas')
    def validate_portas(cls, v):
        if v not in [2, 4]:
            raise ValueError('Número de portas deve ser 2 ou 4')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "marca": "Toyota",
                "modelo": "Corolla",
                "ano": 2020,
                "quilometragem": 45000.0,
                "cor": "Prata",
                "cambio": "Automático",
                "combustivel": "Flex",
                "portas": 4
            }
        }


class PredictionOutput(BaseModel):
    valor_predito: float = Field(..., description="Valor de venda predito em R$")
    valor_minimo: Optional[float] = Field(None, description="Valor mínimo estimado")
    valor_maximo: Optional[float] = Field(None, description="Valor máximo estimado")
    confianca: str = Field(..., description="Nível de confiança da predição")

    class Config:
        json_schema_extra = {
            "example": {
                "valor_predito": 78500.00,
                "valor_minimo": 75582.20,
                "valor_maximo": 81417.80,
                "confianca": "alta"
            }
        }


class ModelInfo(BaseModel):
    nome_modelo: str
    r2_score: float
    mae: float
    rmse: float
    total_features: int
    descricao: str

    class Config:
        json_schema_extra = {
            "example": {
                "nome_modelo": "Lasso Regression",
                "r2_score": 0.9803,
                "mae": 2527.46,
                "rmse": 2917.80,
                "total_features": 52,
                "descricao": "Modelo campeão com regularização L1"
            }
        }


class HealthCheck(BaseModel):
    status: str
    modelo_carregado: bool
    versao_api: str
