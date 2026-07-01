"""
Modelos Pydantic para validación de requests/responses de la API.

PredictResponse:
  - exito: bool → True si la inferencia se completó
  - prediccion: str | None → clase ganadora o None si < threshold
  - top_3: lista con clase, nombre_mostrar y confianza para debugging

ChatRequest/ChatResponse:
  - mensaje: str → texto del usuario
  - respuesta: str | None → texto generado por Gemini
"""

from pydantic import BaseModel
from typing import List, Optional

class PredictionItem(BaseModel):
    clase: str
    confianza: float
    nombre_mostrar: str

class PredictResponse(BaseModel):
    exito: bool
    prediccion: Optional[str] = None
    confianza: Optional[float] = None
    top_3: Optional[List[PredictionItem]] = None
    mensaje: Optional[str] = None

class ChatRequest(BaseModel):
    mensaje: str

class ChatResponse(BaseModel):
    exito: bool
    respuesta: Optional[str] = None
    mensaje: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    modelo_cargado: bool
    clases: int
    version: str
