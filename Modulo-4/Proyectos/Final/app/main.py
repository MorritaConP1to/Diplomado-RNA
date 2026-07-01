"""
FastAPI backend del Clasificador Sanrio 30 clases.

Endpoints:
  GET  /         → Frontend HTML (pastel goth Sanrio)
  GET  /health   → Health check del servidor y modelo
  POST /predict  → Clasificar imagen (multipart) → top-3 + confianza
  POST /chat     → Chatbot Gemini (JSON) → respuesta Sanrio-only

Defensas implementadas:
  - Rate limiting: 100 req/min /predict, 30 req/min /chat (slowapi)
  - Validación de imágenes: formato, tamaño, tipo MIME, corrupción PIL
  - Threshold de confianza <30% → "No parece Sanrio conocido"
  - Limpieza de archivos temporales al finalizar
  - Gemini API key SOLO en backend (variable de entorno)
"""

import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import CONF_THRESHOLD
from app.model import clasificador
from app.utils import validar_archivo, validar_imagen, preprocess_image
from app.schemas import PredictResponse, ChatRequest, ChatResponse, HealthResponse
from app.gemini_chat import generar_respuesta

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Sanrio Classifier", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'static')
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def startup():
    ok = clasificador.cargar()
    if not ok:
        print("[WARN] Modelo ONNX no encontrado. La inferencia no estara disponible.")
    else:
        print(f"[OK] Modelo cargado: {len(clasificador.clases)} clases")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Sanrio Classifier</h1><p>Frontend no encontrado</p>")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        modelo_cargado=clasificador.cargado,
        clases=len(clasificador.clases) if clasificador.cargado else 0,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("100/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    if not clasificador.cargado:
        return PredictResponse(
            exito=False,
            mensaje="El modelo no está cargado. Contacta al administrador."
        )

    valido, msg = validar_archivo(file.filename, file.content_type or "")
    if not valido:
        return PredictResponse(exito=False, mensaje=msg)

    try:
        contents = await file.read()
    except Exception:
        return PredictResponse(exito=False, mensaje="Error al leer el archivo.")

    valido, msg = validar_archivo(file.filename, file.content_type or "", len(contents))
    if not valido:
        return PredictResponse(exito=False, mensaje=msg)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        tmp.write(contents)
        tmp.close()

        valido, msg = validar_imagen(tmp.name)
        if not valido:
            os.unlink(tmp.name)
            return PredictResponse(exito=False, mensaje=msg)

        tensor = preprocess_image(tmp.name)
        resultado = clasificador.predecir(tensor, threshold=CONF_THRESHOLD)
        return PredictResponse(**resultado)
    except Exception as e:
        return PredictResponse(
            exito=False,
            mensaje=f"Error interno: {str(e)}"
        )
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest):
    if not body.mensaje.strip():
        return ChatResponse(exito=False, mensaje="El mensaje no puede estar vacío.")
    try:
        respuesta = generar_respuesta(body.mensaje)
        return ChatResponse(exito=True, respuesta=respuesta)
    except Exception as e:
        return ChatResponse(exito=False, mensaje=f"Error: {str(e)}")
