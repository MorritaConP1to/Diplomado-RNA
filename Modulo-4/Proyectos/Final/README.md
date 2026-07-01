# Sanrio Classifier Web 🎀

Clasificador web de personajes Sanrio (30 clases) usando ResNet18 Transfer Learning.
Proyecto final del Diplomado RNA y Deep Learning — UAEM. Julio 2026.

## Stack
- **Modelo**: ResNet18 Transfer Learning → ONNX Runtime (CPU, 11.4 MB)
- **Backend**: FastAPI + Uvicorn + Rate Limiting (slowapi)
- **Frontend**: HTML/CSS/JS vanilla (identidad visual Sanrio pastel goth)
- **Extra**: Chatbot Gemini con system prompt anti-jailbreak (+10 pts)
- **Deploy**: HuggingFace Spaces (Docker, imagen ~1.5GB sin torch)

## Arquitectura

```
Usuario → Frontend (HTML/CSS/JS) → Backend (FastAPI)
                                     ├── /predict → Modelo ONNX → Top-3
                                     ├── /chat    → Gemini API  → Respuesta
                                     └── /health  → Status check
```

## Las 30 clases

### Top 20 Ranking 2026 Sanrio (11 nuevos de los 20)
| # | Personaje | Ranking 2026 |
|---|-----------|:-----------:|
| 1 | Badtz-Maru | #11 |
| 2 | Chococat | — |
| 3 | Cinnamoroll | #2 |
| 4 | Hello Kitty | #5 |
| 5 | Keroppi | #14 |
| 6 | Kuromi | #4 |
| 7 | My Melody | #7 |
| 8 | Pochacco | #3 |
| 9 | Pompompurin | #1 🏆 |
| 10 | Tuxedo Sam | #8 |
| 11 | Pekkle | #6 |
| 12 | Hangyodon | #9 |
| 13 | Little Twin Stars | #10 |
| 14 | Cogimyun | #12 |
| 15 | My Sweet Piano | #13 |
| 16 | Hanamaruobake | #15 |
| 17 | Wish Me Mell | #16 |
| 18 | Usahana | #17 |
| 19 | Gaopowerroo | #18 |
| 20 | Corocorokuririn | #19 |
| 21 | Gudetama | #20 |

### 9 adicionales
| # | Personaje | Dato clave |
|---|-----------|-----------|
| 22 | Aggretsuko | Netflix star |
| 23 | Kirimichan | Salmon filet |
| 24 | Marroncream | Squirrel |
| 25 | Marumofubiyori | Polar bear |
| 26 | Charmmy Kitty | HK's cat |
| 27 | Dear Daniel | HK's boyfriend |
| 28 | Sugarbunnies | Twin bunnies |
| 29 | Yoshikitty | HK x Yoshiki |
| 30 | Hello Mimmy | HK's twin |

## Cómo ejecutar (full pipeline)

### 1. Scraping (solo si faltan imágenes)
```powershell
python scraping/scraping_ampliacion_20.py   # 20 nuevos personajes (~3-5h)
python scraping/preprocesar_raw.py          # Validar y convertir a JPG
python scraping/split_multiclase.py         # Dividir 80/20 train/test
```

### 2. Re-entrenar modelo 30 clases
Abrir `notebooks/Transfer_Learning_Sanrio_v2.ipynb` en VSCode y ejecutar celdas.
O desde terminal:
```powershell
jupyter notebook notebooks/Transfer_Learning_Sanrio_v2.ipynb
```

### 3. Exportar a ONNX
```powershell
python models/export_onnx.py     # .pth → ONNX INT8 (11.4 MB)
```

### 4. Servir web app
```powershell
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Abrir http://localhost:8000
```

### 5. Activar chatbot Gemini
```powershell
# Copia .env.example a .env y edita con tu API key
Copy-Item .env.example .env
# O directamente:
$env:GEMINI_API_KEY="tu-api-key-aqui"
uvicorn app.main:app --reload
```

### 6. Build Docker con dependencias ligeras (sin torch)
```powershell
docker build -t sanrio-classifier .
docker run -p 7860:7860 -e GEMINI_API_KEY="tu-key" sanrio-classifier
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Frontend HTML |
| GET | `/health` | Health check (modelo cargado, clases) |
| POST | `/predict` | Clasificar imagen (multipart) |
| POST | `/chat` | Chatbot Gemini (JSON) |

### Ejemplo /predict
```bash
curl -X POST http://localhost:8000/predict -F "file=@foto.jpg"
```
```json
{
  "exito": true,
  "prediccion": "kuromi",
  "nombre_mostrar": "Kuromi",
  "confianza": 95.8,
  "top_3": [
    {"clase": "kuromi", "nombre_mostrar": "Kuromi", "confianza": 95.8},
    {"clase": "my_melody", "nombre_mostrar": "My Melody", "confianza": 2.1},
    {"clase": "cinnamon", "nombre_mostrar": "Cinnamoroll", "confianza": 1.0}
  ],
  "mensaje": null
}
```

## Contramedidas (defensa en profundidad)

| Vector de ataque | Contramedida |
|-----------------|--------------|
| Imagen no-Sanrio | Threshold <30% → "No parece Sanrio conocido" |
| Imagen corrupta | Validación PIL + try/except |
| Tamaño extremo | Rechazar <32px o >4000px |
| Jailbreak chatbot | System prompt restrictivo + rejection template + defensa anti-DAN/ignore previous |
| API key expuesta | Sólo en backend (env var), frontend llama a /chat |
| Spam requests | Rate limiting: 100 req/min (/predict), 30 req/min (/chat) |
| Sanrio no cubierto | Mensaje: "Podría ser un Sanrio no entrenado" |

## Estructura del proyecto

```
Final/                              # ← Proyecto unificado
├── app/                            # Web app (FastAPI + frontend)
│   ├── main.py                     # Rutas /predict, /chat, /health
│   ├── model.py                    # Carga ONNX + inferencia + threshold
│   ├── schemas.py                  # Pydantic models
│   ├── config.py                   # Configuración
│   ├── gemini_chat.py              # Proxy Gemini con system prompt
│   ├── utils.py                    # Validación y preprocesamiento
│   └── static/
│       ├── index.html              # Frontend pastel goth
│       └── personajes.json         # Datos de 30 personajes
├── models/                         # Modelo para deploy (ONNX)
│   ├── clases_sanrio.json          # Mapeo clases
│   ├── tl_sanrio_int8.onnx         # ONNX cuantizado (11.4 MB)
│   └── export_onnx.py              # Script de exportación
├── modelos/                        # Pesos originales .pth
│   ├── tl_sanrio_final.pth         # ResNet18 entrenado (10 clases)
│   ├── experimentos_tl.json        # Bitácora
│   └── clases_sanrio.json          # Mapeo clases (entrenamiento)
├── dataset/                        # Imágenes
│   ├── raw/                        # Scraping → aquí caen
│   ├── train/                      # 80% entrenamiento
│   └── test/                       # 20% evaluación
├── scraping/                       # Scrapers
│   ├── scraping_ampliacion_20.py   # Scraper 20 nuevos chars
│   ├── preprocesar_raw.py          # Validar y convertir
│   └── split_multiclase.py        # Split 80/20
├── notebooks/                      # Notebooks entrenamiento
│   ├── Transfer_Learning_Sanrio_v2.ipynb
│   ├── CNN_Sanrio_v2.ipynb
│   └── generar_notebook_sanrio.py
├── Dockerfile                      # Para HuggingFace Spaces
├── requirements.txt                # Dependencias (full, con torch)
├── requirements-deploy.txt         # Dependencias ligeras (sin torch)
├── .env.example                    # Template para GEMINI_API_KEY
└── README.md                       # Este archivo
```

## Ver también
- `AGENTS.md` en raíz del repositorio — documentación para asistentes de IA
- `scraping/` — scripts de descarga y preprocesamiento
- `notebooks/` — notebooks de entrenamiento
