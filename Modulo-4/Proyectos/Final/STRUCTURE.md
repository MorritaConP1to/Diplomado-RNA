# Estructura del Proyecto Final

```
Modulo-4/Proyectos/Final/
│
├── .env.example                    # Template para GEMINI_API_KEY
├── Dockerfile                      # Build para HuggingFace Spaces (port 7860)
├── requirements.txt                # Full (con torch) para desarrollo local
├── requirements-deploy.txt         # Ligero (sin torch) para Docker/HF Spaces
├── README.md
├── Proyecto_Final_Clasificador_Requerimientos.pdf  # Rúbrica
│
├── app/                            # ★ Backend + Frontend (FastAPI)
│   ├── __init__.py
│   ├── main.py                     # FastAPI: /predict, /chat, /health, rate limit
│   ├── model.py                    # ONNX Runtime → inferencia + top-3 + threshold 30%
│   ├── schemas.py                  # Pydantic: PredictRequest, ChatRequest, etc.
│   ├── config.py                   # Thresholds, rutas, límites de imagen
│   ├── gemini_chat.py              # Proxy Gemini + system prompt anti-jailbreak
│   ├── utils.py                    # Validación 3 capas (formato, MIME, PIL)
│   └── static/
│       ├── index.html              # Frontend pastel goth (drag-drop, barras, chat)
│       └── personajes.json         # Datos de 30 personajes (debut, trivia, ranking)
│
├── models/                         # ★ Modelos para deploy
│   ├── clases_sanrio.json          # Mapeo índice → nombre
│   ├── tl_sanrio_int8.onnx         # ONNX INT8 cuantizado (~11 MB)
│   └── export_onnx.py              # Script: .pth → ONNX FP32 + INT8
│
├── modelos/                        # ★ Pesos de entrenamiento
│   ├── tl_sanrio_final.pth         # ResNet18 fine-tuned (43 MB)
│   ├── tl_sanrio_int8.onnx         # Respaldo del ONNX
│   ├── clases_sanrio.json          # Mapeo usado en entrenamiento
│   └── experimentos_tl.json        # Bitácora de experimentos
│
├── dataset/                        # ★ Imágenes
│   ├── raw/                        # 19 carpetas (imágenes recién scrapeadas)
│   │   ├── pekkle/ → 182 img
│   │   ├── hangyodon/ → 209 img
│   │   ├── cogimyun/ → 190 img
│   │   ├── little_twin_stars/ → 210 img
│   │   ├── my_sweet_piano/ → 118 img
│   │   ├── hanamaruobake/ → 180 img
│   │   ├── wish_me_mell/ → 160 img
│   │   ├── usahana/ → 196 img
│   │   ├── gaopowerroo/ → 140 img
│   │   ├── kuririn/ → 132 img
│   │   ├── gudetama/ → 212 img
│   │   ├── aggretsuko/ → 200 img
│   │   ├── kirimichan/ → 175 img
│   │   ├── marroncream/ → 160 img
│   │   ├── marumofubiyori/ → 190 img
│   │   ├── charmmykitty/ → 195 img
│   │   ├── dear_daniel/ → 170 img
│   │   ├── sugarbunnies/ → 150 img
│   │   └── yoshikitty/ → 143 img
│   ├── train/                      # 29 clases, 4,914 imágenes
│   │   ├── cinnamon/ → 474 img     (clase más grande)
│   │   ├── kuromi/ → 467 img
│   │   ├── hello_kitty/ → 320 img
│   │   ├── my_melody/ → 280 img
│   │   ├── pompompurin/ → 260 img
│   │   ├── keroppi/ → 230 img
│   │   ├── badtz_maru/ → 210 img
│   │   ├── chococat/ → 240 img
│   │   ├── pochacco/ → 220 img
│   │   ├── tuxedo_sam/ → 200 img
│   │   ├── pekkle/ → 145 img
│   │   ├── hangyodon/ → 167 img
│   │   ├── little_twin_stars/ → 168 img
│   │   ├── cogimyun/ → 152 img
│   │   ├── my_sweet_piano/ → 94 img
│   │   ├── hanamaruobake/ → 144 img
│   │   ├── wish_me_mell/ → 128 img
│   │   ├── usahana/ → 157 img
│   │   ├── gaopowerroo/ → 102 img
│   │   ├── kuririn/ → 97 img       (clase más chica)
│   │   ├── gudetama/ → 170 img
│   │   ├── aggretsuko/ → 160 img
│   │   ├── kirimichan/ → 140 img
│   │   ├── marroncream/ → 128 img
│   │   ├── marumofubiyori/ → 152 img
│   │   ├── charmmykitty/ → 156 img
│   │   ├── dear_daniel/ → 136 img
│   │   ├── sugarbunnies/ → 120 img
│   │   └── yoshikitty/ → 114 img
│   └── test/                       # 29 clases, 1,246 imágenes
│
├── notebooks/                      # ★ Entrenamiento
│   ├── Transfer_Learning_Sanrio_v2.ipynb  # Notebook producción (SGD, CosineAnnealing, class weights, auto ONNX)
│   ├── CNN_Sanrio_v2.ipynb               # CNN desde cero (referencia)
│   ├── generar_notebook_sanrio.py         # Generador de notebooks
│   └── README_entrenamiento.md
│
└── scraping/                       # ★ Scraping
    ├── scraping_ampliacion_20.py   # Playwright + Brave → 20 nuevos personajes
    ├── scraping_multiclase.py      # Scraper 10 originales
    ├── scraping_sanrio.py          # Scraper v1 (legacy)
    ├── preprocesar_raw.py          # Validar, convertir a JPG, eliminar corruptas
    ├── split_multiclase.py         # Split 80/20 con stratified sampling
    ├── logs/scraping.log
    └── screenshots/
```

## Flujo entrenamiento → deploy

```
scraping/ampliacion_20.py → dataset/raw/ (descarga imágenes)
    → preprocesar_raw.py (validar, convertir JPG, limpiar)
    → split_multiclase.py (80/20 train/test)
    → dataset/train/ + dataset/test/ (29 clases, 6,160 img)
    → notebook Transfer_Learning_Sanrio_v2.ipynb
    → modelos/tl_sanrio_final.pth + modelos/clases_sanrio.json
    → export_onnx.py (FP32 + INT8)
    → models/tl_sanrio_int8.onnx + models/clases_sanrio.json
    → app/model.py lo carga en inferencia
    → uvicorn app.main:app → http://localhost:8000
    → Dockerfile → HuggingFace Spaces (port 7860)
```

## Stack tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Modelo | ResNet18 Transfer Learning → ONNX Runtime (CPU) |
| Backend | FastAPI + Uvicorn + SlowAPI (rate limit) |
| Frontend | HTML/CSS/JS vanilla (pastel goth) |
| Scraping | Playwright + Brave browser |
| Preprocesado | PIL + scikit-learn (split) |
| Entrenamiento | PyTorch + torchvision |
| Deploy | HuggingFace Spaces (Docker) |
| Chatbot | Google Gemini API (proxy backend) |

## Las 30 clases

| # | Personaje | Slug | Imágenes train |
|--:|-----------|------|:----------:|
| 1 | Hello Kitty | hello_kitty | 320 |
| 2 | My Melody | my_melody | 280 |
| 3 | Kuromi | kuromi | 467 |
| 4 | Cinnamoroll | cinnamon | 474 |
| 5 | Pompompurin | pompompurin | 260 |
| 6 | Keroppi | keroppi | 230 |
| 7 | Badtz-Maru | badtz_maru | 210 |
| 8 | Chococat | chococat | 240 |
| 9 | Pochacco | pochacco | 220 |
| 10 | Tuxedo Sam | tuxedo_sam | 200 |
| 11 | Pekkle | pekkle | 145 |
| 12 | Hangyodon | hangyodon | 167 |
| 13 | Little Twin Stars | little_twin_stars | 168 |
| 14 | Cogimyun | cogimyun | 152 |
| 15 | My Sweet Piano | my_sweet_piano | 94 |
| 16 | Hanamaruobake | hanamaruobake | 144 |
| 17 | Wish me mell | wish_me_mell | 128 |
| 18 | Usahana | usahana | 157 |
| 19 | Gaopowerroo | gaopowerroo | 102 |
| 20 | Kuririn | kuririn | 97 |
| 21 | Gudetama | gudetama | 170 |
| 22 | Aggretsuko | aggretsuko | 160 |
| 23 | Kirimichan | kirimichan | 140 |
| 24 | Marroncream | marroncream | 128 |
| 25 | Marumofubiyori | marumofubiyori | 152 |
| 26 | Charmmy Kitty | charmmykitty | 156 |
| 27 | Dear Daniel | dear_daniel | 136 |
| 28 | Sugarbunnies | sugarbunnies | 120 |
| 29 | Yoshikitty | yoshikitty | 114 |
| 30 | Hello Mimmy* | hello_mimmy | — |

*hello_mimmy no se pudo scrapear (slot preservado, agregar después sin cambios de código)
