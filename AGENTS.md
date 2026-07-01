# AGENTS.md — Guía para asistentes de IA

## Descripción del proyecto

Diplomado Superior en Redes Neuronales Artificiales y Deep Learning — UAEM.
Repositorio con 5 módulos que cubren desde fundamentos de Python hasta TinyML,
organizados con estructura uniforme.

## Estructura de carpetas

```
Diplomado-RNA/
├── Machote/           # Librería compartida machote_ML.py
├── Modulo-1/          # Introducción a la IA (teórico + math review)
├── Modulo-2/          # Python para IA (sintaxis, NumPy, Pandas, MP Neuron)
├── Modulo-3/          # Machine Learning (Perceptrón, RL, Árboles, SVM, K-Means)
├── Modulo-4/          # Deep Learning (MLP, CNN, RNN/LSTM, Keras/TF)
│   └── Proyectos/     # Proyectos: Reconocimiento_Digitos, Kuromi_vs_Cinnamoroll
│       ├── Sanrio_Multiclase/  # ← PROYECTO FINAL activo
│       └── Final/              # Rúbrica del proyecto
├── Modulo-5/          # Sistemas Embebidos (Arduino, TinyML, compuertas)
├── AUTORIA.md         # Creditos y specs del equipo de desarrollo
├── Enviroment/        # Entorno conda (environment.yml)
└── README.md
```

---
# ============================================================
# PROYECTO FINAL: Clasificador Web Sanrio (30 clases)
# ============================================================

## Stack tecnológico
- **Modelo**: ResNet18 Transfer Learning → ONNX Runtime (CPU)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: HTML/CSS/JS vanilla (identidad visual Sanrio pastel goth)
- **Deploy**: HuggingFace Spaces (Docker)
- **Extra**: Chatbot Gemini con system prompt restrictivo

## Rúbrica (100 pts base + 10 extra)
| Criterio | Pts | Clave |
|----------|:---:|-------|
| A. Núcleo Clasificador | 25 | Modelo funcional + manejo de errores |
| B. Integración Web | 20 | Frontend ⇄ Backend comunicación limpia |
| C. Estética UI/UX Única | 15 | Diseño pastel goth Sanrio (NO plantilla genérica) |
| D. Despliegue Hosting | 15 | URL pública funcional (HF Spaces) |
| E. Reporte Técnico + Código | 25 | PDF con justificación stack + código comentado |
| Extra: Chatbot Gemini | +10 | System prompt robusto, limitado a Sanrio |

## Las 30 clases definitivas

### Top 20 Ranking 2026 Sanrio (los 11 que faltan de los 10 actuales)
```
pekkle, hangyodon, little_twin_stars, cogimyun, my_sweet_piano,
hanamaruobake, wish_me_mell, usahana, gaopowerroo, kuririn, gudetama
```

### Los 10 actuales (ya entrenados)
```
badtz_maru, chococat, cinnamon, hello_kitty, keroppi, kuromi,
my_melody, pochacco, pompompurin, tuxedo_sam
```

### 9 adicionales para completar 30
```
aggretsuko, kirimichan, marroncream, marumofubiyori, charmmykitty,
dear_daniel, sugarbunnies, yoshikitty, hello_mimmy
```

## Historial de experimentos

| Version | Arquitectura | Dataset | Clases | Train | Test | Best Acc | Problema |
|---------|-------------|--------|:------:|:----:|:----:|:--------:|----------|
| v1 | CNN desde cero | 10 originales | 10 | 2,203 | 557 | ~82% | Sin transfer learning |
| v2 | ResNet18 + 2 capas | 29 original (ruidoso) | 29 | 4,914 | 1,246 | **65.49%** | Demasiadas clases similares, dataset contaminado |
| v3 | EfficientNet-B0 + 1 capa | v3 (12 curadas) | 12-26 | 3,957 | 989 | **66.43%** | Cabeza simple, batch 32, sin limpiar |
| v4 | ResNet18 + 2 capas | v4 (12 limpias) | 12 | 2,673 | 672 | **88.54%** | **MEJOR MODELO** |
| v5.1 (MixUp) | ResNet18 + MixUp α=0.2 | v4 | 12 | 2,673 | 672 | 87.50% | MixUp añadió ruido |
| **v5.2 (FocalLoss)** | **ResNet18 + FocalLoss γ=2.0** | **v4** | **12** | **2,673** | **672** | **88.24%** | **No superó v4** |

### Lo que se descartó (no funciona)
- **EfficientNet-B0 + cabeza de 1 capa**: La cabeza lineal no captura diferencias sutiles entre personajes similares. My Melody cayó de 78.6% → 31.7%.
- **Batch size 32**: Reduce la estocasticidad necesaria para conjuntos pequeños (<300 img/clase).
- **Dataset 29 clases**: Demasiadas clases visualmente idénticas (pink/white round faces) para solo ~150 img/clase.

### Lo que sí funciona
- **ResNet18 + cabeza de 2 capas** (Linear→ReLU→Dropout→Linear): Aprende combinaciones no-lineales de features.
- **Label smoothing η=0.1**: Reduce sobreconfianza en clases similares.
- **SGD + Nesterov momentum**: Generaliza mejor que Adam con datasets chicos.
- **Batch size 16**: Balance entre estocasticidad y eficiencia.
- **3 fases de fine-tuning** (head → layer4 → layer3+layer4): Transición gradual que preserva features de ImageNet.
- **Dataset curado**: Limpiar duplicados, leakage y <200px mejora la calidad sin añadir imágenes.

## Fases del proyecto (actualizado)

| Fase | Fechas | Qué incluye |
|------|--------|-------------|
| Fase 0 | 30 jun–1 jul | Scraping 20 nuevos personajes |
| Fase 0.5 | 1 jul | Preprocesar, limpiar dataset v4, reentrenar v4 |
| Fase 1 | 1–4 jul | Backend FastAPI + export ONNX + validaciones defensivas |
| Fase 2 | 5–8 jul | Frontend HTML/CSS/JS + Chatbot Gemini |
| Fase 3 | 9–11 jul | Deploy HF Spaces + Reporte PDF + pruebas finales |

## Estructura del proyecto (web app)

```
Modulo-4/Proyectos/
├── Final/                         # ← Web App (deploy)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI (rutas /predict, /chat, /health)
│   │   ├── model.py               # Carga ONNX + inferencia + top-3
│   │   ├── schemas.py             # Pydantic models
│   │   ├── config.py              # Thresholds, rutas
│   │   ├── gemini_chat.py         # Proxy Gemini + system prompt
│   │   ├── utils.py               # Validación imágenes
│   │   └── static/
│   │       ├── index.html         # Frontend pastel goth
│   │       └── personajes.json    # Datos de 30 personajes
│   ├── models/
│   │   ├── clases_sanrio.json     # Mapeo clases
│   │   ├── tl_sanrio_int8.onnx    # Modelo ONNX INT8 (11.4 MB)
│   │   └── export_onnx.py         # Export .pth → ONNX
│   ├── modelos/                   # Pesos .pth (para reentrenar)
│   ├── dataset/                   # train_v4/, test_v4/, raw_v3/, raw/
│   ├── scraping/                  # Scrapers + preprocesamiento + limpieza
│   ├── notebooks/                 # Notebooks de entrenamiento (v1-v4)
│   ├── Dockerfile
│   ├── requirements.txt              # Full (con torch)
│   ├── requirements-deploy.txt       # Ligero (sin torch, para Docker)
│   ├── .env.example                  # Template GEMINI_API_KEY
│   ├── STRUCTURE.md                  # Árbol completo del proyecto
│   └── README.md
├── Sanrio_Multiclase/             # ← Solo respaldo (ya no se usa)
│   ├── dataset/                   # train/, test/, raw/
│   ├── modelos/                   # Pesos .pth, experimentos
│   └── scraping/                  # Scrapers + preprocesamiento
```

### Nota: proyecto unificado en Final/
- **Sanrio_Multiclase** queda como respaldo / referencia
- **Final** contiene TODO: scraping, dataset, entrenamiento, web app y deploy

## Vectores de ataque y contramedidas

| Vector | Riesgo | Contramedida |
|--------|--------|-------------|
| Subir perro/paisaje/persona | Clasifica como Sanrio con baja confianza | Threshold < 30% → "No parece personaje Sanrio conocido" |
| Imagen corrupta / .txt renombrado | Crash del servidor | Try/except + validar con PIL antes de inferir |
| Imagen 10x10 o 10000x10000 | Error de memoria/dimensiones | Validar 32x32 < img < 4000x4000 |
| Jailbreak al chatbot | Responder temas no-Sanrio | System prompt con instrucciones anti-jailbreak + rejection template |
| Inspeccionar network tab | Ver API key de Gemini | API key SOLO en backend (variable de entorno), frontend llama a /chat |
| Spamear requests | Server timeout | Rate limiting (slowapi, 100 req/min/IP) |
| Subir Sanrio raro (fuera de 30) | Falsa detección "no Sanrio" | Respuesta: "No identificado. Podría ser un Sanrio no cubierto o no serlo" |
| Subir imagen sin extensión | Error 422 feo | FastAPI maneja automático + mensaje personalizado |
| DAN/roleplay attack | Chatbot ignora filtros | System prompt con defensas: "ignore previous", "eres ahora X", "para propósitos académicos" |

## Cómo ejecutar (local)
```powershell
# Entorno
conda activate diplomado-redes

# Dataset v4 (12 clases limpias, desde cero)
cd Modulo-4/Proyectos/Final

# Opcion A: Usar dataset existente v3 → v4 (recomendado)
python scraping/scraping_12clases.py     # ~4-6h, genera raw_v3/
python scraping/preprocesar_raw_v3.py    # limpia corruptas
python scraping/split_12clases.py        # train_v3/ + test_v3/
python scraping/limpiar_dataset_v4.py    # v3 → v4 limpio

# Opcion B: Ejecutar notebook directamente
# Configurar DATASET_VERSION = 'v4' en el notebook

# Entrenar (v4 - mejor modelo, 88.54%)
jupyter notebook notebooks/Transfer_Learning_Sanrio_v4.ipynb

# Entrenar (v5 - experimentos optimizacion)
python notebooks/train_v5.py                # FocalLoss
jupyter notebook notebooks/Transfer_Learning_Sanrio_v5.ipynb

# Iniciar servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Abrir en http://localhost:8000
```

## Comandos útiles
```
git status                    # ver cambios
git log --oneline -10         # ver últimos commits
uvicorn app.main:app --reload # iniciar servidor dev (desde Final/)
```

## Receta de optimización para TL Multiclase

Usar cuando Transfer Learning con ResNet18 se queda en ~70% con datasets pequeños (<200 img/clase, 10 clases).

### Resultados de optimización (v5)

| Experimento | Cambio | Accuracy | vs v4 |
|-------------|--------|:--------:|:-----:|
| v4 (baseline) | CE + label_smoothing + dropout 0.3 + 20ep Fase3 | **88.54%** | - |
| v5.1 (MixUp) | + MixUp α=0.2 en Fases 1&2 | 87.50% | -1.04% |
| v5.2 (FocalLoss) | FocalLoss γ=2.0 + dropout 0.5 + 30ep Fase3 | **88.24%** | -0.30% |

### Lecciones aprendidas
- **MixUp no ayuda** con ~200 img/clase (introduce ruido que no se compensa)
- **FocalLoss** no mejora vs CrossEntropy + label smoothing a esta escala
- **Dropout 0.5** fue excesivo; 0.3 es mejor para este tamaño de dataset
- **Más épocas** no necesariamente ayudan (early stopping cortó en época 16)
- **v4 es la configuración óptima** para 12 clases Sanrio
- La calidad del dataset (limpieza, sin leakage) importa más que la arquitectura
