# Guia de Deploy — Clasificador Sanrio en Hugging Face Spaces

## Archivos necesarios

Ya estan empaquetados en: `Modulo-4/Proyectos/Final/_deploy_hf/`
Contiene SOLO lo necesario (11 MB):

```
_deploy_hf/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI (predict, chat, health)
│   ├── model.py          # ONNX inference + top-3
│   ├── schemas.py        # Pydantic
│   ├── config.py         # Thresholds, rutas
│   ├── gemini_chat.py    # Gemini proxy
│   ├── utils.py          # Validacion imagenes
│   └── static/
│       ├── index.html    # Frontend pastel goth
│       └── personajes.json  # 30 personajes
├── models/
│   ├── tl_sanrio_int8.onnx  # 11.4 MB (INT8 cuantizado)
│   └── clases_sanrio.json   # 12 clases
├── Dockerfile            # Puerto 7860 (HF)
├── requirements-deploy.txt  # Sin torch
├── .env.example          # Template GEMINI_API_KEY
└── .dockerignore
```

---

## PASO 1: Autenticacion en HF

Abre PowerShell y ejecuta:

```powershell
$env:PYTHONIOENCODING='utf-8'
& "C:\Users\Darck\.local\bin\hf.exe" auth login
```

Te pedira pegar tu token. Escribelo o pegalo ahi (no queda en historial).

Si no tienes token: https://huggingface.co/settings/tokens → "Create new token" → scopes: `read` + `write`

---

## PASO 2: Crear Space en HF (web UI)

1. Ir a https://huggingface.co/spaces
2. Click "Create new Space"
3. Llenar:
   - **Space name:** `sanrio-classifier`
   - **Owner:** `MorritaConP1to`
   - **Space type:** Docker
   - **Hardware:** CPU basic (gratis)
   - **Visibility:** Public
4. Click "Create Space"

El Space se creara vacio con un Dockerfile por defecto.

---

## PASO 3: Clonar y pushear archivos

```powershell
cd D:\Diplomado-RNA

# 3a. Clonar el Space (vacio)
git clone https://huggingface.co/spaces/MorritaConP1to/sanrio-classifier
cd sanrio-classifier

# 3b. Vaciar contenido default y copiar nuestros archivos
Remove-Item -Path * -Recurse -Force
Copy-Item -Path "D:\Diplomado-RNA\Modulo-4\Proyectos\Final\_deploy_hf\*" -Destination "." -Recurse -Force

# 3c. Commit y push
git add -A
git commit -m "Deploy: Sanrio Classifier v1 (ResNet18 INT8, 12 clases)"
git push
```

Te pedira usuario y password. El password es **tu token de HF** (no tu password de HF).

---

## PASO 4: Configurar Gemini API Key

En la web de HF:

1. Ir a: https://huggingface.co/spaces/MorritaConP1to/sanrio-classifier/settings
2. Seccion "Repository Secrets"
3. Agregar:
   - **Name:** `GEMINI_API_KEY`
   - **Value:** (tu API key de Google AI Studio)
4. Click "Save"

Si no tienes API key: https://aistudio.google.com/apikey

---

## PASO 5: Monitorear el build

El build comienza automaticamente tras el push. Para ver logs:

```powershell
$env:PYTHONIOENCODING='utf-8'
& "C:\Users\Darck\.local\bin\hf.exe" spaces logs MorritaConP1to/sanrio-classifier
```

O en web: https://huggingface.co/spaces/MorritaConP1to/sanrio-classifier/tree/main
→ Ver pestaña "Builder" o "Logs"

Para esperar a que termine:

```powershell
& "C:\Users\Darck\.local\bin\hf.exe" spaces wait MorritaConP1to/sanrio-classifier
```

---

## PASO 6: Probar

URL publica: https://MorritaConP1to-sanrio-classifier.hf.space

Probar endpoints:

- `GET  /` — Frontend
- `GET  /health` — Health check
- `POST /predict` — Clasificar imagen (multipart, field: "file")
- `POST /chat` — Chatbot Gemini (JSON: `{"mensaje": "..."}`)

---

## PASO 7: Configurar sleep time

Para no gastar recursos cuando nadie lo usa:

1. Settings → Space settings
2. **Sleep time:** 300 segundos (5 min de inactividad)
3. **Resume mode:** Auto

---

## Si algo sale mal

- **Build falla:** `hf spaces logs MorritaConP1to/sanrio-classifier`
- **Error de modulos:** Verificar que `requirements-deploy.txt` tiene todo
- **Modelo no carga:** Verificar formato de `clases_sanrio.json`
- **Chat no funciona:** Verificar que `GEMINI_API_KEY` esta en Secrets
- **Rate limit:** Esperar 1 minuto o reiniciar el Space

---

## Despues del deploy exitoso

Reportame cuando este funcionando y procedemos a:

1. Probar `/predict` con algunas imagenes
2. Probar el chatbot
3. Escribir el reporte PDF
4. Celebrar ??

---

## Paquete de respaldo

Los archivos de deploy tambien quedaron en:
`D:\Diplomado-RNA\Modulo-4\Proyectos\Final\_deploy_hf\`

Si necesitas reconstruir el paquete, solo ejecuta el script:
`D:\Diplomado-RNA\Modulo-4\Proyectos\Final\notebooks\_empaquetar_deploy.ps1`
