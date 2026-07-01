# Sanrio Multiclase — Clasificador de Personajes

Proyecto de clasificacion multiclase de 8 personajes Sanrio usando:

- **CNN desde cero** (para entender como funcionan las capas internas)
- **Transfer Learning con ResNet18** (para obtener mayor precision)

**Personajes:** hello_kitty, my_melody, pompompurin, keroppi, badtz_maru, chococat, tuxedo_sam, pochacco

> NOTA: Kuromi y Cinnamoroll tienen su propio proyecto binario en `../Kuromi_vs_Cinnamoroll/`.

---

## Prerrequisitos

- Python 3.8+ con el entorno `diplomado-redes` (conda)
- Paquetes: `torch`, `torchvision`, `matplotlib`, `scikit-learn`, `pillow`, `playwright`
- VSCode con extensiones **Python** y **Jupyter**
- GPU: NVIDIA RTX 4060 (8GB) con CUDA 12.1
- Brave Browser (para el scraper) instalado en `C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe`

### Verifica tu entorno

```powershell
conda activate diplomado-redes
python --version
pip list | findstr torch
pip list | findstr playwright
```

Si falta `playwright`, instalalo y configura Brave:

```powershell
pip install playwright
playwright install chromium
```

---

## Flujo de trabajo — Paso a paso

### Paso 1: Scraping — Descargar imagenes

El scraper usa **Playwright + Brave** para buscar imagenes en Google Images.
Descarga ~240 imagenes por personaje en `dataset/raw/`.

**Antes de ejecutar**, verifica que Brave este instalado en la ruta esperada:
`C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe`

Si tu Brave esta en otra ruta, edita `scraping/scraping_multiclase.py` y cambia
la variable `BRAVE_PATH`.

```powershell
cd D:\Diplomado-RNA\Modulo-4\Proyectos\Sanrio_Multiclase
python scraping/scraping_multiclase.py
```

**Que hace el scraper:**

```
Por cada personaje (8):
  1. Abre Brave y navega a Google Images
  2. Busca "sanrio <personaje> figure merch peluche"
  3. Hace scroll hasta recolectar ~240 imagenes
  4. Descarga a dataset/raw/<personaje>/
  5. Cierra Brave y pasa al siguiente personaje
```

> La descarga toma ~5-10 min por personaje (~1-2 horas total).
> Si se interrumpe, puedes reanudar ejecutando otra vez:
> las carpetas ya existentes se saltan automaticamente.
>
> El scraper esta configurado para filtrar figuras fisicas
> (peluches, figuras, llaveros) evitando dibujos 2D.

#### Solucion de problemas del scraper

| Problema | Solucion |
|----------|----------|
| Brave no se abre | Verifica `BRAVE_PATH` en el script |
| "playwright chromium no instalado" | `playwright install chromium` |
| Descarga 0 imagenes | Revisa conexion a internet. Prueba manual: abre google.com |
| Google bloquea | Espera 5 min, reintenta. Cambia las queries en el script |
| Scraper se traba en "Buscando..." | Presiona Ctrl+C, espera 10s, vuelve a ejecutar |

---

### Paso 2: Limpieza manual (opcional pero recomendada)

Abre `dataset/raw/` en el explorador de archivos.
Para cada personaje, revisa y **borra** imagenes que no correspondan:

- Fotos de otro personaje
- Dibujos animados / fanart 2D (solo queremos figuras fisicas)
- Imagenes corruptas o muy pequenas (< 50x50)
- Duplicados evidentes

**Sugerencia:** si ves que alguna clase tiene muchas malas,
dale prioridad a esa. El modelo solo aprende de lo que le des.

---

### Paso 3: Preprocesar y dividir

```powershell
# 3a) Validar integridad, convertir a JPEG, filtrar corruptas/pequenas
python scraping/preprocesar_raw.py

# 3b) Dividir 80% train / 20% test
python scraping/split_multiclase.py
```

Resultado esperado:

```
hello_kitty: 200 validas, 3 corruptas, 12 convertidas, 5 muy pequenas
my_melody: 190 validas, ...
...
Split completado:
  train/hello_kitty: 160 imagenes
  test/hello_kitty: 40 imagenes
  ...
```

---

### Paso 4: Abrir los notebooks en VSCode

1. Abre VSCode
2. `File > Open Folder...` y selecciona `Sanrio_Multiclase/`
3. En el explorador de archivos, da clic en:
   - `CNN_Sanrio_v1.ipynb` — clasificacion CNN desde cero
   - `Transfer_Learning_Sanrio_v1.ipynb` — Transfer Learning con ResNet18
4. Selecciona kernel: `Python 3` (entorno `diplomado-redes`)
   - Si no aparece: `Ctrl+Shift+P` > `Python: Select Interpreter` > busca el de conda

### Paso 5: Ejecutar celdas

Corre las celdas una por una en orden. Los notebooks:

- Detectan automaticamente cuantas clases hay en las carpetas
- Se adaptan al numero de clases sin cambiar codigo
- Muestran explicaciones, graficas y metricas en cada paso
- Guardan los modelos entrenados en `modelos/`

> **Tips:**
> - Si el kernel muere por memoria (OOM), baja `BATCH_SIZE` a 8 o 4
> - CNN desde cero: ~55-65% accuracy esperado (dataset chico)
> - Transfer Learning: ~70-90% accuracy esperado (depende de cuantas imagenes por clase)
> - Entre celda y celda de entrenamiento, el notebook limpia la GPU automaticamente
> - Si ves `KMP_DUPLICATE_LIB_OK` en la salida, es normal (Windows + PyTorch)

---

### Paso 6: Ver resultados

Los modelos guardados aparecen en `modelos/`:

```
modelos/
├── cnn_sanrio_final.pth           # CNN desde cero (pesos finales)
├── tl_sanrio_final.pth            # Transfer Learning (pesos finales)
├── clases_sanrio.json             # Mapeo de clases
├── experimentos_cnn.json          # Bitacora de experimentos CNN
└── experimentos_tl.json           # Bitacora de experimentos TL
```

---

## Estructura del proyecto

```
Sanrio_Multiclase/
├── CNN_Sanrio_v1.ipynb                # Notebook: CNN desde cero (ENTREGABLE 1)
├── Transfer_Learning_Sanrio_v1.ipynb  # Notebook: Transfer Learning (ENTREGABLE 2)
├── generar_notebook_sanrio.py         # Generador de ambos notebooks
├── README.md
├── dataset/
│   ├── raw/                           # Imagenes descargadas (sin procesar)
│   │   ├── hello_kitty/
│   │   ├── my_melody/
│   │   ├── pompompurin/
│   │   ├── keroppi/
│   │   ├── badtz_maru/
│   │   ├── chococat/
│   │   ├── tuxedo_sam/
│   │   └── pochacco/
│   ├── train/                         # 80% (para entrenar)
│   │   ├── hello_kitty/
│   │   ├── my_melody/
│   │   └── ...
│   └── test/                          # 20% (para evaluar)
│       ├── hello_kitty/
│       ├── my_melody/
│       └── ...
├── modelos/                           # Pesos entrenados (.pth) y logs
└── scraping/
    ├── scraping_multiclase.py         # Busca y descarga imagenes via Brave + Playwright
    ├── preprocesar_raw.py             # Valida integridad, convierte a JPEG
    └── split_multiclase.py            # Divide 80/20 en train/test
```

---

## Solucion de problemas

| Problema | Causa | Solucion |
|----------|-------|----------|
| `'python' no se reconoce` | Python no esta en PATH | Usa `py` o la ruta completa del entorno conda |
| `ModuleNotFoundError: playwright` | No instalado | `pip install playwright && playwright install chromium` |
| `ModuleNotFoundError: torch` | Entorno no activado | `conda activate diplomado-redes` |
| VSCode no abre `.ipynb` | Falta extension Jupyter | Instalar "Jupyter" de Microsoft desde Extensiones |
| Kernel muere al entrenar | Out Of Memory (OOM) | Abre el notebook, busca `BATCH_SIZE` y cambialo a 8 o 4 |
| GPU no detectada | CUDA no instalado | `python -c "import torch; print(torch.cuda.is_available())"` |
| Scraping descarga 0 imagenes | Google bloquea | Reintenta mas tarde, o cambia las queries de busqueda |
| Kernel crash 0xC000041D | FATAL_APP_EXIT por CUDA | Ya mitigado con cleanup automatico entre celdas |

### Comandos de diagnostico rapido

```powershell
conda info --envs
python -c "import torch; print('GPU:', torch.cuda.is_available())"
python -c "import torch; print('CUDA:', torch.version.cuda)"
python -c "from playwright.sync_api import sync_playwright; print('Playwright OK')"
```

---

## Entregables para la tarea

Los dos archivos que debes adjuntar:

1. **`CNN_Sanrio_v1.ipynb`** — implementacion CNN desde cero
2. **`Transfer_Learning_Sanrio_v1.ipynb`** — implementacion con Transfer Learning

Ambos se entregan en formato `.ipynb`. La profesora solo necesita
los notebooks; el dataset y los modelos son locales.

Los notebooks incluyen:
- Narrativa en espanol explicando cada paso
- Celdas de codigo ejecutables
- Graficas de entrenamiento (loss, accuracy)
- Matriz de confusion + accuracy por clase
- Bitacora de experimentos
