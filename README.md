# Diplomado-RNA

## Diplomado Superior en Redes Neuronales Artificiales y Deep Learning — UAEM

Este repositorio contiene todos los ejercicios, notas, proyectos y recursos generados durante el diplomado, organizado por módulo.

---

## 🗂️ Estructura del repositorio

```
Diplomado-RNA/
│
├── Enviroment/                     # Configuración del entorno conda
│   ├── environment.yml             # Entorno completo para todos los módulos
│   ├── diplomado_rna_python.yaml   # Variante del entorno
│   └── HowTo.txt                   # Instrucciones de instalación paso a paso
│
├── Machote/                        # Recursos reutilizables (importables desde cualquier notebook)
│   ├── machote_ML.py               # Librería de funciones de ML (cargar, explorar, evaluar)
│   ├── machote_ML.ipynb            # Versión documentada del machote para estudiar
│   ├── glosario_python_ML.ipynb    # Referencia de métodos y funciones de Python/pandas/numpy
│   └── plantilla_ejercicio.ipynb   # Plantilla base para cualquier ejercicio nuevo
│
├── Modulo-1/                       # Fundamentos de Inteligencia Artificial
│   └── CalendarioDSRNDL2026.jpeg
│
├── Modulo-2/                       # Python para Inteligencia Artificial
│   ├── Ejercicios/                 # Ejercicios por tema
│   │   ├── Tema1/                  # Fundamentos: funciones, variables, strings
│   │   ├── Tema2/                  # Operadores aritméticos, lógicos, comparación
│   │   └── Tema5/                  # Neurona MP y Perceptrón
│   ├── Lecturas/                   # Documentos de referencia (f-strings, etc.)
│   ├── Notebooks/                  # Notebooks de clase por tema
│   │   ├── DataSets/               # Datasets locales (cancer, fútbol europeo)
│   │   ├── Machote_ML/             # Copia del machote para Módulo 2
│   │   ├── Presentaciones/         # Diapositivas del curso
│   │   ├── Tema1/                  # Comentarios, funciones, strings, variables
│   │   ├── Tema2/                  # Operadores booleanos, aritmético, identidad
│   │   ├── Tema3/                  # Listas, tuplas, diccionarios, sets
│   │   ├── Tema4/                  # NumPy, Pandas, Matplotlib, Regresión Lineal
│   │   └── Tema5/                  # Neurona McCulloch-Pitts, Perceptrón
│   └── Programas/                  # Scripts Python por tema
│       ├── Tema_0/                 # Hello World y pruebas iniciales
│       ├── Tema_1/                 # Prácticas de funciones, strings, variables
│       ├── Tema_2/                 # Prácticas de operadores
│       └── Proyectos/              # Proyectos evaluados
│           ├── Tema1/              # GamerTags — generador de tags para jugadores
│           └── Tema2/              # Calculadora de métricas corporales (IMC, calorías)
│
└── Modulo-3/                       # Machine Learning (en progreso)
    ├── Clases/                     # Notebooks de clase
    ├── Ejercicios/                 # Ejercicios y plantillas
    ├── Machote/                    # Pipelines y machotes del módulo
    │   ├── Funciones_Activacion.ipynb
    │   ├── Machote_Modulo3.ipynb
    │   ├── Perceptron_machote.ipynb
    │   ├── Pipeline_RegresionLineal.ipynb
    │   ├── Pipeline_RegresionLogistica.ipynb
    │   ├── Pipeline_ArbolDecision.ipynb
    │   ├── Pipeline_SVM.ipynb
    │   └── Pipeline_KMeans.ipynb
    └── Tareas/                     # Tareas entregadas
```

---

## ⚙️ Instalación del entorno

### Requisitos previos
- [Anaconda](https://www.anaconda.com/download) o Miniconda instalado
- Git instalado
- VSCode con la extensión de Jupyter

### Pasos (ver también `Enviroment/HowTo.txt`)

```bash
# 1. Clonar el repositorio
git clone https://github.com/MorritaConP1to/Diplomado-RNA.git
cd Diplomado-RNA

# 2. Crear el entorno conda desde el archivo
conda env create -f Enviroment/environment.yml

# 3. Activar el entorno
conda activate diplomado-redes

# 4. Registrar como kernel en VSCode
python -m ipykernel install --user --name diplomado-redes --display-name "Python (diplomado-redes)"
```

### Si hay errores de compatibilidad con seaborn/matplotlib

```bash
conda activate diplomado-redes
pip install --upgrade seaborn
# Si persiste:
pip install "seaborn>=0.13.0" "matplotlib>=3.7,<3.9"
```

Después reiniciar el kernel en VSCode con el botón **Restart**.

### Versión de Python
- Escuela: Python 3.11.14
- Compatible con versiones recientes (3.10+)

---

## 📓 ¿Cómo usar el machote?

El `machote_ML.py` es una librería de funciones reutilizables que evita copiar código repetido entre notebooks.

**Importar desde cualquier notebook:**

```python
import sys, os

# Ruta relativa — funciona en Windows y Linux sin cambiar nada
RUTA = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'Machote'))
if RUTA not in sys.path:
    sys.path.append(RUTA)

from machote_ML import *
```

**Funciones disponibles:**
- `cargar_desde_sklearn()`, `cargar_desde_uci()`, `cargar_desde_csv()` — cargar datos
- `resumen_rapido()`, `ver_correlacion_con_y()`, `ver_mapa_calor()` — exploración
- `seleccionar_features()`, `dividir_datos()`, `escalar_datos()` — preparación
- `evaluar_clasificacion()`, `evaluar_regresion()`, `comparar_modelos()` — evaluación

---

## 🗂️ Proyectos del Módulo 2

### Proyecto 1 — GamerTag Generator (`Modulo-2/Programas/Proyectos/Tema1/`)

Genera diferentes estilos de GamerTags a partir del nombre, apellido y número favorito del jugador.

```bash
python GamerTags-V2.py
```

**Tipos de tags generados:**
- Básico (primeras 4 letras)
- Invertido (nombre al revés)
- Intercalado (combinación nombre + apellido)
- Élite (primeras 2 + últimas 2 letras)
- Con número (primeras 5 letras + número favorito)

**Dependencias:**
```bash
pip install pyfiglet tabulate colorama
```

### Proyecto 2 — Calculadora de Métricas Corporales (`Modulo-2/Programas/Proyectos/Tema2/`)

Calcula indicadores de salud y fitness personales.

```bash
python Calculadora-V2.py
```

**Métricas calculadas:**
- IMC (Índice de Masa Corporal) con clasificación
- Calorías diarias recomendadas (Fórmula de Harris-Benedict)
- Litros de agua recomendados al día
- Ritmo cardíaco máximo

---

## 📊 Contenido por módulo

| Módulo | Estado | Temas principales |
|--------|--------|------------------|
| Módulo 1 — Fundamentos IA | ✅ Completado | Definiciones, tipos de IA, ética |
| Módulo 2 — Python para IA | ✅ Completado | Python, NumPy, Pandas, Matplotlib, Neurona MP, Perceptrón |
| Módulo 3 — Machine Learning | 🔄 En progreso | Regresión, Clasificación, Clustering, Refuerzo |
| Módulo 4 — Deep Learning | ⏳ Pendiente | CNN, RNN, Transformers, Keras/TensorFlow |
| Módulo 5 — Sistemas Embebidos | ⏳ Pendiente | Arduino, IA simbólica y conexionista |

---

## 📌 Notas importantes

- Los datasets de UCI se descargan en tiempo de ejecución — se requiere conexión a internet
- Los datasets locales están en `Modulo-2/Notebooks/DataSets/`
- El archivo `mapa_diplomado.ipynb` en `Machote/` contiene una guía completa de todo el diplomado
- Los notebooks con `_blank` en el nombre son plantillas de clase sin resolver
- Los notebooks con `_V4`, `_v2`, `mejorado` son versiones corregidas y comentadas

---

## 🔗 Recursos útiles

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Scikit-learn Docs](https://scikit-learn.org/stable/documentation.html)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Keras/TensorFlow Docs](https://keras.io/)
- [Generador de títulos ASCII](https://patorjk.com/software/taag/)
