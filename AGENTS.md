# AGENTS.md — Guía para asistentes de IA

## Descripción del proyecto

Diplomado Superior en Redes Neuronales Artificiales y Deep Learning — UAEM.
Repositorio con 5 módulos que cubren desde fundamentos de Python hasta TinyML,
organizados con estructura uniforme.

## Estructura de carpetas

```
Diplomado-RNA/
├── Machote/           # Librería compartida machote_ML.py (reutilizable entre módulos)
├── Modulo-1/          # Introducción a la IA (teórico + math review)
├── Modulo-2/          # Python para IA (sintaxis, NumPy, Pandas, MP Neuron)
├── Modulo-3/          # Machine Learning (Perceptrón, RL, Árboles, SVM, K-Means)
├── Modulo-4/          # Deep Learning (MLP, CNN, RNN/LSTM, Keras/TF)
│   └── Proyectos/     # Proyectos: Reconocimiento_Digitos, Kuromi_vs_Cinnamoroll
├── Modulo-5/          # Sistemas Embebidos (Arduino, TinyML, compuertas)
├── AUTORIA.md         # Creditos y specs del equipo de desarrollo
├── Enviroment/        # Entorno conda (environment.yml)
└── README.md
```

Cada módulo sigue el mismo patrón:
- `libreria_moduloN.py` → funciones importables
- `cuadernillo_moduloN.ipynb` → referencia rápida
- `machote_moduloN.ipynb` → template con ejercicios
- `moduloN_estudio_general.ipynb` → resumen ejecutivo
- `Estudio/` → notebooks de estudio por tema
- `Pipelines/` → pipelines reutilizables
- `Machotes/` → templates adicionales
- `Material/` → datasets, imágenes, presentaciones
- `Proyectos/` → (Módulo-4) proyectos integradores

## Convenciones de código

### Librerías (libreria_moduloN.py)
- Detección de plataforma: `'google.colab' in sys.modules`
- Reproducibilidad: `np.random.seed(42)`
- Funciones con docstring en español
- Organizadas en secciones numeradas

### Notebooks
- Celdas `# MACHOTE` → código reutilizable
- Celdas `# TODO` → ejercicios para completar
- Primer celda: import de `libreria_moduloN`
- Texto en español, términos técnicos en inglés
- `execution_count: null` en notebooks template

### Estructura de import
```python
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from libreria_moduloN import *
```

## Cómo ejecutar

1. Crear entorno: `conda env create -f Enviroment/environment.yml`
2. Activar: `conda activate diplomado-redes`
3. Abrir notebook: `jupyter notebook` o desde VSCode

## Comandos útiles

```bash
git status                    # ver cambios
git log --oneline -10         # ver últimos commits
conda activate diplomado-redes  # activar entorno
jupyter notebook              # iniciar Jupyter
```
