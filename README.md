<div align="center">

# 🕸️ Diplomado Superior en RNA y Deep Learning

**Universidad Autónoma del Estado de México — UAEM**

[![Python](https://img.shields.io/badge/Python-3.10+-4A148C?style=for-the-badge&logo=python&logoColor=CE93D8)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-7B1FA2?style=for-the-badge&logo=tensorflow&logoColor=CE93D8)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-4A148C?style=for-the-badge&logo=keras&logoColor=CE93D8)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-7B1FA2?style=for-the-badge&logo=scikit-learn&logoColor=CE93D8)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-4A148C?style=for-the-badge&logo=numpy&logoColor=CE93D8)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-7B1FA2?style=for-the-badge&logo=pandas&logoColor=CE93D8)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-4A148C?style=for-the-badge&logo=jupyter&logoColor=CE93D8)](https://jupyter.org)
[![Arduino](https://img.shields.io/badge/Arduino-7B1FA2?style=for-the-badge&logo=arduino&logoColor=CE93D8)](https://arduino.cc)

<br>

> *De la neurona de McCulloch-Pitts al TinyML en microcontroladores.*  
> 5 módulos · 30+ notebooks · Python · Deep Learning · Sistemas Embebidos

<br>

[📖 Instalación](#⚙️-instalación) •
[📂 Estructura](#📂-estructura) •
[🧩 Módulos](#🧩-módulos) •
[🚀 Proyectos](#🚀-proyectos-destacados) •
[📚 Librería](#📚-librería-compartida) •
[🎀 Tips](#🎀-tips-para-empezar)

<br>

---
</div>

<br>

## 🦇 Panorama General

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   🧠 INTELIGENCIA ARTIFICIAL                                       │
│   ├── 🤖 Machine Learning                                          │
│   │    ├── M2 · Python para IA  (preprocesamiento, EDA, MP Neuron) │
│   │    ├── M3 · ML clásico      (Perceptrón, RL, SVM, Árboles)     │
│   │    └── M4 · Deep Learning   (MLP, CNN, RNN/LSTM)               │
│   ├── 🔧 M5 · Sistemas Embebidos (Arduino, TinyML)                 │
│   └── 📐 M1 · Fundamentos       (historia, tipos aprendizaje)       │
│                                                                     │
│   🎯 Progresión: M1 → M2 → M3 → M4 → M5                           │
│   Cada módulo construye sobre el anterior.                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

<br>

## ⚙️ Instalación

<details open>
<summary><b>Requisitos previos</b></summary>

- [Anaconda](https://anaconda.com/download) o Miniconda
- Git
- VSCode + extensión Jupyter

</details>

```bash
# 1. Clonar
git clone https://github.com/MorritaConP1to/Diplomado-RNA.git
cd Diplomado-RNA

# 2. Crear entorno
conda env create -f Enviroment/environment.yml

# 3. Activar
conda activate diplomado-redes

# 4. Registrar kernel
python -m ipykernel install --user --name diplomado-redes --display-name "Python (diplomado-redes)"
```

> 💡 **Python 3.10+** · Si hay conflictos con seaborn/matplotlib:  
> `pip install "seaborn>=0.13.0" "matplotlib>=3.7,<3.9"`

<br>

## 📂 Estructura

```
Diplomado-RNA/
│
├── 🏗️  Machote/              → Librería compartida (machote_ML.py)
│
├── 📐 Modulo-1/              → Introducción a la IA
│   ├── libreria_modulo1.py
│   ├── cuadernillo_modulo1.ipynb
│   ├── machote_modulo1.ipynb
│   └── Estudio/              → 5 temas (historia, matemáticas...)
│
├── 🐍 Modulo-2/              → Python para IA
│   ├── libreria_modulo2.py
│   ├── cuadernillo_modulo2.ipynb
│   ├── Estudio/              → 5 temas (variables → MP Neuron)
│   └── Pipelines/            → GamerTags, Body Metrics
│
├── 🤖 Modulo-3/              → Machine Learning
│   ├── libreria_modulo3.py
│   ├── cuadernillo_modulo3.ipynb
│   ├── Estudio/              → 7 temas (Perceptrón → Redes)
│   └── Pipelines/            → 5 pipelines (RL, SVM, K-Means...)
│
├── 🧠 Modulo-4/              → Deep Learning
│   ├── modulo4_libreria.py
│   ├── cuadernillo_modulo4.ipynb
│   ├── Estudio/              → 5 temas (Fundamentos → Plataformas)
│   └── Pipelines/            → 5 pipelines (CNN, LSTM, Despliegue...)
│
├── 🔧 Modulo-5/              → Sistemas Embebidos
│   ├── libreria_modulo5.py
│   ├── cuadernillo_modulo5.ipynb
│   ├── Estudio/              → 3 temas (Arduino, IA simbólica...)
│   └── Pipelines/            → 3 pipelines (Serial, Compuertas, TinyML)
│
└── 🌐 Enviroment/            → conda environment.yml
```

<br>

## 🧩 Módulos

| # | Módulo | Librería | Topics | Pipelines |
|:-:|--------|----------|--------|-----------|
| 📐 | **M1 — Introducción a la IA** | [`libreria_modulo1.py`](Modulo-1/libreria_modulo1.py) | Historia, tipos de aprendizaje, vectores, matrices, derivadas | — |
| 🐍 | **M2 — Python para IA** | [`libreria_modulo2.py`](Modulo-2/libreria_modulo2.py) | Sintaxis, NumPy, Pandas, Matplotlib, MP Neuron, Boolean Trick | [GamerTags](Modulo-2/Pipelines/GamerTags/), [Body Metrics](Modulo-2/Pipelines/BodyMetrics/) |
| 🤖 | **M3 — Machine Learning** | [`libreria_modulo3.py`](Modulo-3/libreria_modulo3.py) | Perceptrón, Regresión Lineal/Logística, SVM, Árboles, K-Means | [5 pipelines](Modulo-3/Pipelines/) (RL → SVM) |
| 🧠 | **M4 — Deep Learning** | [`modulo4_libreria.py`](Modulo-4/modulo4_libreria.py) | MLP, CNN, RNN/LSTM, Keras/TF, Backpropagation, Hiperparámetros | [5 pipelines](Modulo-4/Pipelines/) (CNN, LSTM, Despliegue...) |
| 🔧 | **M5 — TinyML** | [`libreria_modulo5.py`](Modulo-5/libreria_modulo5.py) | Arduino, compuertas lógicas, Perceptrón, TinyML export | [3 pipelines](Modulo-5/Pipelines/) (Serial, Compuertas, TinyML) |

<details>
<summary><b>📊 Estadísticas del proyecto</b></summary>

| Métrica | Valor |
|---------|-------|
| Total módulos | 5 |
| Notebooks de estudio | 25+ |
| Pipelines reutilizables | 16 |
| Funciones en librerías | 150+ |
| Lenguaje | Python 3.10+ |
| Frameworks | TensorFlow, Keras, scikit-learn |
| Plataformas soportadas | Google Colab + Local (VSCode) |

</details>

<br>

## 🚀 Proyectos Destacados

<details>
<summary><b>🎮 GamerTag Generator</b> — <code>Modulo-2/Pipelines/GamerTags/</code></summary>

Genera estilos de GamerTags a partir de nombre, apellido y número favorito.

```bash
python GamerTags-V2.py
```

**Estilos:** Básico · Invertido · Intercalado · Élite · Con número  
**Deps:** `pip install pyfiglet tabulate colorama`

> 🎀 *GIF próximamente...*
</details>

<details>
<summary><b>📏 Body Metrics Calculator</b> — <code>Modulo-2/Pipelines/BodyMetrics/</code></summary>

Calcula IMC, calorías diarias, hidratación y ritmo cardíaco máximo.

```bash
python Calculadora-V2.py
```

**Métricas:** IMC · Harris-Benedict · Agua recomendada · Ritmo cardíaco

> 🎀 *GIF próximamente...*
</details>

<details>
<summary><b>🐧 Palmer Penguins</b> — <code>Modulo-3/Proyecto/</code></summary>

Clasificación de especies de pingüinos usando múltiples modelos ML.
</details>

<details>
<summary><b>🔌 TinyML + Arduino</b> — <code>Modulo-5/Pipelines/</code></summary>

Exporta modelos entrenados a C++ para microcontroladores Arduino.  
Incluye clase `ArduinoSimulado` para practicar sin hardware físico.
</details>

<br>

## 📚 Librería Compartida

El archivo [`Machote/machote_ML.py`](Machote/machote_ML.py) contiene funciones reutilizables
que evitan copiar código entre notebooks:

```python
import sys, os
RUTA = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'Machote'))
if RUTA not in sys.path:
    sys.path.append(RUTA)
from machote_ML import *
```

**10-step ML pipeline incluido:**
```
Cargar → EDA → Features → Dividir → Escalar → Modelo → Entrenar → Predecir → Evaluar → Comparar
```

Cada módulo además tiene su propia `libreria_moduloN.py` con funciones específicas
de ese módulo, todas con el mismo patrón de import:

```python
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from libreria_moduloN import *
```

<br>

## 🎀 Tips para empezar

🕸️ **Si vienes de cero:** empieza por `Modulo-2/Estudio/01_Fundamentos_Python/` y avanza en orden numérico.

🕸️ **Si ya sabes Python:** salta directo a `Modulo-3/machote_modulo3.ipynb` o al módulo que te interese.

🕸️ **Cada módulo es autocontenido:** solo necesitas el entorno conda y la librería del módulo.

🕸️ **Notebooks `_blank`:** son plantillas de ejercicios sin resolver — perfectas para practicar.

🕸️ **Notebooks `_V4`, `mejorado`:** versiones corregidas y comentadas.

🕸️ **Google Colab vs Local:** todas las librerías detectan automáticamente la plataforma.

<details open>
<summary><b>🔗 Recursos útiles</b></summary>

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Scikit-learn Docs](https://scikit-learn.org/stable/documentation.html)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Keras/TensorFlow Docs](https://keras.io/)
- [Generador ASCII art](https://patorjk.com/software/taag/)
- [ScreenToGif — grabar GIFs de proyectos](https://www.screentogif.com/)

</details>

<br>

---

<div align="center">

<br>

**Diplomado Superior en Redes Neuronales Artificiales y Deep Learning**  
UAEM — Facultad de Ingeniería

<sub>🕷️ *Hecho con dedicación, debugging y demasiado café* 🎀</sub>

</div>
