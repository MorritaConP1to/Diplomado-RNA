# ═══════════════════════════════════════════════════════════════════════════════
# 📦 libreria_modulo1.py — Introducción a la Inteligencia Artificial
# Diplomado Superior en Redes Neuronales Artificiales y Deep Learning
# Módulo 1 | Diana Blanco
#
# ¿Qué encontrarás aquí?
#   · tipos_aprendizaje()   → supervisado, no supervisado, refuerzo
#   · ramas_ia()            → mapa visual de la IA
#   · concepto_clave()      → definiciones de 9 términos fundamentales
#   · hitos_rna()           → línea del tiempo 1943-2024
#   · ruta_diplomado()      → resumen de los 5 módulos
#   · vectores_basicos()    → repaso matemático (vectores)
#   · matrices_basicas()    → repaso matemático (matrices)
#   · derivada_simbolica()  → reglas de derivación
#   · graficar_funciones()  → visualización de funciones
#
# Importa con:  from libreria_modulo1 import *
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys
import numpy as np
import matplotlib.pyplot as plt

# SECCION 1 — DETECCION DE PLATAFORMA

def detectar_plataforma():
    en_colab = 'google.colab' in sys.modules
    print(f"Plataforma: {'Google Colab' if en_colab else 'Local'}")
    return {'en_colab': en_colab, 'plataforma': 'colab' if en_colab else 'local'}

def configurar_reproducibilidad(semilla=42):
    np.random.seed(semilla)
    print(f"Reproducibilidad configurada (semilla={semilla})")

# SECCION 2 — CONCEPTOS FUNDAMENTALES

def tipos_aprendizaje():
    print("=== TIPOS DE APRENDIZAJE ===")
    print()
    print("1. APRENDIZAJE SUPERVISADO")
    print("   Datos: X (features) + y (etiquetas)")
    print("   Objetivo: Predecir y dado X")
    print("   Modelos: Regresión Lineal, Árboles, SVM, Redes Neuronales")
    print("   Ejemplo: Clasificar correos como spam/no-spam")
    print()
    print("2. APRENDIZAJE NO SUPERVISADO")
    print("   Datos: X (features, sin etiquetas)")
    print("   Objetivo: Encontrar patrones ocultos")
    print("   Modelos: K-Means, DBSCAN, PCA")
    print("   Ejemplo: Segmentar clientes por comportamiento")
    print()
    print("3. APRENDIZAJE POR REFUERZO")
    print("   Datos: Estado + Acción + Recompensa")
    print("   Objetivo: Maximizar recompensa acumulada")
    print("   Modelos: Q-Learning, Deep Q-Networks")
    print("   Ejemplo: Agente que aprende a jugar ajedrez")
    print()
    print("4. APRENDIZAJE SEMISUPERVISADO")
    print("   Datos: Pocos etiquetados + muchos sin etiquetar")
    print("   Objetivo: Aprovechar datos no etiquetados")
    print("   Ejemplo: Clasificación con pocas etiquetas")

def ramas_ia():
    print("=== RAMAS DE LA IA ===")
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│                   INTELIGENCIA ARTIFICIAL          │")
    print("│  ├── Machine Learning (ML)                         │")
    print("│  │    ├── Supervisado                              │")
    print("│  │    ├── No Supervisado                           │")
    print("│  │    └── Por Refuerzo                             │")
    print("│  ├── Deep Learning (DL) — subconjunto del ML       │")
    print("│  │    ├── CNN (Visión por Computadora)             │")
    print("│  │    ├── RNN / LSTM (Procesamiento de Texto)      │")
    print("│  │    └── Transformers (LLMs)                      │")
    print("│  ├── Sistemas Expertos                             │")
    print("│  ├── Robótica                                      │")
    print("│  ├── Procesamiento de Lenguaje Natural (NLP)       │")
    print("│  └── Visión por Computadora                        │")
    print("└─────────────────────────────────────────────────────┘")

def concepto_clave(nombre):
    conceptos = {
        'ia': 'IA: Campo que busca crear sistemas capaces de realizar tareas que requieren inteligencia humana.',
        'ml': 'ML: Subcampo de IA donde los modelos aprenden patrones a partir de datos sin ser programados explícitamente.',
        'dl': 'DL: Subcampo de ML que usa redes neuronales con múltiples capas para aprender representaciones jerárquicas.',
        'red_neuronal': 'Red Neuronal: Modelo inspirado en el cerebro, compuesto de neuronas artificiales organizadas en capas.',
        'perceptron': 'Perceptrón: Unidad básica de una red neuronal. Toma entradas, las pondera, suma y aplica activación.',
        'backpropagation': 'Backpropagation: Algoritmo que ajusta pesos propagando el error hacia atrás por la red.',
        'gradiente': 'Gradiente Descendente: Algoritmo de optimización que minimiza el error ajustando pesos en dirección opuesta al gradiente.',
        'overfitting': 'Overfitting: El modelo aprende demasiado bien los datos de entrenamiento pero falla en datos nuevos.',
        'underfitting': 'Underfitting: El modelo no aprende lo suficiente ni siquiera de los datos de entrenamiento.'
    }
    if nombre in conceptos:
        print(f"{nombre.upper()}: {conceptos[nombre]}")
    else:
        print(f"Concepto '{nombre}' no encontrado. Disponibles: {list(conceptos.keys())}")

# SECCION 3 — REPASO MATEMATICO

def vectores_basicos():
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print(f"v1 = {v1}, v2 = {v2}")
    print(f"Suma:      {v1 + v2}")
    print(f"Resta:     {v1 - v2}")
    print(f"Producto punto: {np.dot(v1, v2)}")
    print(f"Magnitud v1: {np.linalg.norm(v1):.2f}")
    print(f"Ángulo (cos): {np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)):.4f}")

def matrices_basicas():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"A + B =\n{A + B}")
    print(f"A @ B (producto matricial) =\n{A @ B}")
    print(f"Transpuesta de A:\n{A.T}")
    print(f"Determinante de A: {np.linalg.det(A):.2f}")
    if np.linalg.det(A) != 0:
        print(f"Inversa de A:\n{np.linalg.inv(A)}")

def derivada_simbolica():
    print("Reglas básicas de derivación:")
    print("  f(x) = x^n    → f'(x) = n*x^(n-1)")
    print("  f(x) = e^x    → f'(x) = e^x")
    print("  f(x) = ln(x)  → f'(x) = 1/x")
    print("  f(x) = σ(x)   → f'(x) = σ(x)*(1-σ(x))  [sigmoide]")
    print("  f(x) = tanh(x) → f'(x) = 1 - tanh(x)^2")
    print("  Regla de la cadena: (f∘g)' = f'(g(x)) * g'(x)")

def graficar_funciones():
    x = np.linspace(-5, 5, 100)
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.plot(x, x**2); plt.title('f(x)=x²')
    plt.grid(alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.plot(x, np.exp(x)); plt.title('f(x)=e^x')
    plt.grid(alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.plot(x, np.log(np.abs(x) + 0.01)); plt.title('f(x)=ln(x)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# SECCION 4 — HISTORIA DE LAS RNA

def hitos_rna():
    hitos = [
        (1943, "McCulloch & Pitts — Primer modelo matemático de neurona"),
        (1958, "Rosenblatt — Perceptrón, primera red neuronal con aprendizaje"),
        (1969, "Minsky & Papert — Demuestran límites del Perceptrón (XOR) → IA Invierno"),
        (1986, "Rumelhart — Backpropagation + Sigmoide → Renacimiento de RNA"),
        (1998, "LeCun — LeNet-5 (CNN para reconocimiento de dígitos)"),
        (2006, "Hinton — Deep Belief Networks, inicio del Deep Learning moderno"),
        (2012, "Krizhevsky — AlexNet gana ImageNet con CNN (GPU)"),
        (2014, "Goodfellow — GANs (Redes Generativas Antagónicas)"),
        (2017, "Vaswani — Transformer (Attention is All You Need)"),
        (2020, "Brown — GPT-3, modelos de lenguaje a gran escala"),
        (2022, "Stable Diffusion — Modelos de difusión para generación de imágenes"),
        (2024, "Avances en IA multimodal y modelos de frontera")
    ]
    print("=== HITOS EN LA HISTORIA DE LAS RNA ===")
    print()
    for año, hito in hitos:
        print(f"  {año} │ {hito}")
    print()
    print("Periodos clave:")
    print("  1943-1968  │ Nacimiento y primer auge")
    print("  1969-1985  │ Primer invierno de la IA")
    print("  1986-2005  │ Segundo auge (backpropagation)")
    print("  2006-hoy   │ Deep Learning y explosión actual")

# SECCION 5 — RUTA DEL DIPLOMADO

def ruta_diplomado():
    print("=" * 55)
    print("RUTA DEL DIPLOMADO SUPERIOR EN RNA Y DEEP LEARNING")
    print("=" * 55)
    print()
    print("Módulo 1 — Introducción a la IA y al Diplomado")
    print("  Temas: Historia, tipos de aprendizaje, matemáticas básicas")
    print()
    print("Módulo 2 — Python para Inteligencia Artificial")
    print("  Temas: Python, NumPy, Pandas, Matplotlib, MP Neuron")
    print()
    print("Módulo 3 — Machine Learning con scikit-learn")
    print("  Temas: Perceptrón, RL, RLG, Árboles, SVM, K-Means")
    print()
    print("Módulo 4 — Deep Learning: CNNs, RNNs y Aplicaciones")
    print("  Temas: MLP, CNN, RNN/LSTM, Keras, TensorFlow, Despliegue")
    print()
    print("Módulo 5 — Embedded Systems y TinyML")
    print("  Temas: Arduino, TinyML, compuertas, despliegue en hardware")
    print()
    print("Habilidades por módulo:")
    print("  M1: Contexto y fundamentos teóricos")
    print("  M2: Programación en Python para ciencia de datos")
    print("  M3: Machine Learning clásico con scikit-learn")
    print("  M4: Deep Learning con TensorFlow/Keras")
    print("  M5: IA embebida en microcontroladores")
    print()
    print("=" * 55)

# SECCION 6 — RESUMEN

def resumen_modulo():
    print("=" * 50)
    print("Módulo 1 — Introducción a la Inteligencia Artificial")
    print("=" * 50)
    print("Funciones disponibles:")
    print("  tipos_aprendizaje()       — Clasificación de métodos de ML")
    print("  ramas_ia()                — Mapa de áreas de la IA")
    print("  concepto_clave(nombre)    — Definiciones rápidas")
    print("  hitos_rna()               — Línea del tiempo de las RNA")
    print("  ruta_diplomado()          — Resumen de los 5 módulos")
    print("  vectores_basicos()        — Repaso de vectores")
    print("  matrices_basicas()        — Repaso de matrices")
    print("  derivada_simbolica()      — Reglas de derivación")
    print("  graficar_funciones()      — Visualización de funciones")
