# ═══════════════════════════════════════════════════════════════════════════════
# 📦 libreria_modulo2.py — Python para IA Utility Library
# Diplomado Superior en Redes Neuronales Artificiales y Deep Learning
# Módulo 2 | Diana Blanco
#
# Importa con:  from libreria_modulo2 import *
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, math, random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style='whitegrid', palette='muted')

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — DETECCIÓN DE PLATAFORMA
# ═══════════════════════════════════════════════════════════════════════════════

def detectar_plataforma():
    """Detecta si estamos en Google Colab o en local."""
    en_colab = 'google.colab' in sys.modules
    print(f"🌐 Plataforma: {'Google Colab' if en_colab else 'Local'}")
    return {'en_colab': en_colab}


def configurar_reproducibilidad(semilla=42):
    """Fija semilla aleatoria para reproducibilidad."""
    np.random.seed(semilla)
    random.seed(semilla)
    print(f"✅ Reproducibilidad (semilla={semilla})")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def cargar_desde_sklearn(nombre_dataset, return_X_y=True):
    """Carga un dataset de scikit-learn.

    Args:
        nombre_dataset (str): 'breast_cancer', 'iris', 'wine', 'diabetes'
        return_X_y (bool): True → devuelve (X, y), False → objeto Bunch

    Returns:
        tuple: (X, y) o Bunch object

    Ejemplo:
        >>> X, y = cargar_desde_sklearn('iris')
        >>> print(X.shape, y.shape)
    """
    from sklearn import datasets
    dataset_funcs = {
        'breast_cancer': datasets.load_breast_cancer,
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'diabetes': datasets.load_diabetes,
        'digits': datasets.load_digits
    }

    if nombre_dataset not in dataset_funcs:
        disponibles = list(dataset_funcs.keys())
        raise ValueError(f"Dataset no disponible. Opciones: {disponibles}")

    data = dataset_funcs[nombre_dataset]()
    print(f"📦 Dataset: {nombre_dataset}")
    print(f"   Features: {data.data.shape[1]}, Muestras: {data.data.shape[0]}")
    print(f"   Clases: {len(set(data.target)) if hasattr(data.target, '__iter__') else 'Regresión'}")

    if return_X_y:
        return data.data, data.target
    return data


def cargar_desde_csv(ruta, sep=',', encoding='utf-8'):
    """Carga un archivo CSV y muestra resumen.

    Args:
        ruta (str): Ruta al archivo CSV
        sep (str): Separador
        encoding (str): Codificación

    Returns:
        DataFrame

    Ejemplo:
        >>> df = cargar_desde_csv('datos.csv')
    """
    if not os.path.exists(ruta):
        print(f"❌ Archivo no encontrado: {ruta}")
        return None

    df = pd.read_csv(ruta, sep=sep, encoding=encoding)
    print(f"📂 CSV cargado: {ruta}")
    print(f"   Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — EXPLORACIÓN DE DATOS (EDA)
# ═══════════════════════════════════════════════════════════════════════════════

def resumen_rapido(df, target=None):
    """Resumen exploratorio rápido de un DataFrame.

    Args:
        df (DataFrame): Datos
        target (str, opcional): Columna objetivo para ver distribución

    Ejemplo:
        >>> resumen_rapido(df, 'species')
    """
    print("=" * 50)
    print("📊 RESUMEN RÁPIDO DE DATOS")
    print("=" * 50)
    print(f"\n🧱 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"\n📋 Columnas y tipos:")
    for col in df.columns:
        print(f"   {col:25s} → {df[col].dtype}")

    print(f"\n🔢 Estadísticas descriptivas:")
    print(df.describe().round(2))

    print(f"\n🟡 Valores nulos (NaN):")
    nulos = df.isnull().sum()
    if nulos.sum() == 0:
        print("   ✅ Sin valores nulos")
    else:
        print(nulos[nulos > 0])

    if target and target in df.columns:
        print(f"\n🎯 Distribución de '{target}':")
        print(df[target].value_counts())


def ver_distribucion(df, columnas=None, bins=30):
    """Histogramas de distribución para columnas numéricas.

    Args:
        df (DataFrame): Datos
        columnas (list, opcional): Columnas a graficar
        bins (int): Número de bins

    Ejemplo:
        >>> ver_distribucion(df, ['age', 'bmi'])
    """
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns[:6]

    n = len(columnas)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4)) if n > 1 else plt.subplots(1, 1, figsize=(6, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columnas):
        ax.hist(df[col].dropna(), bins=bins, color='steelblue', edgecolor='white', alpha=0.7)
        ax.set_title(f'Distribución: {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def ver_correlacion_con_y(df, target):
    """Muestra correlación de cada feature numérica con la variable objetivo.

    Args:
        df (DataFrame): Datos
        target (str): Columna objetivo

    Ejemplo:
        >>> ver_correlacion_con_y(df, 'precio')
    """
    if target not in df.columns:
        print(f"❌ Columna '{target}' no encontrada")
        return

    numericas = df.select_dtypes(include=[np.number]).columns
    if target not in numericas:
        print("⚠️  La variable target no es numérica. Codifícala primero.")
        return

    correlaciones = df[numericas].corr()[target].drop(target).sort_values(ascending=False)

    print(f"\n📈 Correlación con '{target}':")
    for col, corr in correlaciones.items():
        color = '🟢' if abs(corr) > 0.5 else ('🟡' if abs(corr) > 0.3 else '⚪')
        print(f"   {color} {col:25s} → {corr:+.4f}")

    plt.figure(figsize=(10, 5))
    colores = ['green' if v > 0 else 'red' for v in correlaciones.values]
    plt.barh(correlaciones.index, correlaciones.values, color=colores, alpha=0.7)
    plt.title(f'Correlación con {target}')
    plt.xlabel('Correlación')
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def ver_mapa_calor(df, figsize=(10, 8)):
    """Mapa de calor de correlaciones entre todas las columnas numéricas.

    Args:
        df (DataFrame): Datos
        figsize (tuple): Tamaño de la figura

    Ejemplo:
        >>> ver_mapa_calor(df)
    """
    numericas = df.select_dtypes(include=[np.number])
    if numericas.shape[1] < 2:
        print("⚠️  Se necesitan al menos 2 columnas numéricas")
        return

    plt.figure(figsize=figsize)
    sns.heatmap(numericas.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Mapa de Calor — Correlaciones')
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def seleccionar_features(df, columnas):
    """Selecciona un subconjunto de columnas del DataFrame.

    Args:
        df (DataFrame): Datos originales
        columnas (list): Columnas a conservar

    Returns:
        DataFrame

    Ejemplo:
        >>> X = seleccionar_features(df, ['age', 'bmi', 'glucose'])
    """
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        print(f"⚠️  Columnas no encontradas: {faltantes}")
    seleccionadas = [c for c in columnas if c in df.columns]
    print(f"✅ Features seleccionadas: {len(seleccionadas)}")
    return df[seleccionadas]


def dividir_datos(X, y, test_size=0.2, semilla=42, stratify=False):
    """Divide en entrenamiento y prueba.

    Args:
        X (array): Features
        y (array): Target
        test_size (float): Proporción para test
        semilla (int): Semilla
        stratify (bool): True para mantener proporción de clases

    Returns:
        dict: X_train, X_test, y_train, y_test

    Ejemplo:
        >>> data = dividir_datos(X, y)
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=semilla, stratify=strat
    )
    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def escalar_datos(X_train, X_test, metodo='standard'):
    """Escala datos numéricos con StandardScaler o MinMaxScaler.

    Args:
        X_train (array): Train
        X_test (array): Test
        metodo (str): 'standard' o 'minmax'

    Returns:
        dict: X_train_esc, X_test_esc, scaler

    Ejemplo:
        >>> res = escalar_datos(X_train, X_test, 'minmax')
        >>> X_train_esc = res['X_train']
    """
    if metodo == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train_esc = scaler.fit_transform(X_train)
    X_test_esc = scaler.transform(X_test)

    print(f"✅ Datos escalados con {metodo.upper()}")
    return {'X_train': X_train_esc, 'X_test': X_test_esc, 'scaler': scaler}


def binarizar_datos(X, threshold=0.0):
    """Convierte datos numéricos a binarios (0/1) según un umbral.
    Útil para MPNeuron que necesita entradas binarias.

    Args:
        X (array): Datos
        threshold (float): Umbral (por defecto 0 = mediana)

    Returns:
        array: Datos binarizados

    Ejemplo:
        >>> X_bin = binarizar_datos(X)
    """
    if threshold == 0.0:
        threshold = np.median(X, axis=0)
    X_bin = (X > threshold).astype(int)
    print(f"✅ Datos binarizados (threshold={threshold})")
    return X_bin


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def evaluar_clasificacion(y_real, y_pred, nombres_clases=None):
    """Reporte completo de clasificación.

    Args:
        y_real (array): Etiquetas reales
        y_pred (array): Etiquetas predichas
        nombres_clases (list, opcional): Nombres de las clases

    Ejemplo:
        >>> evaluar_clasificacion(y_test, y_pred, ['benigno','maligno'])
    """
    acc = accuracy_score(y_real, y_pred)
    print(f"\n{'='*50}")
    print(f"🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")

    print(f"\n📋 Reporte de clasificación:")
    print(classification_report(y_real, y_pred, target_names=nombres_clases))

    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.show()

    return acc


def evaluar_regresion(y_real, y_pred):
    """Métricas de regresión: MSE, RMSE, MAE, R².

    Args:
        y_real (array): Valores reales
        y_pred (array): Predicciones

    Returns:
        dict: MSE, RMSE, MAE, R²

    Ejemplo:
        >>> evaluar_regresion(y_test, y_pred)
    """
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)

    print(f"\n{'='*50}")
    print("📊 Métricas de Regresión")
    print(f"{'='*50}")
    print(f"   MSE : {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²  : {r2:.4f}")
    print(f"{'='*50}")

    return {'mse': mse, 'rmse': rmse, 'r2': r2}


def comparar_modelos(resultados):
    """Compara accuracy/métrica de varios modelos.

    Args:
        resultados (dict): {nombre_modelo: accuracy_score}

    Ejemplo:
        >>> comparar_modelos({'Perceptrón': 0.95, 'MPNeuron': 0.90})
    """
    modelos = sorted(resultados.items(), key=lambda x: x[1], reverse=True)

    print(f"\n📊 COMPARACIÓN DE MODELOS")
    print(f"{'='*50}")
    for i, (nombre, metrica) in enumerate(modelos, 1):
        barra = '█' * int(metrica * 50)
        print(f"  {i}. {nombre:15s} │ {metrica:.4f}  {barra}")

    plt.figure(figsize=(10, 5))
    nombres = [m[0] for m in modelos]
    valores = [m[1] for m in modelos]
    colores = plt.cm.Blues(np.linspace(0.4, 0.9, len(modelos)))
    plt.barh(nombres[::-1], valores[::-1], color=colores[::-1])
    plt.xlabel('Accuracy')
    plt.title('Comparación de Modelos')
    plt.xlim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — MPNEURON (McCulloch-Pitts)
# ═══════════════════════════════════════════════════════════════════════════════

class MPNeuron:
    """McCulloch-Pitts Neuron — clasificador binario basado en umbral.

    Ejemplo:
        >>> mp = MPNeuron()
        >>> mp.entrenar(X_bin, y)
        >>> preds = mp.predict(X_bin)
    """

    def __init__(self):
        self.umbral = 0

    def modelo(self, x):
        """Suma las entradas y compara con el umbral."""
        return 1 if sum(x) >= self.umbral else 0

    def predecir(self, X):
        """Predice para múltiples muestras."""
        return np.array([self.modelo(x) for x in X])

    def entrenar(self, X, y):
        """Encuentra el mejor umbral probando valores de 0 a N."""
        mejor_acc = 0
        mejor_umbral = 0
        for u in range(X.shape[1] + 1):
            self.umbral = u
            preds = self.predecir(X)
            acc = accuracy_score(y, preds)
            if acc > mejor_acc:
                mejor_acc = acc
                mejor_umbral = u
        self.umbral = mejor_umbral
        print(f"🧠 MPNeuron entrenada: umbral={mejor_umbral}, accuracy={mejor_acc:.4f}")
        return mejor_acc


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def configurar_path(ruta_base=None):
    """Configura rutas para datos y outputs.

    Args:
        ruta_base (str, opcional): Ruta base (default: directorio actual)

    Returns:
        dict: rutas de datos, modelos, outputs
    """
    if ruta_base is None:
        ruta_base = os.getcwd()

    rutas = {
        'base': ruta_base,
        'datos': os.path.join(ruta_base, 'datos'),
        'outputs': os.path.join(ruta_base, 'outputs')
    }

    for ruta in rutas.values():
        if ruta != rutas['base']:
            os.makedirs(ruta, exist_ok=True)

    print(f"📁 Rutas configuradas: {rutas['datos']}, {rutas['outputs']}")
    return rutas


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — MANEJO DE STRINGS (Tema 1)
# ═══════════════════════════════════════════════════════════════════════════════

def invertir_string(s):
    """Invierte un string.

    Args:
        s (str): Texto a invertir

    Returns:
        str: Texto invertido

    Ejemplo:
        >>> invertir_string('hola')
        'aloh'
    """
    return s[::-1]


def contar_vocales(s):
    """Cuenta vocales (a, e, i, o, u) en un string.

    Args:
        s (str): Texto

    Returns:
        int: Número de vocales

    Ejemplo:
        >>> contar_vocales('hola mundo')
        4
    """
    vocales = 'aeiouáéíóúAEIOUÁÉÍÓÚ'
    return sum(1 for c in s if c in vocales)


def es_palindromo(s):
    """Verifica si un string es palíndromo.

    Args:
        s (str): Texto

    Returns:
        bool

    Ejemplo:
        >>> es_palindromo('anita lava la tina')
        True
    """
    limpio = ''.join(c.lower() for c in s if c.isalnum())
    return limpio == limpio[::-1]


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — FUNCIONES MATEMÁTICAS (Tema 1-2)
# ═══════════════════════════════════════════════════════════════════════════════

def es_par(n):
    """Verifica si un número es par."""
    return n % 2 == 0


def es_primo(n):
    """Verifica si un número es primo."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def factorial(n):
    """Calcula factorial de n de forma iterativa.

    Args:
        n (int): Número entero >= 0

    Returns:
        int: n!

    Ejemplo:
        >>> factorial(5)
        120
    """
    if n < 0:
        raise ValueError("Factorial no definido para negativos")
    resultado = 1
    for i in range(2, n + 1):
        resultado *= i
    return resultado


def fibonacci(n):
    """Genera los primeros n números de Fibonacci.

    Args:
        n (int): Cantidad de términos

    Returns:
        list: [0, 1, 1, 2, 3, 5, ...]

    Ejemplo:
        >>> fibonacci(7)
        [0, 1, 1, 2, 3, 5, 8]
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    secuencia = [0, 1]
    for _ in range(2, n):
        secuencia.append(secuencia[-1] + secuencia[-2])
    return secuencia


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 — RESÚMEN
# ═══════════════════════════════════════════════════════════════════════════════

def resumen_ambiente(info=None):
    """Imprime resumen del entorno."""
    print("=" * 55)
    print("  📋 RESUMEN — Módulo 2: Python para IA")
    print("=" * 55)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  NumPy   : {np.__version__}")
    print(f"  Pandas  : {pd.__version__}")
    print(f"  sklearn : N/A")
    print("=" * 55)
    print("  ✅ Todo listo 🐍")
    print("=" * 55)
