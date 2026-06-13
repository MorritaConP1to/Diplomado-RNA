# ═══════════════════════════════════════════════════════════════════════════════
# 📦 libreria_modulo3.py — Machine Learning con scikit-learn
# Diplomado Superior en Redes Neuronales Artificiales y Deep Learning
# Módulo 3 | Diana Blanco
#
# ¿Qué encontrarás aquí?
#   · Modelos de clasificación   → Perceptron, LogisticRegression, DecisionTree, SVC
#   · Modelos de regresión       → LinearRegression, DecisionTreeRegressor, SVR
#   · Clustering                 → KMeans, encontrar_k_optimo (método del codo)
#   · Preprocesamiento           → dividir_datos, escalar_datos
#   · Evaluación                 → matriz de confusión, métricas de regresión
#   · Visualización              → frontera de decisión, árboles, activaciones
#   · PerceptronManual           → implementación desde cero
#   · Compuertas lógicas         → AND, OR, XOR con neuronas
#   · 13 secciones en total
#
# Importa con:  from libreria_modulo3 import *
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)
from sklearn.linear_model import Perceptron, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn import datasets

# SECCION 1 — DETECCION DE PLATAFORMA

def detectar_plataforma():
    en_colab = 'google.colab' in sys.modules
    print(f"Plataforma: {'Google Colab' if en_colab else 'Local'}")
    return {'en_colab': en_colab, 'plataforma': 'colab' if en_colab else 'local'}

def configurar_reproducibilidad(semilla=42):
    np.random.seed(semilla)
    print(f"Reproducibilidad configurada (semilla={semilla})")

# SECCION 2 — CARGA DE DATOS

def cargar_dataset_sklearn(nombre='iris'):
    nombres = {
        'iris': datasets.load_iris,
        'digits': datasets.load_digits,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
        'diabetes': datasets.load_diabetes
    }
    if nombre not in nombres:
        print(f"Dataset '{nombre}' no disponible. Opciones: {list(nombres.keys())}")
        return None, None
    data = nombres[nombre]()
    X, y = data.data, data.target
    print(f"Cargado: {nombre} | Muestras: {X.shape[0]}, Features: {X.shape[1]}, Clases: {len(np.unique(y))}")
    return X, y

def cargar_desde_csv(ruta):
    df = pd.read_csv(ruta)
    print(f"Cargado: {ruta} | Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    return df

# SECCION 3 — EXPLORACION DE DATOS (EDA)

def resumen_rapido(df):
    print("=== HEAD ==="); display(df.head())
    print("=== INFO ==="); df.info()
    print("=== DESCRIBE ==="); display(df.describe())
    print("=== NULOS ==="); print(df.isnull().sum())

def ver_distribucion(df, columna):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    df[columna].hist(bins=30)
    plt.title(f'Histograma: {columna}')
    plt.subplot(1, 2, 2)
    df[columna].plot(kind='box')
    plt.title(f'Boxplot: {columna}')
    plt.tight_layout()
    plt.show()

def ver_mapa_calor(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de Calor — Correlaciones')
    plt.show()

def ver_correlacion_con_y(df, columna_y):
    numericas = df.select_dtypes(include=[np.number])
    if columna_y not in numericas.columns:
        print(f"Columna '{columna_y}' no es numérica")
        return
    corr = numericas.corr()[columna_y].sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    corr.drop(columna_y).plot(kind='bar')
    plt.title(f'Correlación con {columna_y}')
    plt.axhline(0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()

# SECCION 4 — PREPROCESAMIENTO

def dividir_datos(X, y, test_size=0.2, val_size=0.0, semilla=42):
    if val_size > 0:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, random_state=semilla)
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size+val_size), random_state=semilla)
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=semilla)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def escalar_datos(X_train, X_test=None, metodo='standard'):
    if metodo == 'standard':
        scaler = StandardScaler()
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
    else:
        print(f"Método '{metodo}' no reconocido. Usando StandardScaler.")
        scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_s = scaler.transform(X_test)
        return X_train_s, X_test_s, scaler
    return X_train_s, scaler

# SECCION 5 — MODELOS DE CLASIFICACION

def crear_perceptron(max_iter=1000, tol=1e-3, random_state=42):
    modelo = Perceptron(max_iter=max_iter, tol=tol, random_state=random_state)
    return modelo

def crear_regresion_logistica(C=1.0, max_iter=1000, random_state=42):
    modelo = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    return modelo

def crear_arbol_decision(max_depth=None, min_samples_split=2, random_state=42):
    modelo = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
    return modelo

def crear_svm(kernel='rbf', C=1.0, gamma='scale', random_state=42):
    modelo = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state, probability=True)
    return modelo

# SECCION 6 — MODELOS DE REGRESION

def crear_regresion_lineal():
    modelo = LinearRegression()
    return modelo

def crear_arbol_regresion(max_depth=None, random_state=42):
    modelo = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    return modelo

def crear_svr(kernel='rbf', C=1.0, epsilon=0.1):
    modelo = SVR(kernel=kernel, C=C, epsilon=epsilon)
    return modelo

# SECCION 7 — CLUSTERING

def crear_kmeans(n_clusters=3, random_state=42):
    modelo = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return modelo

def encontrar_k_optimo(X, k_max=10):
    inercias = []
    for k in range(1, k_max+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, k_max+1), inercias, 'bo-')
    plt.xlabel('k (clusters)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.grid(alpha=0.3)
    plt.show()
    return inercias

# SECCION 8 — EVALUACION

def evaluar_clasificacion(y_true, y_pred, clases=None):
    print("=== REPORTE DE CLASIFICACION ===")
    print(classification_report(y_true, y_pred, target_names=clases))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.show()

def evaluar_regresion(y_true, y_pred):
    print("=== METRICAS DE REGRESION ===")
    print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R2:   {r2_score(y_true, y_pred):.4f}")

def graficar_predicciones(y_true, y_pred, titulo='Predicciones vs Reales'):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(titulo)
    plt.grid(alpha=0.3)
    plt.show()

# SECCION 9 — VISUALIZACION DE MODELOS

def visualizar_arbol(modelo, feature_names=None, class_names=None):
    plt.figure(figsize=(15, 10))
    plot_tree(modelo, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

def visualizar_frontera(modelo, X, y, titulo='Frontera de Decisión'):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title(titulo)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    plt.show()

# SECCION 10 — FUNCIONES DE ACTIVACION

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def graficar_activaciones():
    x = np.linspace(-5, 5, 100)
    plt.figure(figsize=(12, 3))
    funciones = [
        ('Sigmoide', sigmoide(x)),
        ('Tanh', tanh(x)),
        ('ReLU', relu(x)),
        ('Escalón', (x >= 0).astype(int))
    ]
    for i, (nombre, y) in enumerate(funciones, 1):
        plt.subplot(1, 4, i)
        plt.plot(x, y)
        plt.title(nombre)
        plt.grid(alpha=0.3)
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)
    plt.tight_layout()
    plt.show()

# SECCION 11 — COMPUERTAS LOGICAS

def compuerta_and(x1, x2, w1=0.5, w2=0.5, umbral=1.0):
    return int(x1 * w1 + x2 * w2 >= umbral)

def compuerta_or(x1, x2, w1=0.5, w2=0.5, umbral=0.5):
    return int(x1 * w1 + x2 * w2 >= umbral)

def compuerta_xor(x1, x2):
    return compuerta_and(compuerta_or(x1, x2), compuerta_or(not x1, not x2))

def mostrar_tabla_compuertas():
    print("AND | OR  | XOR")
    print("-" * 20)
    for a in [0, 1]:
        for b in [0, 1]:
            print(f"{a} & {b} = {compuerta_and(a,b)} | {a} | {b} = {compuerta_or(a,b)} | {a} ^ {b} = {compuerta_xor(a,b)}")

# SECCION 12 — PERCEPTRON MANUAL

class PerceptronManual:
    def __init__(self, tasa_aprendizaje=0.01, epochs=100):
        self.tasa = tasa_aprendizaje
        self.epochs = epochs
        self.pesos = None
        self.sesgo = None

    def entrenar(self, X, y):
        n_muestras, n_features = X.shape
        self.pesos = np.zeros(n_features)
        self.sesgo = 0
        for _ in range(self.epochs):
            for idx in range(n_muestras):
                lineal = np.dot(X[idx], self.pesos) + self.sesgo
                y_pred = 1 if lineal >= 0 else 0
                error = y[idx] - y_pred
                self.pesos += self.tasa * error * X[idx]
                self.sesgo += self.tasa * error
        print(f"Perceptron entrenado | Pesos: {self.pesos}, Sesgo: {self.sesgo}")
        return self

    def predecir(self, X):
        lineales = np.dot(X, self.pesos) + self.sesgo
        return np.where(lineales >= 0, 1, 0)

# SECCION 13 — RESUMEN

def resumen_modulo():
    print("=" * 50)
    print("Módulo 3 — Machine Learning con scikit-learn")
    print("=" * 50)
    print("Modelos disponibles:")
    print("  Clasificación: Perceptron, LogisticRegression, DecisionTree, SVC")
    print("  Regresión:     LinearRegression, DecisionTreeRegressor, SVR")
    print("  Clustering:    KMeans")
    print("  Manual:        PerceptronManual")
    print()
    print("Funciones de preprocesamiento:")
    print("  dividir_datos, escalar_datos")
    print()
    print("Evaluación:")
    print("  evaluar_clasificacion, evaluar_regresion")
    print()
    print("Funciones de activación:")
    print("  sigmoide, tanh, relu, graficar_activaciones")
    print()
    print("Compuertas lógicas:")
    print("  and, or, xor, mostrar_tabla_compuertas")
