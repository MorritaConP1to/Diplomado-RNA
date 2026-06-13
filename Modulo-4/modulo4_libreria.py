# ═══════════════════════════════════════════════════════════════════════════════
# 📦 modulo4_libreria.py — Deep Learning Utility Library
# Diplomado Superior en Redes Neuronales Artificiales y Deep Learning
# Módulo 4 | Diana Blanco
#
# Importa con:  from modulo4_libreria import *
# Funciona en:  Google Colab  ✅  |  VSCode/Jupyter Local  ✅
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, json, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ── TensorFlow / Keras ─────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, applications

# ── Scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — DETECCIÓN DE PLATAFORMA Y CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def detectar_plataforma():
    """Detecta automáticamente el entorno y configura rutas.
    
    Returns:
        dict: Información del entorno con claves:
            - 'en_colab' (bool): True si estamos en Google Colab
            - 'plataforma' (str): 'google_colab' o 'local'
            - 'rutas' (dict): rutas base/datos/modelos/outputs
            - 'gpu' (list): dispositivos GPU detectados
            - 'semilla' (int): semilla de reproducibilidad
    """
    EN_COLAB = 'google.colab' in sys.modules
    
    info = {
        'en_colab': EN_COLAB,
        'plataforma': 'google_colab' if EN_COLAB else 'local',
        'semilla': 42,
        'gpu': [],
        'rutas': {}
    }
    
    if EN_COLAB:
        print("🌐 Plataforma: Google Colab")
        from google.colab import drive
        drive.mount('/content/drive')
        
        RUTA_BASE    = '/content/drive/MyDrive/Diplomado/Modulo4'
        RUTA_DATOS   = os.path.join(RUTA_BASE, 'datos')
        RUTA_MODELOS = os.path.join(RUTA_BASE, 'modelos')
        RUTA_OUTPUTS = os.path.join(RUTA_BASE, 'outputs')
    else:
        print("🖥️  Plataforma: Local (VSCode / Jupyter)")
        RUTA_BASE    = os.getcwd()
        RUTA_DATOS   = os.path.join(RUTA_BASE, 'Material')
        RUTA_MODELOS = os.path.join(RUTA_BASE, 'modelos')
        RUTA_OUTPUTS = os.path.join(RUTA_BASE, 'outputs')
    
    for ruta in [RUTA_DATOS, RUTA_MODELOS, RUTA_OUTPUTS]:
        os.makedirs(ruta, exist_ok=True)
    
    info['rutas'] = {
        'base': RUTA_BASE,
        'datos': RUTA_DATOS,
        'modelos': RUTA_MODELOS,
        'outputs': RUTA_OUTPUTS
    }
    
    return info


def configurar_reproducibilidad(semilla=42):
    """Fija semillas aleatorias para resultados reproducibles.
    
    Args:
        semilla (int): Semilla para numpy, TF y Python random
    
    Ejemplo:
        >>> configurar_reproducibilidad(42)
    """
    np.random.seed(semilla)
    tf.random.set_seed(semilla)
    random.seed(semilla)
    print(f"✅ Reproducibilidad configurada (semilla={semilla})")


def verificar_gpu(info=None):
    """Verifica disponibilidad de GPU y configura memoria dinámica.
    
    Args:
        info (dict, opcional): Output de detectar_plataforma()
    
    Returns:
        list: Lista de GPUs detectadas
    
    Ejemplo:
        >>> info = detectar_plataforma()
        >>> gpus = verificar_gpu(info)
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"🎮 GPU(s) detectada(s): {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            tf.config.experimental.set_memory_growth(gpu, True)
        
        try:
            import subprocess as _sp
            _result = _sp.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if _result.returncode == 0:
                print(f"   Modelo: {_result.stdout.strip()}")
        except:
            pass
    else:
        print("⚠️  No se detectó GPU — se usará CPU")
        if info and info['en_colab']:
            print("   → En Colab: ve a Entorno → Cambiar tipo → T4 GPU")
    
    print(f"🔧 Dispositivo activo: {tf.test.gpu_device_name() or 'CPU'}")
    return gpus


def setup_completo():
    """Ejecuta la configuración completa (plataforma + reproducibilidad + GPU).
    
    Returns:
        dict: Información del entorno con rutas y GPU
    
    Ejemplo:
        >>> INFO = setup_completo()
        >>> print(INFO['rutas']['datos'])
    """
    print("=" * 55)
    print("  🔧 CONFIGURACIÓN AUTOMÁTICA — Módulo 4 Deep Learning")
    print("=" * 55)
    
    info = detectar_plataforma()
    configurar_reproducibilidad(info['semilla'])
    info['gpu'] = verificar_gpu(info)
    
    print(f"\n📁 Rutas:")
    for k, v in info['rutas'].items():
        print(f"   {k}: {v}")
    
    print("=" * 55)
    print("  ✅ Todo listo para trabajar")
    print("=" * 55)
    
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CARGA Y PREPROCESAMIENTO DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def cargar_csv(ruta, target=None, test_size=0.2, val_size=0.1, semilla=42, 
               normalize=False, categorias=None):
    """Carga un CSV, separa features/target y divide en train/val/test.
    
    Args:
        ruta (str): Ruta al archivo CSV
        target (str): Nombre de la columna objetivo
        test_size (float): Proporción para test (0.0 a 1.0)
        val_size (float): Proporción de train para validación
        semilla (int): Semilla para reproducibilidad
        normalize (bool): Si escalar features con StandardScaler
        categorias (list): Columnas categóricas a codificar
    
    Returns:
        dict: Con claves X_train, X_val, X_test, y_train, y_val, y_test,
              scaler (si normalize), encoder (si categorias)
    
    Ejemplo:
        >>> data = cargar_csv('datos.csv', target='precio', normalize=True)
        >>> X_train, y_train = data['X_train'], data['y_train']
    """
    df = pd.read_csv(ruta)
    print(f"📂 CSV cargado: {ruta} — {df.shape[0]} filas × {df.shape[1]} columnas")
    
    X = df.drop(columns=[target]) if target else df
    y = df[target] if target else None
    
    if categorias:
        X = pd.get_dummies(X, columns=categorias)
        print(f"   Categorías codificadas: {categorias}")
    
    scaler = None
    if normalize:
        scaler = StandardScaler()
        cols_num = X.select_dtypes(include=[np.number]).columns
        X[cols_num] = scaler.fit_transform(X[cols_num])
        print("   Features normalizadas con StandardScaler ✓")
    
    if y is not None:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=semilla
        )
        val_frac = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac, random_state=semilla
        )
        print(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        
        return {
            'X_train': X_train.values if hasattr(X_train, 'values') else X_train,
            'X_val': X_val.values if hasattr(X_val, 'values') else X_val,
            'X_test': X_test.values if hasattr(X_test, 'values') else X_test,
            'y_train': y_train.values if hasattr(y_train, 'values') else y_train,
            'y_val': y_val.values if hasattr(y_val, 'values') else y_val,
            'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
            'scaler': scaler,
            'feature_names': list(X.columns)
        }
    
    return {'X': X.values if hasattr(X, 'values') else X, 'scaler': scaler}


def cargar_imagenes_desde_carpeta(ruta_carpeta, target_size=(224, 224), 
                                   batch_size=32, validation_split=0.2, 
                                   seed=42, class_mode='categorical'):
    """Carga imágenes organizadas en subcarpetas con ImageDataGenerator.
    
    Estructura esperada:
        carpeta/
        ├── clase_1/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── clase_2/
        │   └── ...
    
    Args:
        ruta_carpeta (str): Ruta a la carpeta con subcarpetas por clase
        target_size (tuple): (alto, ancho) para redimensionar
        batch_size (int): Tamaño de lote
        validation_split (float): Proporción para validación
        seed (int): Semilla
        class_mode (str): 'categorical', 'binary', 'sparse'
    
    Returns:
        tuple: (train_generator, val_generator)
    
    Ejemplo:
        >>> train_gen, val_gen = cargar_imagenes_desde_carpeta('datos/flores/')
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    train_gen = datagen.flow_from_directory(
        ruta_carpeta,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='training',
        seed=seed
    )
    
    val_gen = datagen.flow_from_directory(
        ruta_carpeta,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='validation',
        seed=seed
    )
    
    print(f"   Clases encontradas: {list(train_gen.class_indices.keys())}")
    return train_gen, val_gen


def preprocesar_texto(textos_train, textos_test=None, vocab_size=10000, max_len=200):
    """Tokeniza y paddes secuencias de texto.
    
    Args:
        textos_train (list): Lista de strings para entrenar el tokenizador
        textos_test (list, opcional): Lista de strings a transformar
        vocab_size (int): Tamaño máximo del vocabulario
        max_len (int): Longitud máxima de secuencia (padding/truncado)
    
    Returns:
        dict: Con X_train_pad, X_test_pad, tokenizador
    
    Ejemplo:
        >>> res = preprocesar_texto(['hola mundo', 'adiós'], ['mundo'])
        >>> X_train = res['X_train_pad']
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    tokenizador = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizador.fit_on_texts(textos_train)
    
    seq_train = tokenizador.texts_to_sequences(textos_train)
    X_train_pad = pad_sequences(seq_train, maxlen=max_len, padding='pre', truncating='pre')
    
    X_test_pad = None
    if textos_test is not None:
        seq_test = tokenizador.texts_to_sequences(textos_test)
        X_test_pad = pad_sequences(seq_test, maxlen=max_len, padding='pre', truncating='pre')
    
    print(f"📝 Vocabulario: {min(len(tokenizador.word_index), vocab_size)} palabras")
    print(f"   Train: {X_train_pad.shape} | Test: {X_test_pad.shape if X_test_pad is not None else 'N/A'}")
    
    return {
        'X_train_pad': X_train_pad,
        'X_test_pad': X_test_pad,
        'tokenizador': tokenizador
    }


def crear_ventanas_tiempo(serie, n_pasos):
    """Convierte serie 1D en ventanas para LSTM.
    
    Args:
        serie (array): Serie temporal
        n_pasos (int): Pasos de tiempo por ventana
    
    Returns:
        tuple: (X, y) para entrenar LSTM
    
    Ejemplo:
        >>> X, y = crear_ventanas_tiempo([1,2,3,4,5,6], 3)
        >>> # X = [[1,2,3],[2,3,4],[3,4,5]], y = [4,5,6]
    """
    X, y = [], []
    for i in range(len(serie) - n_pasos):
        X.append(serie[i:i+n_pasos])
        y.append(serie[i+n_pasos])
    
    X = np.array(X).reshape(-1, n_pasos, 1)
    y = np.array(y)
    print(f"📊 Ventanas: {X.shape[0]} muestras, {n_pasos} pasos")
    return X, y


def train_val_test_split(X, y, test_size=0.2, val_size=0.1, semilla=42):
    """Divide datos en train/validation/test de forma consistente.
    
    Returns:
        dict: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=semilla
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=semilla
    )
    print(f"📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — CONSTRUCCIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

def crear_mlp(input_shape, num_clases, capas_ocultas=[128, 64], dropout=0.3):
    """Perceptrón Multicapa para datos tabulares.
    
    Args:
        input_shape (tuple): Forma de entrada, ej. (20,) para 20 features
        num_clases (int): 2 para binaria, >2 para multiclase
        capas_ocultas (list): Neuronas por capa oculta
        dropout (float): Tasa de dropout (0.0 a 1.0)
    
    Returns:
        keras.Model: Modelo compilable
    
    Ejemplo:
        >>> modelo = crear_mlp((20,), 3, capas_ocultas=[64, 32])
        >>> modelo.summary()
    """
    modelo = keras.Sequential(name="MLP_Basico")
    modelo.add(layers.Input(shape=input_shape))
    
    for neuronas in capas_ocultas:
        modelo.add(layers.Dense(neuronas, activation='relu'))
        modelo.add(layers.BatchNormalization())
        modelo.add(layers.Dropout(dropout))
    
    if num_clases == 2:
        modelo.add(layers.Dense(1, activation='sigmoid'))
    else:
        modelo.add(layers.Dense(num_clases, activation='softmax'))
    
    return modelo


def crear_cnn(input_shape, num_clases, filtros=[32, 64, 128], kernel_size=3):
    """CNN para clasificación de imágenes.
    
    Args:
        input_shape (tuple): (alto, ancho, canales), ej. (28,28,1)
        num_clases (int): Número de categorías
        filtros (list): Filtros por capa convolucional
        kernel_size (int): Tamaño del kernel
    
    Returns:
        keras.Model
    
    Ejemplo:
        >>> cnn = crear_cnn((28,28,1), 10)
    """
    modelo = keras.Sequential(name="CNN_Clasificacion")
    
    for i, f in enumerate(filtros):
        if i == 0:
            modelo.add(layers.Conv2D(f, kernel_size, activation='relu', 
                                     padding='same', input_shape=input_shape))
        else:
            modelo.add(layers.Conv2D(f, kernel_size, activation='relu', padding='same'))
        modelo.add(layers.BatchNormalization())
        modelo.add(layers.MaxPooling2D((2,2)))
    
    modelo.add(layers.GlobalAveragePooling2D())
    modelo.add(layers.Dense(128, activation='relu'))
    modelo.add(layers.Dropout(0.5))
    
    if num_clases == 2:
        modelo.add(layers.Dense(1, activation='sigmoid'))
    else:
        modelo.add(layers.Dense(num_clases, activation='softmax'))
    
    return modelo


def crear_lstm_texto(vocab_size, max_len, embedding_dim=64, num_clases=2):
    """LSTM bidireccional para clasificación de texto.
    
    Args:
        vocab_size (int): Tamaño del vocabulario
        max_len (int): Longitud máxima de secuencia
        embedding_dim (int): Dimensiones del embedding
        num_clases (int): 2 para binaria, >2 para multiclase
    
    Returns:
        keras.Model
    
    Ejemplo:
        >>> lstm = crear_lstm_texto(10000, 200, num_clases=2)
    """
    modelo = keras.Sequential(name="LSTM_Texto")
    modelo.add(layers.Embedding(vocab_size, embedding_dim, mask_zero=True))
    modelo.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    modelo.add(layers.Bidirectional(layers.LSTM(32)))
    modelo.add(layers.Dense(32, activation='relu'))
    modelo.add(layers.Dropout(0.3))
    
    if num_clases == 2:
        modelo.add(layers.Dense(1, activation='sigmoid'))
    else:
        modelo.add(layers.Dense(num_clases, activation='softmax'))
    
    return modelo


def crear_lstm_series(n_pasos, n_features=1, tipo='regresion'):
    """LSTM para predicción de series de tiempo.
    
    Args:
        n_pasos (int): Pasos de tiempo de entrada
        n_features (int): Variables por paso
        tipo (str): 'regresion' o 'clasificacion'
    
    Returns:
        keras.Model
    """
    modelo = keras.Sequential(name="LSTM_Series")
    modelo.add(layers.LSTM(64, return_sequences=True, input_shape=(n_pasos, n_features)))
    modelo.add(layers.Dropout(0.2))
    modelo.add(layers.LSTM(32))
    modelo.add(layers.Dropout(0.2))
    modelo.add(layers.Dense(16, activation='relu'))
    
    if tipo == 'regresion':
        modelo.add(layers.Dense(1, activation='linear'))
    else:
        modelo.add(layers.Dense(2, activation='softmax'))
    
    return modelo


def crear_transfer_learning(num_clases, input_shape=(224, 224, 3), 
                             base='MobileNetV2', trainable_base=False):
    """Modelo con transfer learning usando base preentrenada de ImageNet.
    
    Args:
        num_clases (int): Número de categorías
        input_shape (tuple): (224,224,3) típico para ImageNet
        base (str): 'MobileNetV2', 'VGG16', 'ResNet50', 'EfficientNetB0'
        trainable_base (bool): Si descongelar la base (fine-tuning)
    
    Returns:
        tuple: (modelo_completo, base_model)
    
    Ejemplo:
        >>> modelo, base = crear_transfer_learning(5, base='MobileNetV2')
        >>> modelo.summary()
    """
    bases = {
        'MobileNetV2': applications.MobileNetV2,
        'VGG16': applications.VGG16,
        'ResNet50': applications.ResNet50,
        'EfficientNetB0': applications.EfficientNetB0
    }
    
    if base not in bases:
        raise ValueError(f"Base no soportada. Opciones: {list(bases.keys())}")
    
    base_model = bases[base](
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base_model.trainable = trainable_base
    
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    if num_clases == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_clases, activation='softmax')(x)
    
    modelo = keras.Model(inputs, outputs, name=f"TransferLearning_{base}")
    return modelo, base_model


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def compilar_y_entrenar(modelo, X_train, y_train, X_val, y_val,
                         num_clases=2, lr=0.001, epochs=50, batch_size=32,
                         verbose=1, early_stop_paciencia=7, reducir_lr=True):
    """Compila y entrena un modelo Keras con callbacks automáticos.
    
    Args:
        modelo (keras.Model): Modelo a entrenar
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        num_clases (int): 2 para binaria, >2 para multiclase
        lr (float): Learning rate para Adam
        epochs (int): Épocas máximas
        batch_size (int): Tamaño de lote
        verbose (int): 0=silencioso, 1=barra, 2=por época
        early_stop_paciencia (int): Paciencia para early stopping
        reducir_lr (bool): Si usar ReduceLROnPlateau
    
    Returns:
        History: Objeto history del entrenamiento
    
    Ejemplo:
        >>> hist = compilar_y_entrenar(modelo, X_train, y_train, X_val, y_val)
    """
    if num_clases == 2:
        loss = 'binary_crossentropy'
        metric = 'accuracy'
    else:
        loss = 'sparse_categorical_crossentropy'
        metric = 'accuracy'
    
    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[metric]
    )
    
    cbs = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=early_stop_paciencia,
            restore_best_weights=True, verbose=1
        )
    ]
    
    if reducir_lr:
        cbs.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, verbose=1
            )
        )
    
    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=verbose
    )
    
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — EVALUACIÓN Y VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def evaluar_clasificacion(modelo, X_test, y_test, nombres_clases=None):
    """Evalúa un clasificador: accuracy, reporte, matriz de confusión.
    
    Args:
        modelo (keras.Model): Modelo entrenado
        X_test, y_test: Datos de prueba
        nombres_clases (list, opcional): Nombres de las clases
    
    Returns:
        ndarray: Predicciones
    
    Ejemplo:
        >>> preds = evaluar_clasificacion(cnn, X_test, y_test)
    """
    y_pred_prob = modelo.predict(X_test, verbose=0)
    
    if y_pred_prob.shape[-1] == 1:
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = np.mean(y_pred == y_test)
    print(f"\n{'='*50}")
    print(f"🎯 Accuracy en test: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")
    
    print(f"\n📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=nombres_clases))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.show()
    
    return y_pred


def evaluar_regresion(modelo, X_test, y_test):
    """Evalúa un modelo de regresión: MSE, RMSE, MAE, R².
    
    Args:
        modelo (keras.Model): Modelo entrenado
        X_test, y_test: Datos de prueba
    
    Returns:
        dict: Métricas de regresión
    """
    y_pred = modelo.predict(X_test, verbose=0).flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print("📊 Métricas de Regresión")
    print(f"{'='*50}")
    print(f"   MSE  : {mse:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print(f"   R²   : {r2:.4f}")
    print(f"{'='*50}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def graficar_historia(history, titulo="Entrenamiento"):
    """Grafica curvas de pérdida y accuracy del entrenamiento.
    
    Args:
        history: Objeto History de keras.fit()
        titulo (str): Título del gráfico
    
    Ejemplo:
        >>> history = modelo.fit(X_train, y_train, ...)
        >>> graficar_historia(history, "Mi Modelo")
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"📈 {titulo}", fontsize=14, fontweight='bold')
    
    axes[0].plot(history.history['loss'], label='Train Loss', color='steelblue')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='tomato', linestyle='--')
    axes[0].set_title('Función de Pérdida')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Train Acc', color='steelblue')
        axes[1].plot(history.history['val_accuracy'], label='Val Acc', color='tomato', linestyle='--')
        axes[1].set_title('Exactitud (Accuracy)')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def mostrar_lote_imagenes(images, labels, clases=None, n=10, cmap='gray'):
    """Muestra un lote de imágenes con sus etiquetas.
    
    Args:
        images (array): Imágenes (N, alto, ancho, canales)
        labels (array): Etiquetas
        clases (list): Nombres de clases
        n (int): Número de imágenes a mostrar
        cmap (str): Mapa de color ('gray', 'viridis', etc.)
    
    Ejemplo:
        >>> mostrar_lote_imagenes(X_train, y_train, clases=['gato','perro'])
    """
    n = min(n, len(images))
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for i in range(n):
        img = images[i].squeeze()
        axes[i].imshow(img, cmap=cmap)
        label = labels[i]
        if clases:
            label = clases[int(label)] if label < len(clases) else label
        axes[i].set_title(f"Clase: {label}")
        axes[i].axis('off')
    
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — GUARDADO Y CARGA DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

def guardar_modelo(modelo, nombre, ruta_modelos='modelos'):
    """Guarda modelo en formato .keras.
    
    Args:
        modelo (keras.Model): Modelo a guardar
        nombre (str): Nombre del archivo (sin extensión)
        ruta_modelos (str): Carpeta de destino
    
    Returns:
        str: Ruta completa del archivo guardado
    
    Ejemplo:
        >>> ruta = guardar_modelo(modelo, 'mi_cnn')
    """
    os.makedirs(ruta_modelos, exist_ok=True)
    ruta = os.path.join(ruta_modelos, f'{nombre}.keras')
    modelo.save(ruta)
    print(f"✅ Modelo guardado: {ruta}")
    return ruta


def cargar_modelo(nombre, ruta_modelos='modelos'):
    """Carga un modelo guardado previamente.
    
    Args:
        nombre (str): Nombre del archivo (sin extensión)
        ruta_modelos (str): Carpeta donde está el modelo
    
    Returns:
        keras.Model o None: Modelo cargado
    
    Ejemplo:
        >>> modelo = cargar_modelo('mi_cnn')
    """
    ruta = os.path.join(ruta_modelos, f'{nombre}.keras')
    if os.path.exists(ruta):
        modelo = keras.models.load_model(ruta)
        print(f"✅ Modelo cargado: {ruta}")
        return modelo
    print(f"❌ No se encontró: {ruta}")
    return None


def guardar_figura(nombre, ruta_outputs='outputs', dpi=150):
    """Guarda la figura actual de matplotlib.
    
    Args:
        nombre (str): Nombre del archivo (sin extensión)
        ruta_outputs (str): Carpeta de destino
        dpi (int): Resolución en DPI
    
    Returns:
        str: Ruta completa
    """
    os.makedirs(ruta_outputs, exist_ok=True)
    ruta = os.path.join(ruta_outputs, f'{nombre}.png')
    plt.savefig(ruta, dpi=dpi, bbox_inches='tight')
    print(f"✅ Figura guardada: {ruta}")
    return ruta


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def crear_data_augmentation():
    """Crea pipeline de aumento de datos para imágenes.
    
    Returns:
        keras.Sequential: Capa de data augmentation
    
    Ejemplo:
        >>> aug = crear_data_augmentation()
        >>> # Usar dentro de un modelo funcional:
        >>> # x = aug(inputs)
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — SERIES DE TIEMPO (helpers compatibles con LSTM)
# ═══════════════════════════════════════════════════════════════════════════════

def preparar_serie_lstm(df, columna_target, n_pasos=10, test_size=0.2):
    """Prepara DataFrame completo para LSTM de series de tiempo.
    
    Args:
        df (DataFrame): Datos con columna temporal
        columna_target (str): Columna a predecir
        n_pasos (int): Ventana de contexto
        test_size (float): Proporción para test
    
    Returns:
        dict: X_train, X_test, y_train, y_test
    
    Ejemplo:
        >>> data = preparar_serie_lstm(df_ventas, 'ventas', n_pasos=30)
    """
    valores = df[columna_target].values
    X, y = crear_ventanas_tiempo(valores, n_pasos)
    
    split = int(len(X) * (1 - test_size))
    return {
        'X_train': X[:split], 'X_test': X[split:],
        'y_train': y[:split], 'y_test': y[split:]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — RESUMEN DEL AMBIENTE
# ═══════════════════════════════════════════════════════════════════════════════

def resumen_ambiente(info=None):
    """Imprime resumen del entorno de trabajo.
    
    Args:
        info (dict, opcional): Output de detectar_plataforma()
    """
    gpu_name = tf.test.gpu_device_name() or 'No detectada (CPU)'
    
    print("=" * 55)
    print("  📋 RESUMEN DE AMBIENTE — Módulo 4 Deep Learning")
    print("=" * 55)
    print(f"  Plataforma   : {info['plataforma'] if info else '—'}")
    print(f"  TensorFlow   : {tf.__version__}")
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  GPU          : {gpu_name}")
    print(f"  NumPy        : {np.__version__}")
    print(f"  Pandas       : {pd.__version__}")
    print(f"  Matplotlib   : {plt.matplotlib.__version__}")
    if info:
        print(f"  Ruta base    : {info['rutas']['base']}")
        print(f"  Semilla      : {info['semilla']}")
    print("=" * 55)
    print("  ✅ Todo listo — feliz entrenamiento 🧠")
    print("=" * 55)
