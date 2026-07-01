#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Genera CNN_Sanrio.ipynb y Transfer_Learning_Sanrio.ipynb con estilo biclase."""
import json, os

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

BASE = os.path.dirname(os.path.abspath(__file__))

VERSION = 'v2'  # Incrementa al regenerar (v1, v2, ...)
SUFIJO = f'_{VERSION}' if VERSION else ''

# ============================================================
# CELDAS COMPARTIDAS (Setup, Datos, Visualizacion, Eval, Footer)
# ============================================================

shared_md_contexto = md_cell("""---
# Clasificador Multiclase Sanrio

**Diplomado Superior RNA y Deep Learning — UAEM**
**Modulo 4: Deep Learning | Sanrio_Multiclase**

### Que estamos construyendo?

Un clasificador que mira una foto de un personaje de
Sanrio y dice:
"?Esto es Hello Kitty? ?O My Melody? ?O Pompompurin?"

No es solo un juguete. La misma tecnica se usa para:
- **Diagnostico medico**: ?que tipo de cancer? (multiclase)
- **Robotica**: ?que objeto tengo enfrente?
- **Moderacion**: ?que tipo de contenido?

### Diferencias con clasificacion binaria

| Aspecto | Binario (2 clases) | Multiclase (N clases) |
|---------|-------------------|----------------------|
| Salida | 1 neurona (0/1) | N neuronas (una por clase) |
| Loss | BCELoss o CrossEntropy | CrossEntropy (N clases) |
| Metrica | Accuracy, Precision, Recall | Accuracy + accuracy por clase |
| Dificultad | Separar 2 grupos | Separar N grupos simultaneamente |

Este notebook es uno de los **2 entregables** de la tarea.
""")

shared_md_datos = md_cell("""---
## Data Augmentation: estirando el dataset

### Por que?

Con ~1000 imagenes por clase y N clases, la aumentacion es clave
para que el modelo generalice y no memorice fondos o angulos.

### Analogia

Es como estudiar para un examen donde el profesor puede cambiar
el orden de las preguntas, el color del papel y el tipo de letra.
Si practicas siempre con el mismo formato, el dia del examen
cualquier cambio te desconcierta.

### Transformaciones

| Transformacion | Simula |
|---------------|--------|
| RandomResizedCrop(224) | La figura no siempre esta centrada |
| RandomHorizontalFlip | Foto desde el otro lado |
| RandomRotation(15deg) | Camara ligeramente inclinada |
| ColorJitter | Distinta iluminacion |
""")

shared_code_setup = code_cell("""# ============================================
# Setup: imports, plataforma, hiperparametros
#
# ?Que hace: Importa herramientas, detecta
#   Colab vs local, define CONFIG.
#
# ?Variables:
#   - EN_COLAB (bool): True si estamos en Google Colab
#   - RUTA_BASE (str): raiz del proyecto
#   - device (str): 'cuda' (GPU) o 'cpu'
#   - CONFIG (dict): todas las perillas juntas
#     - BATCH_SIZE: imagenes por lote
#     - EPOCHS: cuantas veces ver el dataset completo
#     - LR: learning rate (que tan rapido aprende)
#     - DROPOUT: % de neuronas apagadas al azar
#     - PATIENCE: epocas sin mejora antes de parar
#
# ?Por que batch_size=32?
#   En binario usamos 16 porque ResNet18 pesa ~2.5GB.
#   Para CNN desde cero, el modelo es mas liviano
#   y podemos usar 32 sin problemas.
#
# ?Por que lr=0.001?
#   Valor por defecto de Adam. Funciona en ~80% de los casos.
#
# ?Para experimentar:
#   - Si el kernel muere (OOM), baja BATCH_SIZE a 16 o 8
#   - Si la loss no baja, prueba lr=0.0005
# ============================================

import os, sys, time, json, warnings, math, gc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

AUTORA = "Diana Blanco - MorritaConP1to"
INICIO = time.time()

EN_COLAB = 'google.colab' in sys.modules

if EN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/Diplomado/Modulo4/Proyectos/Sanrio_Multiclase'
else:
    RUTA_BASE = os.path.dirname(os.path.abspath('__file__')) if '__file__' in dir() else os.getcwd()

RUTA_DATASET = os.path.join(RUTA_BASE, 'dataset')
RUTA_MODELOS = os.path.join(RUTA_BASE, 'modelos')
os.makedirs(RUTA_MODELOS, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', device)
print('Dataset:', RUTA_DATASET)

CONFIG = {
    'BATCH_SIZE': 32,
    'EPOCHS': 80,   # Early stopping detiene antes; 80 es suficiente, evita kernel crash
    'LR': 0.001,
    'DROPOUT': 0.5,
    'PATIENCE': 7,
    'IMG_SIZE': 224,
    'AUTORA': AUTORA,
}
print('CONFIG:', CONFIG)
""")

shared_code_dataloader = code_cell("""# ============================================
# Cargar dataset: augment + ImageFolder
#
# ?Que hace: Crea pipelines de transformacion
#   y carga imagenes con ImageFolder.
#
# ?Variables:
#   - transform_train: con aumentacion (entrenar)
#   - transform_test: sin aumentacion (evaluar)
#   - train_dataset (ImageFolder): asocia cada imagen
#     con su clase segun la carpeta donde esta
#   - clases (list): nombres de las clases detectadas
#   - NUM_CLASES (int): numero de personajes
#   - train_loader (DataLoader): itera en lotes
#
# ?Por que ImageFolder?
#   Toma la estructura de carpetas y asigna
#   automaticamente: carpeta = clase.
#   dataset/train/hello_kitty/ -> clase 0
#   dataset/train/kuromi/ -> clase 1
#   etc. (orden ALFABETICO)
#
# ?Para experimentar:
#   - Agrega o quita carpetas en train/
#     El notebook se adapta solo al NUM_CLASES
# ============================================

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=os.path.join(RUTA_DATASET, 'train'),
                            transform=transform_train)
test_dataset = ImageFolder(root=os.path.join(RUTA_DATASET, 'test'),
                           transform=transform_test)

clases = train_dataset.classes
NUM_CLASES = len(clases)
print('Clases detectadas:', clases)
print('Total clases:', NUM_CLASES)
print('Train:', len(train_dataset), 'imagenes')
print('Test:', len(test_dataset), 'imagenes')
print('Mapa clase->indice:', train_dataset.class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
""")

shared_code_visualizacion = code_cell("""# ============================================
# Visualizar muestras por clase
#
# ?Que hace: Muestra 4 imagenes de cada clase
#   para confirmar que los datos se cargaron bien.
#   Las imagenes de train se ven "diferentes"
#   cada vez por la aumentacion aleatoria.
# ============================================

# Limpiar CUDA residual antes de graficar
plt.close('all')
gc.collect()
torch.cuda.empty_cache()

def mostrar_muestras(dataset, clases, num_por_clase=4):
    num_clases = len(clases)
    fig, axes = plt.subplots(num_clases, num_por_clase,
                             figsize=(num_por_clase*2, num_clases*2.5))
    if num_clases == 1:
        axes = axes.reshape(1, -1)

    for idx_clase, clase in enumerate(clases):
        indices = [i for i, (_, label) in enumerate(dataset) if label == idx_clase]
        np.random.shuffle(indices)
        indices = indices[:num_por_clase]

        for j, idx in enumerate(indices):
            img, _ = dataset[idx]
            img = img.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)

            axes[idx_clase, j].imshow(img)
            axes[idx_clase, j].axis('off')
            if j == 0:
                axes[idx_clase, j].set_ylabel(clase[:12], fontsize=9,
                                              fontweight='bold')

    plt.suptitle('Muestras por clase (train)', fontsize=14)
    plt.tight_layout()
    plt.show()

mostrar_muestras(train_dataset, clases)
""")

shared_code_graficas = code_cell("""# ============================================
# Graficas de evolucion
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(historial['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[0].plot(historial['test_loss'], 'r-', label='Test Loss', linewidth=2)
axes[0].set_title('Perdida (Loss)')
axes[0].set_xlabel('Epoca')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(historial['test_acc'], 'g-', linewidth=2)
axes[1].set_title('Accuracy en Test')
axes[1].set_xlabel('Epoca')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 105)
axes[1].grid(True, alpha=0.3)

ultimo_acc = historial['test_acc'][-1]
axes[2].text(0.5, 0.6, '{:.2f}%'.format(ultimo_acc),
             fontsize=42, ha='center', fontweight='bold', color='green')
axes[2].text(0.5, 0.3, 'Accuracy Final',
             fontsize=16, ha='center', color='gray')
axes[2].axis('off')

plt.tight_layout()
plt.show()
""")

shared_md_metricas = md_cell("""---
## Leyendo las metricas

### Matriz de confusion multiclase

La diagonal principal son los aciertos. Fuera de la diagonal son errores.
Con N clases, la matriz es N x N.

**Ejemplo de lectura:**
| Real -> | Predijo A | Predijo B | Predijo C |
|---------|-----------|-----------|-----------|
| **Era A** | 50 (bien) | 2 | 1 |
| **Era B** | 3 | 45 (bien) | 5 |
| **Era C** | 1 | 2 | 48 (bien) |

### Accuracy por clase

Mide que tan bien clasifica CADA personaje individualmente.
Si una clase tiene accuracy bajo, esa figura necesita mas datos
o fotos mas variadas.

### Que esperar

| Accuracy | Significado |
|----------|-------------|
| ~30-50% | Aleatorio (N=12, azar = 8.3%) |
| 60-70% | CNN desde cero, aprendizaje basico |
| 70-85% | CNN decente |
| 85-95% | CNN buena (con aumentacion) |
| 95%+ | Transfer Learning |
""")

shared_code_evaluacion = code_cell("""# ============================================
# Evaluacion: matriz de confusion + accuracy por clase
#
# ?Que hace: Pasa todas las imagenes de test por
#   el modelo y calcula metricas detalladas.
#
# ?Por que matriz de confusion?
#   El accuracy total puede esconder problemas:
#   - Una clase con muchos errores
#   - El modelo confunde sistematicamente 2 clases
#   - La matriz revela estos patrones
# ============================================

# Limpiar CUDA residual antes de evaluar
plt.close('all')
gc.collect()
torch.cuda.empty_cache()

def evaluar(modelo, test_loader, device, clases):
    modelo.eval()
    todas_reales = []
    todas_predichas = []

    with torch.no_grad():
        for imagenes, etiquetas in test_loader:
            imagenes = imagenes.to(device)
            outputs = modelo(imagenes)
            _, predichas = torch.max(outputs, 1)

            todas_reales.extend(etiquetas.cpu().numpy())
            todas_predichas.extend(predichas.cpu().numpy())

    cm = confusion_matrix(todas_reales, todas_predichas)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(clases)*0.8),
                                          max(6, len(clases)*0.7)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d',
              xticks_rotation=45)
    ax.set_title('Matriz de Confusion Multiclase', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print('\\n=== Accuracy por Clase ===')
    aciertos_total = 0
    for i, clase in enumerate(clases):
        total_clase = cm[i, :].sum()
        correctos = cm[i, i]
        acc = 100.0 * correctos / total_clase if total_clase > 0 else 0
        aciertos_total += correctos
        barra = '?' * int(acc / 5) + '?' * (20 - int(acc / 5))
        print('  {:<20s} {:>3d}/{:>3d} = {:>5.1f}% {}'.format(
            clase, correctos, total_clase, acc, barra))

    accuracy = 100.0 * aciertos_total / cm.sum()
    print('\\n  Accuracy TOTAL: {:.2f}%  ({:d}/{:d} aciertos)'.format(
        accuracy, aciertos_total, cm.sum()))
    return cm

cm = evaluar(model, test_loader, device, clases)
""")

shared_md_errores = md_cell("""---
## Donde falla? Aprendiendo de los errores

Los errores revelan las debilidades del modelo. Esta celda
muestra las imagenes mal clasificadas.

### Que buscar

| Patron de error | Posible causa | Solucion |
|----------------|--------------|----------|
| Confunde siempre 2 clases | Son muy parecidas visualmente | Mas datos de ambas, o aumentacion especifica |
| Errores en una sola clase | Pocas imagenes de esa clase en test | Balancear dataset |
| Errores con fondo similar | El modelo aprendio el fondo, no la figura | Variar fondos en las fotos |
""")

shared_code_errores = code_cell("""# ============================================
# Grilla de errores
# ============================================

plt.close('all')
gc.collect()
torch.cuda.empty_cache()

model.eval()
imagenes_guardadas = []
with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        imagenes = imagenes.to(device)
        outputs = model(imagenes)
        probabilidades = torch.softmax(outputs, dim=1)
        _, predichas = torch.max(outputs, 1)
        for i in range(len(imagenes)):
            if len(imagenes_guardadas) < 200:
                imagenes_guardadas.append((
                    imagenes[i].cpu(), etiquetas[i].item(),
                    predichas[i].item(), probabilidades[i]))

errores = [(img, r, p, probs) for (img, r, p, probs) in imagenes_guardadas if r != p]

if len(errores) == 0:
    print('\\n? No hubo errores en las 200 muestras guardadas!')
else:
    n_mostrar = min(len(errores), 12)
    cols = 4
    rows = math.ceil(n_mostrar / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    axes = axes.flatten() if n_mostrar > 1 else [axes]

    for i in range(n_mostrar):
        img, real, pred, probs = errores[i]
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
              torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        axes[i].imshow(img.permute(1, 2, 0).numpy())
        conf = probs[pred].item() * 100
        axes[i].set_title('Real: {}\\nPred: {} ({:.0f}%)'.format(
            clases[real], clases[pred], conf),
            fontsize=9, color='red')
        axes[i].axis('off')

    for i in range(n_mostrar, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Errores ({}/{} muestras)'.format(
        len(errores), len(imagenes_guardadas)),
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print('\\nTasa de error: {:.1f}%'.format(
        100 * len(errores) / len(imagenes_guardadas)))
""")

shared_md_inferencia = md_cell("""---
## Probando con tus propias fotos

Toma una foto de un personaje Sanrio y el modelo te dira cual es.
Muestra el Top-3 con barras de confianza.
""")

shared_code_inferencia = code_cell("""# ============================================
# Prediccion batch desde carpeta
#
# ?Que hace: Escanea RUTA_PRUEBAS por imagenes,
#   corre prediccion en cada una, y muestra
#   una grilla con el resultado.
#
# Cambia RUTA_PRUEBAS si tus imagenes estan en
# otra carpeta.
# ============================================

RUTA_PRUEBAS = 'modelos'  # Cambia esta ruta si gustas

EXTENSIONES = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

def predecir_imagen(ruta_imagen, modelo, device, clases):
    modelo.eval()
    imagen = Image.open(ruta_imagen).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(imagen).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = modelo(tensor)
        probabilidades = torch.softmax(outputs, dim=1)[0]

    prob_np = probabilidades.cpu().numpy()
    idx_pred = int(torch.argmax(probabilidades))
    conf = prob_np[idx_pred] * 100
    return imagen, clases[idx_pred], conf, prob_np

if EN_COLAB:
    from google.colab import files
    subidos = files.upload()
    archivos_prueba = list(subidos.keys())
    RUTA_PRUEBAS = None  # No usamos carpeta en Colab
else:
    if os.path.isdir(RUTA_PRUEBAS):
        archivos_prueba = [os.path.join(RUTA_PRUEBAS, f)
                           for f in os.listdir(RUTA_PRUEBAS)
                           if f.lower().endswith(EXTENSIONES)]
        archivos_prueba.sort()
    else:
        archivos_prueba = []
        print('? La carpeta "%s" no existe.' % RUTA_PRUEBAS)

if not archivos_prueba:
    print('No se encontraron imagenes en %s.' % RUTA_PRUEBAS)
else:
    n = len(archivos_prueba)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for i, ruta in enumerate(archivos_prueba):
        nombre = os.path.basename(ruta)
        try:
            img, pred, conf, _ = predecir_imagen(ruta, model, device, clases)
            axes[i].imshow(img)
            axes[i].set_title('%s\\n? %s (%.1f%%)' % (nombre, pred, conf),
                              fontsize=10, fontweight='bold', color='green')
        except Exception as e:
            axes[i].set_title('%s\\n? Error' % nombre, fontsize=10, color='red')
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Predicciones del modelo', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
""")

shared_code_guardar = code_cell("""# ============================================
# Guardar modelo
# ============================================

from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

torch.save(model.state_dict(),
           os.path.join(RUTA_MODELOS, NOMBRE_MODELO + '_{}.pth'.format(timestamp)))
torch.save(model.state_dict(),
           os.path.join(RUTA_MODELOS, NOMBRE_MODELO + '_final.pth'))

with open(os.path.join(RUTA_MODELOS, 'clases_sanrio.json'), 'w') as f:
    json.dump({
        'classes': clases,
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': NUM_CLASES,
        'fecha': timestamp,
        'autora': AUTORA,
    }, f, indent=2)

print('Modelo y clases guardados.')
print('Clases:', clases)
""")

shared_code_footer = code_cell("""# ============================================
# Footer: autoria y rendimiento
# ============================================

import psutil, platform

fin = time.time()
segundos = fin - INICIO
mm, ss = divmod(int(segundos), 60)

print('')
print('=' * 55)
print('  Hecho con carino por ' + AUTORA)
print('  Diplomado RNA - Modulo 4: Deep Learning')
print('  UAEM')
print('')
print('  Especificaciones:')
cpu = platform.processor() or 'AMD Ryzen'
print('    CPU: ' + cpu)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
print('    GPU: ' + gpu_name + ' (' + str(gpu_mem) + ' GB)')
ram = round(psutil.virtual_memory().total / 1e9, 1)
print('    RAM: ' + str(ram) + ' GB')
print('    PyTorch ' + torch.__version__)
print('')
print('  Tiempo: {:d} min {:d} seg'.format(mm, ss))
print('  Accuracy: {:.2f}%'.format(historial['test_acc'][-1]))
print('  Clases: {:d}'.format(NUM_CLASES))
print('  Fecha: ' + time.strftime('%Y-%m-%d %H:%M'))
print('=' * 55)
""")

# ============================================================
# CELDAS ESPECIFICAS: CNN desde cero
# ============================================================

cnn_md_arquitectura = md_cell("""---
## Arquitectura CNN desde cero

### Que es una CNN?

Una CNN (Convolutional Neural Network) aprende a reconocer patrones
visuales en niveles de abstraccion creciente:

| Capa | Ve | Analogia |
|------|-----|----------|
| Conv1 | Bordes, esquinas, colores | Trazos basicos de un dibujo |
| Conv2 | Texturas (rayas, puntos) | Partes de un objeto |
| Conv3 | Formas simples (circulos, triangulos) | Piezas de un rompecabezas |
| Conv4 | Partes de objetos (ojos, orejas) | Piezas mas complejas |
| Classifier | Combina todo para decidir la clase | El veredicto final |

### Arquitectura de nuestro modelo

```
Input: 224x224x3 (imagen a color)
    |
Conv1 (3 -> 32)  + BN + ReLU + MaxPool  -> 112x112x32
    |
Conv2 (32 -> 64) + BN + ReLU + MaxPool  -> 56x56x64
    |
Conv3 (64 -> 128) + BN + ReLU + MaxPool -> 28x28x128
    |
Conv4 (128 -> 256) + BN + ReLU + MaxPool -> 14x14x256
    |
Global Average Pooling -> Vector de 256
    |
FC(256 -> 128) -> ReLU -> Dropout -> FC(128 -> N_CLASES)
```

### Por que Batch Normalization (BN)?

Estabiliza el entrenamiento. Como un corrector de ortografia:
cada capa recibe datos con formato consistente,
aunque la entrada original sea ruidosa.

### Por que Global Average Pooling (GAP)?

Reduce cada mapa de caracteristicas (14x14x256) a un solo valor
por canal (1x1x256). Elimina la necesidad de capas FC grandes.
Menos parametros = menos overfitting.
""")

cnn_md_entrenamiento = md_cell("""---
## Entrenamiento

### Que pasa en cada epoca

1. **Modo entrenamiento**: Toma lotes de imagenes, las pasa por el modelo,
   calcula el error (loss), ajusta pesos (backpropagation).

2. **Modo evaluacion**: Pasa todas las imagenes de test SIN ajustar pesos,
   solo mide accuracy.

3. **Scheduler**: Si la loss de test deja de mejorar, divide el
   learning rate a la mitad (ReduceLROnPlateau).

4. **Early Stopping**: Si pasan 7 epocas sin mejora, detiene el
   entrenamiento automaticamente.

### Que esperar

| Epoca | Train Loss | Test Loss | Acc | Que significa |
|-------|-----------|-----------|-----|--------------|
| 1-5 | 2.5 -> 1.5 | 2.5 -> 1.5 | 20-40% | Aprendizaje inicial (mejor que azar) |
| 5-15 | 1.5 -> 0.8 | 1.5 -> 0.9 | 40-60% | Empieza a distinguir formas |
| 15-25 | 0.8 -> 0.3 | 0.9 -> 0.5 | 60-80% | Mejora progresiva |
| 25+ | 0.3 -> 0.1 | 0.5 -> 0.3 | 80-90%+ | Refinamiento (si no overfittea) |

> Con Transfer Learning estos valores son mucho mejores desde la epoca 1.
> Esta CNN desde cero parte de pesos aleatorios, es normal que tarde mas.
""")

cnn_code_modelo = code_cell("""# ============================================
# Definir la CNN desde cero
#
# ?Clase: CNNMulticlase(nn.Module)
#   Hereda de nn.Module (clase base de PyTorch).
#
# ?Metodos:
#   - __init__: define las capas
#   - forward: define el flujo de datos
#
# ?Por que 4 bloques y no 3 o 5?
#   Con 224x224 de entrada:
#   - 3 bloques: muy pocos, no extrae suficientes features
#   - 5 bloques: demasiados para entrenar desde cero
#   - 4 bloques: punto dulce para este tamano de imagen
#
# ?Por que duplicar Conv en cada bloque?
#   Dos convoluciones seguidas permiten aprender
#   combinaciones de patrones mas complejas que una sola.
#   Es el diseno clasico de VGG.
#
# ?Para experimentar:
#   - Agrega un 5to bloque (256 -> 512).
#     ?Mejora el accuracy o overfitea?
#   - Quita BatchNorm. ?El entrenamiento es mas inestable?
# ============================================

class CNNMulticlase(nn.Module):
    def __init__(self, num_clases, dropout=0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(max(0.1, dropout * 0.6)),
            nn.Linear(128, num_clases),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

model = CNNMulticlase(num_clases=NUM_CLASES, dropout=CONFIG['DROPOUT']).to(device)

total_params = sum(p.numel() for p in model.parameters())
print('CNNMulticlase para {:d} clases'.format(NUM_CLASES))
print('Parametros: {:,}'.format(total_params))
""")

cnn_code_entrenar = code_cell("""# ============================================
# Funcion de entrenamiento
#
# ?Parametros:
#   - modelo: la CNN definida arriba
#   - train_loader, test_loader: datos
#   - criterion: CrossEntropyLoss (mide el error)
#   - optimizer: Adam (ajusta los pesos)
#   - scheduler: ReduceLROnPlateau
#   - num_epochs: maximo de epocas
#   - device: 'cuda' o 'cpu'
#
# ?Retorna: historial (dict) con losses y accuracies
#
# ?Por que .item() en losses?
#   loss es un tensor. .item() lo convierte a numero.
#   Si acumularamos tensores directamente, la memoria creceria.
#
# ?Por que torch.no_grad() en test?
#   No necesitamos gradientes para evaluar.
#   Desactivarlos ahorra memoria y acelera.
# ============================================

def entrenar(modelo, train_loader, test_loader, criterion,
             optimizer, scheduler, num_epochs, device):
    historial = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    mejor_loss = float('inf')
    patience = 0

    for epoch in range(num_epochs):
        modelo.train()
        train_loss = 0.0

        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)

            optimizer.zero_grad()
            predicciones = modelo(imagenes)
            loss = criterion(predicciones, etiquetas)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        historial['train_loss'].append(train_loss)

        modelo.eval()
        test_loss = 0.0
        correctos = 0
        total = 0

        with torch.no_grad():
            for imagenes, etiquetas in test_loader:
                imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
                outputs = modelo(imagenes)
                loss = criterion(outputs, etiquetas)
                test_loss += loss.item()

                _, predichos = torch.max(outputs, 1)
                total += etiquetas.size(0)
                correctos += (predichos == etiquetas).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100.0 * correctos / total
        historial['test_loss'].append(test_loss)
        historial['test_acc'].append(accuracy)

        lr_actual = optimizer.param_groups[0]['lr']
        print('Epoch [{:2d}/{}]  Train Loss: {:.4f}  Test Loss: {:.4f}  Acc: {:.2f}%  LR: {:.6f}'.format(
              epoch+1, num_epochs, train_loss, test_loss, accuracy, lr_actual))

        scheduler.step(test_loss)

        if test_loss < mejor_loss:
            mejor_loss = test_loss
            torch.save(modelo.state_dict(),
                       os.path.join(RUTA_MODELOS, 'mejor_cnn.pth'))
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG['PATIENCE']:
                print('Early stopping en epoca {}'.format(epoch+1))
                break

    print('\\nEntrenamiento completado.')
    print('Mejor accuracy: {:.2f}%'.format(max(historial['test_acc'])))
    return historial
""")

cnn_code_ejecutar = code_cell("""# ============================================
# Ejecutar entrenamiento
# ============================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

print('Iniciando entrenamiento CNN desde cero...')
print('Clases: {:d} | Batch: {:d} | Epochs: {:d} | LR: {:.4f} | Dropout: {:.2f}'.format(
    NUM_CLASES, CONFIG['BATCH_SIZE'], CONFIG['EPOCHS'], CONFIG['LR'], CONFIG['DROPOUT']))

historial = entrenar(model, train_loader, test_loader, criterion,
                     optimizer, scheduler, CONFIG['EPOCHS'], device)
""")

# ============================================================
# CELDAS DE EXPERIMENTOS (compartidas por CNN y TL)
# ============================================================

def make_experimento_cell(prefijo):
    """Crea celda de historial de experimentos para un prefijo (cnn/tl)."""
    return code_cell("""# ============================================
# Historial de experimentos anteriores
# ============================================
# Edita EXPERIMENTO_ACTUAL antes de entrenar para
# registrar QUE estas probando y POR QUE.
#
# Variantes sugeridas:
#   1) BASE: config actual
#   2) LR 0.0005  -> si la loss se estanca
#   3) WD 1e-4    -> regularizacion contra overfitting
#   4) Mas filtros (64->128->256->512) -> mas capacidad
#   5) Combinado: LR bajo + WD + mas filtros
# ============================================

EXPERIMENTO_ACTUAL = {{
    'id': 1,
    'desc': 'Base',
    'razon': 'Configuracion inicial',
    'cambios': 'Ninguno',
}}

RUTA_EXP = os.path.join(RUTA_MODELOS, 'experimentos_{}.json'.format('{{}}'))
exp_historial = []""".format(prefijo))

cnn_code_experimento = make_experimento_cell('cnn')
tl_code_experimento = make_experimento_cell('tl')

cnn_code_variantes = code_cell("""# ============================================
# VARIANTES DE HIPERPARAMETROS
# ============================================
# Descomenta UNA variante para probarla.
# Cambia EXPERIMENTO_ACTUAL arriba para
# registrar el cambio en el historial.
#
# --- Variante 2: LR mas bajo ---
# CONFIG['LR'] = 0.0005
# EXPERIMENTO_ACTUAL = {'id': 2, 'desc': 'LR 0.0005',
#     'razon': 'El loss se estanca temprano con LR alto',
#     'cambios': 'LR 0.001 -> 0.0005'}
#
# --- Variante 3: Weight Decay ---
# CONFIG['WD'] = 1e-4
# EXPERIMENTO_ACTUAL = {'id': 3, 'desc': '+ WD 1e-4',
#     'razon': 'Overfitting: train loss baja pero test no',
#     'cambios': '+ weight_decay=1e-4'}
#
# --- Variante 4: Mas filtros ---
# class CNNMulticlase(nn.Module): ...
# No es solo CONFIG, hay que modificar la clase.
# Prueba esta variante DESPUES de la celda de
# definicion del modelo, antes de entrenar.
# Cambia block1: 32->64, block2: 64->128, etc.
# EXPERIMENTO_ACTUAL = {'id': 4, 'desc': 'Filtros 64-128-256-512',
#     'razon': 'Mas capacidad para 10 clases',
#     'cambios': '32->64, 64->128, 128->256, 256->512'}
#
# --- Variante 5: Combinado ---
# CONFIG['LR'] = 0.0005
# CONFIG['WD'] = 1e-4
# EXPERIMENTO_ACTUAL = {'id': 5, 'desc': 'Combinado LR+WD',
#     'razon': 'Mejor combinacion de cambios',
#     'cambios': 'LR 0.0005 + WD 1e-4'}
#
# --- Variante 6: SGD con momentum ---
# Cambiar el optimizador en la celda de entrenamiento
# optimizer = optim.SGD(model.parameters(), lr=CONFIG['LR'], momentum=0.9)
# EXPERIMENTO_ACTUAL = {'id': 6, 'desc': 'SGD momentum 0.9',
#     'razon': 'SGD a veces generaliza mejor que Adam',
#     'cambios': 'Adam -> SGD+momentum'}

print('Variantes disponibles. Descomenta una en esta celda si deseas.')
print('Experimento actual: #{} - {}'.format(
    EXPERIMENTO_ACTUAL['id'], EXPERIMENTO_ACTUAL['desc']))
""")

shared_code_guardar_exp = code_cell("""# ============================================
# Guardar resultado en historial de experimentos
# ============================================

from datetime import datetime

nuevo_exp = {
    'id': EXPERIMENTO_ACTUAL['id'],
    'descripcion': EXPERIMENTO_ACTUAL['desc'],
    'razon': EXPERIMENTO_ACTUAL['razon'],
    'cambios': EXPERIMENTO_ACTUAL['cambios'],
    'accuracy': max(historial['test_acc']),
    'final_acc': historial['test_acc'][-1],
    'epochs': len(historial['test_acc']),
    'config': {k: v for k, v in CONFIG.items()
               if k not in ('AUTORA',)},
    'fecha': datetime.now().strftime('%Y-%m-%d %H:%M'),
}

if not any(e['id'] == nuevo_exp['id'] for e in exp_historial):
    exp_historial.append(nuevo_exp)
    with open(RUTA_EXP, 'w') as f:
        json.dump(exp_historial, f, indent=2)
    print('Experimento #{} guardado: {:.2f}%'.format(
        nuevo_exp['id'], nuevo_exp['accuracy']))
else:
    print('Experimento #{} ya existe, no se sobreescribe'.format(
        nuevo_exp['id']))
""")

shared_code_comparativa = code_cell("""# ============================================
# Tabla comparativa de todos los experimentos
# ============================================

print('=== COMPARATIVA DE EXPERIMENTOS ===')
print('  {:>3s}  {:25s}  {:30s}  {:>6s}  {:>6s}'.format(
    '#', 'Descripcion', 'Cambio', 'Acc max', 'Epocas'))
print('  ' + '-' * 77)

if exp_historial:
    for e in sorted(exp_historial, key=lambda x: x['accuracy'], reverse=True):
        print('  {:3d}  {:25s}  {:30s}  {:5.1f}%  {:4d}'.format(
            e['id'], e['descripcion'],
            e.get('cambios', e['razon'])[:30],
            e['accuracy'], e['epochs']))

    mejor = max(exp_historial, key=lambda x: x['accuracy'])
    print()
    print('  MEJOR EXPERIMENTO: #{} - {} con {:.2f}%'.format(
        mejor['id'], mejor['descripcion'], mejor['accuracy']))
else:
    print('  (No hay datos)')
""")

cnn_code_loop = code_cell("""# ============================================
# BATERIA AUTOMATICA DE EXPERIMENTOS
# ============================================
# Corre 6 variantes de hiperparametros una tras
# otra. Al final deja el MEJOR modelo cargado
# como 'model' y su historial como 'historial'
# para las celdas de evaluacion y graficas.
#
# Los resultados se guardan en:
#   modelos/experimentos_cnn.json
# ============================================

experimentos_plan = [
    {'id': 1, 'desc': 'BASE: lr=0.001, Adam',
     'razon': 'Configuracion inicial (linea base)',
     'lr': 0.001, 'wd': 0, 'optim': 'adam', 'dropout': 0.5, 'sched_patience': 3},
    # ============================================================
    # Para probar otros experimentos:
    # 1. Descomenta SOLO uno a la vez
    # 2. REINICIA el kernel (Kernel > Restart)
    # 3. Ejecuta esta celda + las siguientes
    # ============================================================
    # {'id': 2, 'desc': 'LR 0.0005 + WD 1e-4',
    #  'razon': 'Adam con LR bajo y regularizacion L2',
    #  'lr': 0.0005, 'wd': 1e-4, 'optim': 'adam', 'dropout': 0.5, 'sched_patience': 3},
    # {'id': 3, 'desc': 'SGD momentum + WD 1e-4',
    #  'razon': 'SGD con momentum y regularizacion',
    #  'lr': 0.01, 'wd': 1e-4, 'optim': 'sgd', 'dropout': 0.5, 'sched_patience': 3},
]

print('=' * 60)
print('BATERIA DE EXPERIMENTOS: 1 activo (los otros comentados)')
print('Cada uno usa early stopping (patience=%d) con max %d epocas.' % (
    CONFIG['PATIENCE'], CONFIG['EPOCHS']))
print('Si el kernel muere, reinicia y corre solo 1 experimento a la vez.')
print('=' * 60)

resultados = []
best_model_state = None
best_acc = 0
best_exp_id = 0
best_historial = None

for exp in experimentos_plan:
    print()
    print('=' * 60)
    print('EXPERIMENTO #%d: %s' % (exp['id'], exp['desc']))
    print('  %s' % exp['razon'])
    print('  lr=%.4f | wd=%.6f | optim=%-4s | dropout=%.1f | sched_patience=%d' % (
        exp['lr'], exp['wd'], exp['optim'],
        exp['dropout'], exp['sched_patience']))
    print('=' * 60)

    modelo_exp = CNNMulticlase(num_clases=NUM_CLASES,
                                dropout=exp['dropout']).to(device)

    if exp['optim'] == 'sgd':
        opt = optim.SGD(modelo_exp.parameters(), lr=exp['lr'],
                        momentum=0.9, weight_decay=exp['wd'])
    else:
        opt = optim.Adam(modelo_exp.parameters(), lr=exp['lr'],
                         weight_decay=exp['wd'])

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=exp['sched_patience'])
    criterion = nn.CrossEntropyLoss()

    h = entrenar(modelo_exp, train_loader, test_loader, criterion,
                 opt, sched, CONFIG['EPOCHS'], device)

    acc_max = max(h['test_acc'])
    acc_final = h['test_acc'][-1]
    resultados.append({**exp, 'accuracy': acc_max, 'final_acc': acc_final,
                       'epochs': len(h['test_acc']), 'historial': h})

    if acc_max > best_acc:
        best_acc = acc_max
        best_model_state = {k: v.cpu() for k, v in modelo_exp.state_dict().items()}
        best_historial = h
        best_exp_id = exp['id']

    print('  >> Resultado: %.2f%% (mejor hasta ahora: %.2f%%)' % (
        acc_max, best_acc))

    # Limpiar CUDA entre experimentos para evitar corrupcion
    del modelo_exp, opt, sched
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Cargar el mejor modelo (CON ESTADO EN CPU -> seguro)
print()
print('=' * 60)
print('CARGANDO MEJOR MODELO: Experimento #%d' % best_exp_id)
print('=' * 60)
model = CNNMulticlase(num_clases=NUM_CLASES, dropout=0.5)
model.load_state_dict(best_model_state)
model = model.to(device)
model.load_state_dict(best_model_state)
historial = best_historial

# Guardar resultados en historial JSON
from datetime import datetime

for r in resultados:
    nuevo = {
        'id': r['id'], 'descripcion': r['desc'],
        'razon': r['razon'],
        'cambios': 'lr=%.4f wd=%.6f optim=%s' % (
            r['lr'], r['wd'], r['optim']),
        'accuracy': r['accuracy'], 'final_acc': r['final_acc'],
        'epochs': r['epochs'],
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    if not any(e['id'] == nuevo['id'] for e in exp_historial):
        exp_historial.append(nuevo)

with open(RUTA_EXP, 'w') as f:
    json.dump(exp_historial, f, indent=2)

# Tabla comparativa
print()
print('=' * 80)
print('COMPARATIVA DE EXPERIMENTOS (ordenado por accuracy)')
print('=' * 80)
print('  %-3s  %-30s  %6s  %6s  %4s' % (
    '#', 'Descripcion', 'Acc max', 'Final', 'Eps'))
print('  ' + '-' * 55)
resultados.sort(key=lambda x: x['accuracy'], reverse=True)
for r in resultados:
    marca = ' <<<' if r['id'] == best_exp_id else ''
    print('  %-3d  %-30s  %5.1f%%  %5.1f%%  %4d%s' % (
        r['id'], r['desc'][:30], r['accuracy'], r['final_acc'],
        r['epochs'], marca))
print('-' * 55)
print('  MEJOR: Experimento #%d - %s (%.2f%%)' % (
    best_exp_id, resultados[0]['desc'], best_acc))
print()
print('Las celdas siguientes evaluan y grafican ESTE modelo.')
""")

# ============================================================
# CELDAS ESPECIFICAS: Transfer Learning
# ============================================================

tl_md_transfer = md_cell("""---
## Transfer Learning con ResNet18

### Que es Transfer Learning?

En lugar de entrenar desde cero (pesos aleatorios), tomamos una red
que ya fue entrenada con **1.2 MILLONES de imagenes** (ImageNet)
y la adaptamos a nuestro problema.

### Analogia del chef

> Contratas a un chef que ya sabe cocina francesa.
> Tu quieres que aprenda cocina mexicana.
> No le ensenas desde "pelar una papa" (ya lo sabe).
> Solo le ensenas las recetas nuevas (tus ingredientes).
>
> **Eso es Transfer Learning:**
> - El chef = ResNet18 pre-entrenada en ImageNet
> - Las recetas = nuestras fotos de Sanrio
> - Pelar papas = detectar bordes, colores, texturas

### Por que ResNet18?

| Modelo | Parametros | Con 1000-2000 imagenes |
|--------|-----------|------------------------|
| ResNet18 | 11.7M | Punto dulce: suficiente capacidad, poco overfitting |
| ResNet34 | 21.8M | Demasiada capacidad para ~1000 img/clase |
| ResNet50 | 25.6M | Solo con 5000+ imagenes por clase |

**Arbol de decision:**
```
?Cuantas imagenes por clase?
    |
< 500  ---> ResNet18 (mucho aumento)
500-2000 -> ResNet18 (estas aqui)
2000+  ---> ResNet34 o EfficientNet
```

### Estrategia de 2 fases

```
Fase 1: Solo la cabeza nueva
  +----------+     La cabeza aprende a interpretar
  | Cabeza   |     los features de ImageNet
  | ENTRENA  |     (512 -> 256 -> N_CLASES)
  +----------+
  | layer4   | CONGELADO
  | layer1-3 | CONGELADO (bordes, texturas: universales)

Fase 2: Fine-tuning de layer4
  +----------+
  | Cabeza   | ENTRENA (lr bajo)
  +----------+
  | layer4   | ENTRENA (lr 10x menor)
  +----------+     Ajusta detalles finos de Sanrio
  | layer1-3 | CONGELADO
```
""")

tl_code_modelo = code_cell("""# ============================================
# Crear ResNet18 con Transfer Learning
#
# ?Que hace: Carga ResNet18 pre-entrenada, congela
#   todo, y reemplaza la cabeza por una nueva
#   que clasifica NUM_CLASES en vez de 1000.
#
# ?Variables:
#   - model.fc.in_features = 512 (entrada a la FC)
#   - model.fc: la cabeza clasificadora original
#   - requires_grad = False: congela pesos
#
# ?Por que 512 -> 256 -> N_CLASES?
#   Podriamos ir directo 512 -> N_CLASES, pero la
#   capa intermedia de 256 con ReLU y dropout
#   le da mas capacidad a la cabeza sin ser enorme.
#
# ?Para experimentar:
#   - Cambia a models.resnet34(weights=...)
#     ?Cuantos parametros mas tiene?
#   - Prueba sin la capa intermedia (512 -> N_CLASES)
#     ?Aprende igual o peor?
# ============================================

def crear_modelo(num_clases, congelar=True):
    modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if congelar:
        for param in modelo.parameters():
            param.requires_grad = False

    num_features = modelo.fc.in_features

    modelo.fc = nn.Sequential(
        nn.Dropout(CONFIG['DROPOUT']),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_clases),
    )

    return modelo

model = crear_modelo(num_clases=NUM_CLASES, congelar=True).to(device)

total = sum(p.numel() for p in model.parameters())
entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('ResNet18 para {:d} clases'.format(NUM_CLASES))
print('Parametros: {:,} total | {:,} entrenables ({:.1f}%)'.format(
    total, entrenables, 100*entrenables/total))
""")

tl_md_fases = md_cell("""---
## Dos fases de entrenamiento

### Por que no entrenar todo desde el principio?

| Si entrenas todo desde epoca 1 | Pasa esto |
|------------------------------|-----------|
| La cabeza nueva (pesos aleatorios) | Produce gradientes enormes |
| Esos gradientes "contaminan" layer4 | El modelo olvida ImageNet |
| Resultado | Overfitting inmediato |

### La secuencia ideal

**Fase 1 (10 epocas, lr=0.001):**
Solo la cabeza aprende a interpretar los features de ImageNet.
Loss baja rapido (~0.5 -> 0.1), accuracy sube a ~90-95%.

**Fase 2 (max 15 epocas, lr=0.0001):**
Descongelamos layer4 con learning rate 10x menor.
Ajusta detalles finos de los personajes Sanrio.
Accuracy puede llegar a 95-99%.

### Early Stopping

Si la loss de test no mejora en 5 epocas, paramos.
No tiene sentido seguir si ya no hay progreso.
""")

tl_code_optim = code_cell("""# ============================================
# Optimizadores para cada fase
# ============================================

criterion = nn.CrossEntropyLoss()

optimizer_fase1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG['LR_FASE1']
)

scheduler_fase1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_fase1, mode='min', factor=0.5, patience=3
)

print('Fase 1 lista: solo cabeza (lr={})'.format(CONFIG['LR_FASE1']))
print('Fase 2 (despues): fine-tuning layer4 (lr={})'.format(CONFIG['LR_FASE2']))
""")

tl_code_entrenar = code_cell("""# ============================================
# Funcion de entrenamiento (reutilizable)
# ============================================

def entrenar(modelo, train_loader, test_loader, criterion,
             optimizer, scheduler, num_epochs, device, nombre='Fase'):
    historial = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    mejor_loss = float('inf')
    patience = 0

    for epoch in range(num_epochs):
        modelo.train()
        train_loss = 0.0

        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            optimizer.zero_grad()
            predicciones = modelo(imagenes)
            loss = criterion(predicciones, etiquetas)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        historial['train_loss'].append(train_loss)

        modelo.eval()
        test_loss = 0.0
        correctos = 0
        total = 0

        with torch.no_grad():
            for imagenes, etiquetas in test_loader:
                imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
                outputs = modelo(imagenes)
                loss = criterion(outputs, etiquetas)
                test_loss += loss.item()
                _, predichos = torch.max(outputs, 1)
                total += etiquetas.size(0)
                correctos += (predichos == etiquetas).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100.0 * correctos / total
        historial['test_loss'].append(test_loss)
        historial['test_acc'].append(accuracy)

        lr_actual = optimizer.param_groups[0]['lr']
        print('{} [{:2d}/{}]  Train Loss: {:.4f}  Test Loss: {:.4f}  Acc: {:.2f}%  LR: {:.6f}'.format(
              nombre, epoch+1, num_epochs, train_loss, test_loss, accuracy, lr_actual))

        scheduler.step(test_loss)

        if test_loss < mejor_loss:
            mejor_loss = test_loss
            torch.save(modelo.state_dict(),
                       os.path.join(RUTA_MODELOS, 'mejor_tl.pth'))
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG['PATIENCE']:
                print('Early stopping en epoca {}'.format(epoch+1))
                break

    print('{} completada. Mejor accuracy: {:.2f}%'.format(nombre, max(historial['test_acc'])))
    return historial
""")

tl_code_ejecutar = code_cell("""# ============================================
# Ejecutar Fase 1 + Fase 2
# ============================================

print('=' * 60)
print('FASE 1: ENTRENANDO CABEZA')
print('=' * 60)

historial1 = entrenar(
    model, train_loader, test_loader, criterion,
    optimizer_fase1, scheduler_fase1, CONFIG['EPOCHS_FASE1'],
    device, 'Fase1'
)

print('\\n' + '=' * 60)
print('FASE 2: FINE-TUNING (layer4 descongelado)')
print('=' * 60)

for param in model.layer4.parameters():
    param.requires_grad = True

optimizer_fase2 = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': CONFIG['LR_FASE2']},
    {'params': model.fc.parameters(), 'lr': CONFIG['LR_FASE2']}
])

scheduler_fase2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_fase2, mode='min', factor=0.5, patience=3
)

historial2 = entrenar(
    model, train_loader, test_loader, criterion,
    optimizer_fase2, scheduler_fase2, CONFIG['EPOCHS_FASE2'],
    device, 'Fase2'
)

historial = {
    'train_loss': historial1['train_loss'] + historial2['train_loss'],
    'test_loss': historial1['test_loss'] + historial2['test_loss'],
    'test_acc': historial1['test_acc'] + historial2['test_acc']
}

print('\\nEntrenamiento completado.')
acc_inicial = max(historial['test_acc'])
print('Mejor accuracy: {:.2f}%'.format(acc_inicial))
""")

tl_md_graficas = md_cell("""---
## Visualizando el progreso

### Que muestra cada grafica

| Grafica | Que esperar |
|---------|-------------|
| Loss | Azul (train) y roja (test) bajan juntas |
| Accuracy | Sube y se estabiliza |
| Comparacion | Fase 2 debe superar Fase 1 |
| Accuracy final | El numero grande |

La linea punteada separa Fase 1 (izquierda) de Fase 2 (derecha).
Nota el "salto" al empezar Fase 2.
""")

tl_code_graficas = code_cell("""# ============================================
# Graficas con separacion de fases
# ============================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))

sep = len(historial1['train_loss'])

ax1.plot(range(1, len(historial['train_loss'])+1), historial['train_loss'],
         'b-', label='Train Loss', linewidth=2)
ax1.plot(range(1, len(historial['test_loss'])+1), historial['test_loss'],
         'r-', label='Test Loss', linewidth=2)
ax1.axvline(x=sep, color='gray', linestyle='--', alpha=0.5, label='Fine-tuning')
ax1.set_title('Perdida (Loss)')
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(historial['test_acc'])+1), historial['test_acc'],
         'g-', linewidth=2)
ax2.axvline(x=sep, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Accuracy en Test')
ax2.set_xlabel('Epoca')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.3)

ax3.bar(['Fase 1', 'Fase 2'],
        [max(historial1['test_acc']), max(historial2['test_acc'])],
        color=['steelblue', 'coral'])
ax3.set_title('Mejor Accuracy por Fase')
ax3.set_ylabel('Accuracy (%)')
ax3.set_ylim(0, 105)

ultimo_acc = historial['test_acc'][-1]
ax4.text(0.5, 0.6, '{:.2f}%'.format(ultimo_acc),
         fontsize=42, ha='center', fontweight='bold', color='green')
ax4.text(0.5, 0.3, 'Accuracy Final',
         fontsize=16, ha='center', color='gray')
ax4.axis('off')

plt.tight_layout()
plt.show()
""")

# ============================================================
# ENSAMBLAR NOTEBOOK 1: CNN_Sanrio.ipynb
# ============================================================

cnn_cells = [
    shared_md_contexto,
    shared_code_setup,
    cnn_code_experimento,
    shared_md_datos,
    shared_code_dataloader,
    shared_code_visualizacion,
    cnn_md_arquitectura,
    cnn_code_modelo,
    cnn_md_entrenamiento,
    cnn_code_entrenar,
    cnn_code_loop,
    shared_code_graficas,
    shared_md_metricas,
    shared_code_evaluacion,
    shared_md_errores,
    shared_code_errores,
    shared_md_inferencia,
    shared_code_inferencia,
    code_cell("NOMBRE_MODELO = 'cnn_sanrio'; " + shared_code_guardar['source'][0]),
    shared_code_footer,
]

# Fix guardar cell for CNN
cnn_cells[-2] = code_cell("""# ============================================
# Guardar modelo CNN
# ============================================

from datetime import datetime

NOMBRE_MODELO = 'cnn_sanrio'
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

torch.save(model.state_dict(),
           os.path.join(RUTA_MODELOS, NOMBRE_MODELO + '_{}.pth'.format(timestamp)))
torch.save(model.state_dict(),
           os.path.join(RUTA_MODELOS, NOMBRE_MODELO + '_final.pth'))

with open(os.path.join(RUTA_MODELOS, 'clases_sanrio.json'), 'w') as f:
    json.dump({
        'classes': clases,
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': NUM_CLASES,
        'fecha': timestamp,
        'autora': AUTORA,
    }, f, indent=2)

print('Modelo CNN Sanrio guardado.')
print('Clases:', clases)
""")

notebook_1 = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU"
    },
    "cells": cnn_cells
}
out_1 = os.path.join(BASE, 'CNN_Sanrio{}.ipynb'.format(SUFIJO))
with open(out_1, 'w', encoding='utf-8') as f:
    json.dump(notebook_1, f, indent=1, ensure_ascii=False)
print('Generado:', out_1, 'con', len(cnn_cells), 'celdas')

# ============================================================
# ENSAMBLAR NOTEBOOK 2: Transfer_Learning_Sanrio.ipynb
# ============================================================

tl_cells = [
    shared_md_contexto,
    shared_code_setup,
    code_cell("""# (Ajuste: batch_size menor para ResNet18)
CONFIG['BATCH_SIZE'] = 16
CONFIG['EPOCHS_FASE1'] = 10
CONFIG['EPOCHS_FASE2'] = 15
CONFIG['LR_FASE1'] = 0.001
CONFIG['LR_FASE2'] = 0.0001
CONFIG['DROPOUT'] = 0.3
CONFIG['PATIENCE'] = 5
print('CONFIG (ajustado para TL):', CONFIG)
"""),
    tl_code_experimento,
    shared_md_datos,
    shared_code_dataloader,
    shared_code_visualizacion,
    tl_md_transfer,
    tl_code_modelo,
    tl_md_fases,
    tl_code_optim,
    tl_code_entrenar,
    tl_code_ejecutar,
    shared_code_guardar_exp,
    tl_md_graficas,
    tl_code_graficas,
    shared_code_comparativa,
    shared_md_metricas,
    shared_code_evaluacion,
    shared_md_errores,
    shared_code_errores,
    shared_md_inferencia,
    shared_code_inferencia,
    code_cell("""# ============================================
# Guardar modelo Transfer Learning
# ============================================

from datetime import datetime

NOMBRE_MODELO = 'tl_sanrio'
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

torch.save(model.state_dict(),
           os.path.join(RUTA_MODELOS, NOMBRE_MODELO + '_{}.pth'.format(timestamp)))
torch.save(model.state_dict(),
           os.path.join(RUTA_MODELOS, NOMBRE_MODELO + '_final.pth'))

with open(os.path.join(RUTA_MODELOS, 'clases_sanrio.json'), 'w') as f:
    json.dump({
        'classes': clases,
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': NUM_CLASES,
        'fecha': timestamp,
        'autora': AUTORA,
    }, f, indent=2)

print('Modelo Transfer Learning Sanrio guardado.')
print('Clases:', clases)
"""),
    shared_code_footer,
]

notebook_2 = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU"
    },
    "cells": tl_cells
}
out_2 = os.path.join(BASE, 'Transfer_Learning_Sanrio{}.ipynb'.format(SUFIJO))
with open(out_2, 'w', encoding='utf-8') as f:
    json.dump(notebook_2, f, indent=1, ensure_ascii=False)
print('Generado:', out_2, 'con', len(tl_cells), 'celdas')

print('\\nListo! Ambos notebooks generados con estilo biclase compacto.')
