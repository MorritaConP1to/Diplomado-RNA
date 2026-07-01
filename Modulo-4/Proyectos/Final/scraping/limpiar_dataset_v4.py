# ============================================
# limpiar_dataset_v4.py
# 
# Toma dataset v3 (train_v3/ + test_v3/), lo copia
# a v4 eliminando:
#   1. Duplicados cross-class (misma imagen en 2 personajes)
#   2. Data leakage (misma imagen en train y test)
#   3. Imagenes <200 px en cualquier dimension
#
# Uso: python scraping/limpiar_dataset_v4.py
# ============================================

import os, sys, json, hashlib, shutil, time
from collections import defaultdict
from PIL import Image

RUTA_BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DATASET= os.path.join(RUTA_BASE, 'dataset')
ORIGEN_TRAIN= os.path.join(RUTA_DATASET, 'train_v3')
ORIGEN_TEST = os.path.join(RUTA_DATASET, 'test_v3')
DESTINO_TRAIN= os.path.join(RUTA_DATASET, 'train_v4')
DESTINO_TEST = os.path.join(RUTA_DATASET, 'test_v4')
REPORTE     = os.path.join(RUTA_DATASET, 'limpieza_v4_reporte.txt')

MIN_PX      = 200   # Imagenes con dimension menor a esto se eliminan

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def md5_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def contar_imgs(directorio):
    total = 0
    if not os.path.isdir(directorio):
        return 0
    for carpeta in os.listdir(directorio):
        ruta = os.path.join(directorio, carpeta)
        if os.path.isdir(ruta):
            total += len([f for f in os.listdir(ruta) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])
    return total

# ── Validar que existen los directorios fuente ──
if not os.path.isdir(ORIGEN_TRAIN) or not os.path.isdir(ORIGEN_TEST):
    log(f'ERROR: No se encuentra train_v3/ o test_v3/ en {RUTA_DATASET}')
    log('Ejecuta primero scraping/scraping_12clases.py + preprocesar_raw_v3.py + split_12clases.py')
    sys.exit(1)

log(f'Inicio limpieza: v3 → v4')
log(f'Origen train:   {ORIGEN_TRAIN} ({contar_imgs(ORIGEN_TRAIN)} img)')
log(f'Origen test:    {ORIGEN_TEST} ({contar_imgs(ORIGEN_TEST)} img)')
log(f'Min dimension:  {MIN_PX}px')
log('='*60)

# ── 1. COPIA COMPLETA DE v3 A v4 ──
if os.path.isdir(DESTINO_TRAIN):
    shutil.rmtree(DESTINO_TRAIN)
if os.path.isdir(DESTINO_TEST):
    shutil.rmtree(DESTINO_TEST)

def copiar_tree(origen, destino):
    os.makedirs(destino, exist_ok=True)
    for carpeta in os.listdir(origen):
        ruta_carpeta = os.path.join(origen, carpeta)
        if not os.path.isdir(ruta_carpeta):
            continue
        dst_carpeta = os.path.join(destino, carpeta)
        os.makedirs(dst_carpeta, exist_ok=True)
        for archivo in os.listdir(ruta_carpeta):
            if archivo.lower().endswith(('.jpg','.jpeg','.png','.webp')):
                shutil.copy2(os.path.join(ruta_carpeta, archivo),
                             os.path.join(dst_carpeta, archivo))

log('Copiando train_v3 → train_v4...')
copiar_tree(ORIGEN_TRAIN, DESTINO_TRAIN)
log('Copiando test_v3 → test_v4...')
copiar_tree(ORIGEN_TEST, DESTINO_TEST)
log(f'Copia completa ({contar_imgs(DESTINO_TRAIN)} train, {contar_imgs(DESTINO_TEST)} test)')
log('')

# ── 2. DETECTAR DUPLICADOS CROSS-CLASS ──
log('Paso 1: Detectando duplicados cross-class...')
hash_totales = {}   # md5 → [(ruta, split, clase)]
eliminados_dup = 0

for split, directorio in [('train', DESTINO_TRAIN), ('test', DESTINO_TEST)]:
    for carpeta in sorted(os.listdir(directorio)):
        ruta_carpeta = os.path.join(directorio, carpeta)
        if not os.path.isdir(ruta_carpeta):
            continue
        for archivo in os.listdir(ruta_carpeta):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            if not os.path.isfile(ruta_archivo):
                continue
            try:
                h = md5_hash(ruta_archivo)
            except:
                continue
            if h in hash_totales:
                # Duplicado encontrado
                prev_split, prev_clase, prev_ruta = hash_totales[h]
                log(f'  Duplicado: {carpeta}/{archivo}  (tambien en {prev_clase}/{prev_split})')
                os.remove(ruta_archivo)
                eliminados_dup += 1
            else:
                hash_totales[h] = (split, carpeta, ruta_archivo)

log(f'  Eliminados: {eliminados_dup} duplicados cross-class')
log('')

# ── 3. DETECTAR DATA LEAKAGE (misma imagen en train y test) ──
log('Paso 2: Detectando data leakage (train ↔ test)...')
hash_train = {}   # md5 → (clase, ruta)
leakage_eliminados = 0

# Primero indexar todas las imagenes de train
for carpeta in sorted(os.listdir(DESTINO_TRAIN)):
    ruta_carpeta = os.path.join(DESTINO_TRAIN, carpeta)
    if not os.path.isdir(ruta_carpeta):
        continue
    for archivo in os.listdir(ruta_carpeta):
        ruta_archivo = os.path.join(ruta_carpeta, archivo)
        if not os.path.isfile(ruta_archivo):
            continue
        try:
            hash_train[md5_hash(ruta_archivo)] = (carpeta, ruta_archivo)
        except:
            continue

# Ahora revisar test contra train
for carpeta in sorted(os.listdir(DESTINO_TEST)):
    ruta_carpeta = os.path.join(DESTINO_TEST, carpeta)
    if not os.path.isdir(ruta_carpeta):
        continue
    for archivo in os.listdir(ruta_carpeta):
        ruta_archivo = os.path.join(ruta_carpeta, archivo)
        if not os.path.isfile(ruta_archivo):
            continue
        try:
            h = md5_hash(ruta_archivo)
        except:
            continue
        if h in hash_train:
            train_clase, train_ruta = hash_train[h]
            log(f'  Leakage: {carpeta}/{archivo} (test) = {train_clase}/{os.path.basename(train_ruta)} (train)')
            # Eliminar del test (se quedo con la version de train)
            os.remove(ruta_archivo)
            leakage_eliminados += 1
            # Tambien eliminar la version de train si estan en clases DIFERENTES
            # (contaminacion de etiquetas)
            if train_clase != carpeta:
                log(f'    → Etiqueta INCORRECTA: train dice {train_clase}, test dice {carpeta}')
                log(f'    → Eliminando ambas copias (contaminan ambas clases)')
                os.remove(train_ruta)
                leakage_eliminados += 1

log(f'  Eliminados: {leakage_eliminados} imagenes con leakage')
log('')

# ── 4. ELIMINAR IMAGENES <200px ──
log(f'Paso 3: Filtrando imagenes <{MIN_PX}px...')
pequenas_eliminadas = 0

for split, directorio in [('train', DESTINO_TRAIN), ('test', DESTINO_TEST)]:
    for carpeta in sorted(os.listdir(directorio)):
        ruta_carpeta = os.path.join(directorio, carpeta)
        if not os.path.isdir(ruta_carpeta):
            continue
        for archivo in list(os.listdir(ruta_carpeta)):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            if not os.path.isfile(ruta_archivo):
                continue
            try:
                with Image.open(ruta_archivo) as img:
                    w, h = img.size
                    if w < MIN_PX or h < MIN_PX:
                        os.remove(ruta_archivo)
                        pequenas_eliminadas += 1
                        if pequenas_eliminadas <= 30:
                            log(f'  {carpeta}/{archivo} ({w}x{h})')
            except Exception as e:
                log(f'  Error al abrir {carpeta}/{archivo}: {e}')
                os.remove(ruta_archivo)
                pequenas_eliminadas += 1

log(f'  Eliminadas: {pequenas_eliminadas} imagenes <{MIN_PX}px')
log('')

# ── 5. ELIMINAR CARPETAS VACIAS ──
for directorio in [DESTINO_TRAIN, DESTINO_TEST]:
    for carpeta in list(os.listdir(directorio)):
        ruta = os.path.join(directorio, carpeta)
        if os.path.isdir(ruta) and not os.listdir(ruta):
            os.rmdir(ruta)
            log(f'  Carpeta vacia eliminada: {carpeta}')

# ── 6. RESUMEN ──
total_train = contar_imgs(DESTINO_TRAIN)
total_test  = contar_imgs(DESTINO_TEST)
clases_train = sorted(os.listdir(DESTINO_TRAIN)) if os.path.isdir(DESTINO_TRAIN) else []
clases_test  = sorted(os.listdir(DESTINO_TEST)) if os.path.isdir(DESTINO_TEST) else []

log('='*60)
log('RESUMEN LIMPIEZA v4')
log('='*60)
log(f'  Train: {total_train} imagenes en {len(clases_train)} clases')
log(f'  Test:  {total_test} imagenes en {len(clases_test)} clases')
log(f'  Total: {total_train + total_test} imagenes')
log(f'')
log(f'  Duplicados cross-class eliminados: {eliminados_dup}')
log(f'  Data leakage eliminado:            {leakage_eliminados}')
log(f'  Imagenes <{MIN_PX}px eliminadas:      {pequenas_eliminadas}')
log(f'')
log('Distribucion train_v4:')
for carpeta in clases_train:
    n = len([f for f in os.listdir(os.path.join(DESTINO_TRAIN, carpeta))
             if f.lower().endswith(('.jpg','.jpeg'))])
    barra = '#' * max(1, n // 8)
    log(f'  {carpeta:20s}: {n:4d} img  {barra}')
log(f'')
log('Distribucion test_v4:')
for carpeta in clases_test:
    n = len([f for f in os.listdir(os.path.join(DESTINO_TEST, carpeta))
             if f.lower().endswith(('.jpg','.jpeg'))])
    log(f'  {carpeta:20s}: {n:4d} img')

log(f'')
log(f'Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}')

# Guardar reporte
with open(REPORTE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))

print(f'\nReporte guardado: {REPORTE}')
