#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_12clases.py — Diana Blanco (MorritaConP1to)
==================================================
Split 80/20 estratificado desde dataset/raw_v3/ hacia:
    dataset/train_v3/
    dataset/test_v3/

El notebook v3 solo necesita cambiar TRAIN_DIR/TEST_DIR para apuntar aquí.
"""
import os, shutil, random
from pathlib import Path

random.seed(42)

BASE       = Path(__file__).resolve().parent.parent / 'dataset'
RAW        = BASE / 'raw_v3'
TRAIN      = BASE / 'train_v3'
TEST       = BASE / 'test_v3'
SPLIT      = 0.80
MIN_TRAIN  = 200   # Avisa si una clase quedó con menos de esto

if not RAW.is_dir():
    raise FileNotFoundError(
        f"No se encuentra {RAW}\n"
        f"Ejecuta primero scraping_12clases.py y preprocesar_raw.py"
    )

# Limpiar destinos si ya existen (para re-ejecutar limpio)
for d in [TRAIN, TEST]:
    if d.exists():
        shutil.rmtree(d)
        print(f"  ({d.name}/ anterior eliminado)")

print()
print("=" * 55)
print("  SPLIT 80/20 — 12 CLASES SANRIO")
print("=" * 55)
print(f"  Fuente: {RAW}")
print(f"  Train:  {TRAIN}")
print(f"  Test:   {TEST}")
print()

total_train = 0
total_test  = 0
advertencias = []

for clase_dir in sorted(RAW.iterdir()):
    if not clase_dir.is_dir():
        continue
    clase = clase_dir.name

    # Solo JPG (preprocesar_raw.py ya convirtió todo)
    archivos = sorted([
        f for f in clase_dir.iterdir()
        if f.is_file() and f.suffix.lower() == '.jpg'
    ])

    if not archivos:
        print(f"  {clase}: SIN ARCHIVOS, saltando")
        continue

    random.shuffle(archivos)
    n_train = int(len(archivos) * SPLIT)
    n_test  = len(archivos) - n_train

    # Garantizar al menos 1 imagen en test
    if n_test == 0:
        n_train -= 1
        n_test   = 1

    train_files = archivos[:n_train]
    test_files  = archivos[n_train:]

    dst_train = TRAIN / clase
    dst_test  = TEST  / clase
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_test.mkdir(parents=True,  exist_ok=True)

    for f in train_files:
        shutil.copy2(f, dst_train / f.name)   # copy2 preserva metadatos
    for f in test_files:
        shutil.copy2(f, dst_test  / f.name)

    total_train += n_train
    total_test  += n_test

    estado = ""
    if n_train < MIN_TRAIN:
        estado = f"  ⚠️  POCAS IMÁGENES ({n_train} train)"
        advertencias.append(clase)

    print(f"  {clase:20s}: {n_train:3d} train + {n_test:3d} test = "
          f"{len(archivos):3d} total{estado}")

print()
print("=" * 55)
print(f"  TOTAL TRAIN: {total_train} imágenes")
print(f"  TOTAL TEST:  {total_test} imágenes")
print(f"  CLASES:      {len(list(TRAIN.iterdir()))} clases")
print()
if advertencias:
    print(f"  ⚠️  Clases con pocas imágenes (<{MIN_TRAIN} train):")
    for c in advertencias:
        print(f"     - {c}: considera scrapear más")
    print()
print("  Siguiente paso:")
print("  Abre Transfer_Learning_Sanrio_v3.ipynb")
print("  Cambia DATASET_VERSION = 'v3' en la celda de config")
print("=" * 55)
