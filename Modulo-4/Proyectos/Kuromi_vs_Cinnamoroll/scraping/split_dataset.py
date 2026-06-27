#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split dataset en train/test (80/20) preservando estructura ImageFolder.

Uso:
    python scraping/split_dataset.py

Luego de ejecutar, revisa que los archivos esten en las carpetas correctas.
"""
import os, shutil, random

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
TRAIN = os.path.join(BASE, 'train')
TEST  = os.path.join(BASE, 'test')
SPLIT = 0.8  # 80% train, 20% test
SEED = 42

random.seed(SEED)

for clase in os.listdir(TRAIN):
    src = os.path.join(TRAIN, clase)
    if not os.path.isdir(src):
        continue

    # Solo dividir imagenes de scraping (sin prefijo de modelo)
    archivos = [f for f in os.listdir(src) if not any(f.startswith(p) for p in ['Mod','mod'])]
    random.shuffle(archivos)

    n_train = int(len(archivos) * SPLIT)
    test_files  = archivos[n_train:]

    dst_test  = os.path.join(TEST, clase)
    os.makedirs(dst_test, exist_ok=True)

    for f in test_files:
        shutil.move(os.path.join(src, f), os.path.join(dst_test, f))

    total_train = len(os.listdir(src))
    total_test  = len(os.listdir(dst_test))
    print(f'{clase}: {total_train} train + {total_test} test = {total_train+total_test} total (scraping dividido, modelos intactos)')

print('\nSplit completado.')
