#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split dataset multi-clase: mueve 80% a train/ y 20% a test/ desde raw/.
Preserva la estructura ImageFolder.

Uso:
    python scraping/split_multiclase.py
"""
import os, shutil, random

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
RAW = os.path.join(BASE, 'raw')
TRAIN = os.path.join(BASE, 'train')
TEST = os.path.join(BASE, 'test')
SPLIT = 0.8
SEED = 42

random.seed(SEED)

for clase in sorted(os.listdir(RAW)):
    src = os.path.join(RAW, clase)
    if not os.path.isdir(src):
        continue

    archivos = [f for f in os.listdir(src)
                if os.path.isfile(os.path.join(src, f))
                and f.lower().endswith('.jpg')]
    random.shuffle(archivos)

    if len(archivos) == 0:
        print(f'{clase}: sin archivos, saltando')
        continue

    n_train = int(len(archivos) * SPLIT)
    train_files = archivos[:n_train]
    test_files = archivos[n_train:]

    dst_train = os.path.join(TRAIN, clase)
    dst_test = os.path.join(TEST, clase)
    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_test, exist_ok=True)

    for f in train_files:
        shutil.move(os.path.join(src, f), os.path.join(dst_train, f))
    for f in test_files:
        shutil.move(os.path.join(src, f), os.path.join(dst_test, f))

    print(f'{clase}: {len(train_files)} train + {len(test_files)} test = {len(archivos)} total')

print('\nSplit completado.')
print('Estructura resultante:')
for split in ['train', 'test']:
    ruta = os.path.join(BASE, split)
    if os.path.isdir(ruta):
        total_clases = 0
        total_imgs = 0
        for clase in sorted(os.listdir(ruta)):
            carpeta = os.path.join(ruta, clase)
            if os.path.isdir(carpeta):
                n = len([f for f in os.listdir(carpeta) if f.endswith('.jpg')])
                print(f'  {split}/{clase}: {n} imagenes')
                total_clases += 1
                total_imgs += n
        print(f'  Total {split}: {total_clases} clases, {total_imgs} imagenes')
