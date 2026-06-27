#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocesar dataset: validar integridad, convertir a JPEG, eliminar corruptas.
"""
import os
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
CARPETAS = [
    os.path.join(BASE, 'train', 'kuromi'),
    os.path.join(BASE, 'train', 'cinnamon'),
    os.path.join(BASE, 'test', 'kuromi'),
    os.path.join(BASE, 'test', 'cinnamon'),
]

MIN_SIZE_BYTES = 5120  # 5 KB
EXT_VALIDAS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

for carpeta in CARPETAS:
    if not os.path.isdir(carpeta):
        print(f'[SKIP] No existe: {carpeta}')
        continue

    archivos = [f for f in os.listdir(carpeta) if os.path.splitext(f)[1].lower() in EXT_VALIDAS]
    total = len(archivos)
    ok = 0
    corruptas = 0
    convertidas = 0
    muy_pequenas = 0

    for f in archivos:
        ruta = os.path.join(carpeta, f)
        tamano = os.path.getsize(ruta)

        if tamano < MIN_SIZE_BYTES:
            os.remove(ruta)
            muy_pequenas += 1
            continue

        try:
            with Image.open(ruta) as img:
                img.verify()
            with Image.open(ruta) as img:
                img.load()
        except:
            os.remove(ruta)
            corruptas += 1
            continue

        ext = os.path.splitext(f)[1].lower()
        if ext != '.jpg':
            nueva_ruta = os.path.splitext(ruta)[0] + '.jpg'
            if ruta != nueva_ruta:
                try:
                    with Image.open(ruta) as img:
                        img = img.convert('RGB')
                        img.save(nueva_ruta, 'JPEG')
                    os.remove(ruta)
                    convertidas += 1
                except:
                    os.remove(ruta)
                    corruptas += 1
                    continue

        ok += 1

    print(f'{os.path.basename(carpeta)} ({os.path.dirname(carpeta).split(os.sep)[-1]}): {ok} validas, {corruptas} corruptas, {convertidas} convertidas, {muy_pequenas} muy pequenas (de {total})')
