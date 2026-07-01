#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocesar dataset RAW: validar integridad, convertir a JPEG, eliminar corruptas.
Procesa todas las subcarpetas dentro de dataset/raw/.
"""
import os
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'raw_v3')

MIN_SIZE_BYTES = 10_000  # 10 KB — consistente con filtro del scraper
EXT_VALIDAS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

for clase in sorted(os.listdir(BASE)):
    carpeta = os.path.join(BASE, clase)
    if not os.path.isdir(carpeta):
        continue

    archivos = [f for f in os.listdir(carpeta)
                if os.path.isfile(os.path.join(carpeta, f))
                and os.path.splitext(f)[1].lower() in EXT_VALIDAS]
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

    print(f'{clase}: {ok} validas, {corruptas} corruptas, {convertidas} convertidas, {muy_pequenas} muy pequenas (de {total})')

print('\nPreprocesamiento completado.')
