#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scraping de imagenes: Kuromi vs Cinnamoroll

Este script descarga imagenes de Google Images para complementar
el dataset de fotos de tus figuras.

Uso:
    python scraping_imagenes.py

Requisitos:
    pip install icrawler
"""

import os, sys, time

# Crear carpetas destino
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETAS = {
    'kuromi':  os.path.join(BASE_DIR, 'dataset', 'train', 'kuromi'),
    'cinnamon': os.path.join(BASE_DIR, 'dataset', 'train', 'cinnamon'),
}

for nombre, ruta in CARPETAS.items():
    os.makedirs(ruta, exist_ok=True)
    print('Carpeta lista: ' + ruta)


# ── Descargar con icrawler ────────────────────────────────────────────
def descargar_imagenes(query, destino, max_num=200):
    """
    Descarga imagenes de Google Images usando icrawler.

    Args:
        query (str): Termino de busqueda (ej: 'Kuromi Sanrio figure').
        destino (str): Carpeta donde guardar las imagenes.
        max_num (int): Numero maximo de imagenes a descargar.
    """
    try:
        from icrawler.builtin import GoogleImageCrawler
    except ImportError:
        print('\nERROR: icrawler no esta instalado.')
        print('Instalalo con: pip install icrawler')
        print('Luego ejecuta este script de nuevo.\n')
        sys.exit(1)

    crawler = GoogleImageCrawler(
        storage={'root_dir': destino},
        parser_threads=2,
        downloader_threads=4,
    )

    crawler.crawl(
        keyword=query,
        max_num=max_num,
        min_size=(200, 200),
        max_size=(1920, 1080),
        file_idx_offset=0,
    )


print('\n--- Descargando imagenes de Kuromi ---')
descargar_imagenes('Kuromi Sanrio figure', CARPETAS['kuromi'], max_num=200)

time.sleep(2)

print('\n--- Descargando imagenes de Cinnamoroll ---')
descargar_imagenes('Cinnamoroll Sanrio figure', CARPETAS['cinnamon'], max_num=200)

time.sleep(2)

print('\n--- Descargando imagenes extra de Kuromi ---')
descargar_imagenes('Kuromi Sanrio anime', CARPETAS['kuromi'], max_num=200)

time.sleep(2)

print('\n--- Descargando imagenes extra de Cinnamoroll ---')
descargar_imagenes('Cinnamoroll Sanrio anime', CARPETAS['cinnamon'], max_num=200)

print('\nDescarga completada.')
print('Revisa las carpetas y elimina manualmente las imagenes que no correspondan.')
