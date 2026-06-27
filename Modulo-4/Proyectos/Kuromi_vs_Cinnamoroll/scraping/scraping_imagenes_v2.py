#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scraping de imagenes v2 - Bing Image Crawler (mas estable que Google).
"""
import os, sys, time
from icrawler.builtin import BingImageCrawler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETAS = {
    'kuromi':   os.path.join(BASE_DIR, 'dataset', 'train', 'kuromi'),
    'cinnamon': os.path.join(BASE_DIR, 'dataset', 'train', 'cinnamon'),
}

for nombre, ruta in CARPETAS.items():
    os.makedirs(ruta, exist_ok=True)
    print('Carpeta lista:', ruta)

def descargar(query, destino, max_num=200):
    crawler = BingImageCrawler(
        storage={'root_dir': destino},
        parser_threads=2,
        downloader_threads=4,
    )
    crawler.crawl(
        keyword=query,
        max_num=max_num,
        min_size=(200, 200),
        max_size=(1920, 1080),
    )

for clase, (query1, query2) in {
    'kuromi':   ('Kuromi Sanrio figure', 'Kuromi Sanrio anime'),
    'cinnamon': ('Cinnamoroll Sanrio figure', 'Cinnamoroll Sanrio anime'),
}.items():
    ruta = CARPETAS[clase]
    for q in [query1, query2]:
        print(f'\n--- Descargando "{q}" -> {clase} ---')
        descargar(q, ruta, max_num=200)
        time.sleep(2)

print('\nDescarga completada.')
print('Revisa las carpetas y elimina manualmente las imagenes que no correspondan.')
