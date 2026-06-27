#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scraping v3: 6 queries por clase para maximizar dataset (~600-700 c/u).
"""
import os, time
from icrawler.builtin import BingImageCrawler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETAS = {
    'kuromi':   os.path.join(BASE_DIR, 'dataset', 'train', 'kuromi'),
    'cinnamon': os.path.join(BASE_DIR, 'dataset', 'train', 'cinnamon'),
}

for r in CARPETAS.values():
    os.makedirs(r, exist_ok=True)

def descargar(query, destino, max_num=200, offset=0):
    crawler = BingImageCrawler(
        storage={'root_dir': destino},
        parser_threads=2,
        downloader_threads=4,
    )
    crawler.crawl(keyword=query, max_num=max_num, min_size=(200, 200),
                  file_idx_offset=offset)

QUERIES = {
    'kuromi': [
        'Kuromi Sanrio figure',
        'Kuromi Sanrio anime',
        'Kuromi plush toy',
        'Kuromi merchandise',
        'Kuromi doll collection',
        'Kuromi kawaii figurine',
    ],
    'cinnamon': [
        'Cinnamoroll Sanrio figure',
        'Cinnamoroll Sanrio anime',
        'Cinnamoroll plush toy',
        'Cinnamoroll merchandise',
        'Cinnamoroll doll collection',
        'Cinnamoroll kawaii figurine',
    ],
}

for clase, queries in QUERIES.items():
    destino = CARPETAS[clase]
    offset = 0
    for i, q in enumerate(queries):
        print(f'\n--- Descargando ({i+1}/{len(queries)}) "{q}" -> {clase} ---')
        descargar(q, destino, max_num=150, offset=offset)
        offset += 200
        time.sleep(2)

    total = len(os.listdir(destino))
    print(f'\n>>> Total {clase}: {total} imagenes')

print('\n=== DESCARGA COMPLETADA ===')
for clase, ruta in CARPETAS.items():
    total = len(os.listdir(ruta))
    print(f'  {clase}: {total} imagenes en {ruta}')
