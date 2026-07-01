#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scraping_12clases.py — Diana Blanco (MorritaConP1to)
=====================================================
Scraper MEJORADO para las 12 clases finales del clasificador Sanrio.

MEJORAS vs versiones anteriores:
  - Queries quirúrgicas: fuerzan al personaje como sujeto principal
  - Filtros de tamaño mínimo más estrictos (10 KB vs 3 KB)
  - Filtro de resolución mínima: descarta thumbnails (< 100x100 px)
  - Meta-filtro de aspecto: descarta banners horizontales muy anchos
  - Más queries por personaje (6 en lugar de 4) para más variedad
  - max_imgs=80 por query → ~480 imágenes brutas por personaje
    (después de preprocesar_raw.py quedan ~300-350 limpias)
  - Pausa aleatoria entre queries para evitar rate-limit de Google

INSTRUCCIONES:
  1. CIERRA Brave completamente
  2. Activa tu entorno: conda activate diplomado-redes
  3. Ejecuta: python scraping/scraping_12clases.py
  4. Si pide captcha en la primera búsqueda, resuélvelo una vez
  5. El resto avanza solo (~4-6 horas para las 12 clases)

DESPUÉS:
  python scraping/preprocesar_raw.py   ← valida + convierte a JPG
  python scraping/split_12clases.py    ← split 80/20 → train_v3/ test_v3/

CLASES (12 — ranking oficial Sanrio 2026 + distintividad visual):
  pompompurin, cinnamoroll, pochacco, kuromi, hello_kitty,
  my_melody, keroppi, badtz_maru, gudetama, aggretsuko,
  kirimichan, hangyodon
"""

import os, time, random, hashlib, requests, io, urllib.parse
from datetime import datetime
from playwright.sync_api import sync_playwright
from PIL import Image

# ── Rutas ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, 'dataset', 'raw_v3')   # carpeta nueva, no pisa raw/
LOG_DIR  = os.path.join(BASE_DIR, 'scraping', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

BRAVE_PATH    = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
BRAVE_PROFILE = os.path.expandvars(
    r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\User Data"
)

# ── Filtros de calidad ──────────────────────────────────────────
MIN_BYTES      = 10_000   # 10 KB mínimo (antes era 3 KB → entraban thumbnails)
MIN_DIMENSION  = 120      # ancho Y alto mínimos en píxeles
MAX_RATIO      = 4.0      # descartar banners: ancho/alto > 4 o alto/ancho > 4

# ── 12 clases con queries quirúrgicas ──────────────────────────
# Estrategia:
#   - "solo" / "alone" → evita imágenes grupales
#   - "white background" / "transparent background" → fondo limpio
#   - "official art" → arte oficial Sanrio, más limpio que fanart
#   - "plush" / "figure" → objetos físicos donde el personaje ES el sujeto
#   - Búsquedas en japonés → accede a fuentes distintas (Pixiv, blogs japoneses)
PERSONAJES = [

    ('pompompurin', [
        'Pompompurin Sanrio official art white background alone',
        'Pompompurin Sanrio character solo transparent background',
        'ポムポムプリン サンリオ 公式 単体 白背景',          # japonés
        'Pompompurin golden retriever plush figure Sanrio only',
        'Pompompurin Sanrio sticker icon single character',
        'Pompompurin Sanrio close up face merchandise',
    ]),

    ('cinnamon', [
        'Cinnamoroll Sanrio official art white background alone',
        'Cinnamoroll Sanrio character solo transparent background',
        'シナモロール サンリオ 公式 単体 白背景',
        'Cinnamoroll blue puppy plush figure Sanrio only',
        'Cinnamoroll Sanrio sticker icon single character',
        'Cinnamoroll Sanrio close up face merchandise',
    ]),

    ('pochacco', [
        'Pochacco Sanrio official art white background alone',
        'Pochacco Sanrio character solo transparent background',
        'ポチャッコ サンリオ 公式 単体 白背景',
        'Pochacco dog plush figure Sanrio only white',
        'Pochacco Sanrio sticker icon single character',
        'Pochacco Sanrio sports cap dog merchandise',
    ]),

    ('kuromi', [
        'Kuromi Sanrio official art white background alone',
        'Kuromi Sanrio character solo transparent background',
        'クロミ サンリオ 公式 単体 白背景',
        'Kuromi black skull jester plush figure Sanrio only',
        'Kuromi Sanrio sticker icon single character',
        'Kuromi Sanrio dark gothic close up merchandise',
    ]),

    ('hello_kitty', [
        'Hello Kitty Sanrio official art white background alone',
        'Hello Kitty Sanrio character solo no mouth bow',
        'ハローキティ サンリオ 公式 単体 白背景',
        'Hello Kitty red bow plush figure Sanrio only',
        'Hello Kitty Sanrio sticker icon single character',
        'Hello Kitty Sanrio classic original design merchandise',
    ]),

    ('my_melody', [
        'My Melody Sanrio official art white background alone',
        'My Melody Sanrio character solo hood bunny',
        'マイメロディ サンリオ 公式 単体 白背景',
        'My Melody pink hood rabbit plush figure Sanrio only',
        'My Melody Sanrio sticker icon single character',
        'My Melody Sanrio close up face pink hood merchandise',
    ]),

    ('keroppi', [
        'Keroppi Sanrio official art white background alone',
        'Keroppi Sanrio character solo frog green',
        'けろっぴ サンリオ 公式 単体 白背景',
        'Keroppi frog plush figure Sanrio only white',
        'Keroppi Sanrio sticker icon single character frog',
        'Keroppi Sanrio close up face green frog merchandise',
    ]),

    ('badtz_maru', [
        'Badtz-Maru Sanrio official art white background alone',
        'Badtz-Maru Sanrio character solo penguin mohawk',
        'バッドばつ丸 サンリオ 公式 単体 白背景',
        'Badtz-Maru penguin plush figure Sanrio only white',
        'Badtz Maru Sanrio sticker icon single character',
        'Badtz-Maru Sanrio close up face punk penguin merchandise',
    ]),

    ('gudetama', [
        'Gudetama Sanrio official art white background alone',
        'Gudetama Sanrio character solo lazy egg yolk',
        'ぐでたま サンリオ 公式 単体 白背景',
        'Gudetama egg plush figure Sanrio only white',
        'Gudetama Sanrio sticker icon single lazy egg character',
        'Gudetama Sanrio close up yellow egg merchandise',
    ]),

    ('aggretsuko', [
        'Aggretsuko Sanrio official art white background alone',
        'Aggretsuko Retsuko Sanrio character solo red panda',
        'アグレッシブ烈子 サンリオ 公式 単体 白背景',
        'Aggretsuko red panda plush figure Sanrio only',
        'Aggretsuko Sanrio sticker icon single character',
        'Retsuko Aggretsuko Sanrio close up face merchandise',
    ]),

    ('kirimichan', [
        'Kirimi-chan Sanrio official art white background alone',
        'Kirimichan Sanrio character solo salmon fish',
        'キリミちゃん サンリオ 公式 単体 白背景',
        'Kirimichan salmon plush figure Sanrio only white',
        'Kirimi chan Sanrio sticker icon single salmon character',
        'Kirimichan Sanrio close up fish triangle merchandise',
    ]),

    ('hangyodon', [
        'Hangyodon Sanrio official art white background alone',
        'Hangyodon Sanrio character solo blue fish',
        'ハンギョドン サンリオ 公式 単体 白背景',
        'Hangyodon fish plush figure Sanrio only white',
        'Hangyodon Sanrio sticker icon single fish character',
        'Hangyodon Sanrio close up blue fish merchandise',
    ]),
]


# ── Helpers ────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    with open(os.path.join(LOG_DIR, "scraping_12.log"), "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def extraer_urls(page):
    """Extrae URLs de imágenes del DOM (igual que versiones anteriores)."""
    return page.evaluate("""
        () => {
            const urls = new Set();
            document.querySelectorAll('img').forEach(img => {
                const attrs = [
                    'src', 'data-src', 'data-iurl',
                    'data-attr', 'data-deferred', 'data-ksrc'
                ];
                attrs.forEach(attr => {
                    let val = img.getAttribute(attr);
                    if (val && val.startsWith('http') &&
                        !val.includes('favicon') &&
                        !val.includes('logo') &&
                        !val.includes('google_web')) {
                        urls.add(val);
                    }
                });
            });
            return Array.from(urls);
        }
    """)


def imagen_valida(contenido_bytes):
    """
    Valida imagen por tamaño, resolución y aspecto.
    Devuelve True si la imagen es aceptable para el dataset.

    Filtros:
      1. Tamaño mínimo en bytes (thumbnails muy chicos)
      2. Dimensiones mínimas (ancho Y alto >= MIN_DIMENSION)
      3. Ratio de aspecto (descarta banners y tiras muy alargadas)
    """
    if len(contenido_bytes) < MIN_BYTES:
        return False
    try:
        img = Image.open(io.BytesIO(contenido_bytes))
        w, h = img.size
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            return False                     # Muy chica en alguna dimensión
        ratio = max(w, h) / min(w, h)
        if ratio > MAX_RATIO:
            return False                     # Banner muy alargado
        return True
    except Exception:
        return False                         # No se pudo abrir → corrupta


def descargar_url(url, destino, vistos, fallidos):
    """
    Descarga una URL y la guarda si pasa los filtros de calidad.
    Igual que versiones anteriores + validación de imagen_valida().
    """
    if url in fallidos:
        return False
    try:
        r = requests.get(url, timeout=15, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"
            ),
        })
        if r.status_code != 200:
            fallidos.add(url)
            return False

        contenido = r.content

        # Filtro de calidad (nuevo vs versiones anteriores)
        if not imagen_valida(contenido):
            fallidos.add(url)
            return False

        # Deduplicación por hash MD5
        h = hashlib.md5(contenido).hexdigest()
        if h in vistos:
            return False

        ct  = r.headers.get("content-type", "")
        ext = "jpg" if "jpeg" in ct else "png" if "png" in ct else "jpg"
        with open(os.path.join(destino, f"{h}.{ext}"), "wb") as f:
            f.write(contenido)
        vistos.add(h)
        return True

    except Exception:
        fallidos.add(url)
        return False


def buscar_en_google(page, query, destino, vistos, fallidos, max_imgs=80):
    """
    Busca una query en Google Images y descarga hasta max_imgs imágenes.
    Igual que versiones anteriores, con pausa aleatoria al final.
    """
    os.makedirs(destino, exist_ok=True)
    q   = urllib.parse.quote(query)
    url = f"https://www.google.com/search?q={q}&tbm=isch"

    log(f"  Buscando: {query[:70]}...")
    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
    except Exception:
        log("  Timeout en goto, continuando...")
    page.wait_for_timeout(4000)

    # Captcha handler
    for _ in range(3):
        if "sorry" in page.url.lower() or "captcha" in page.url.lower():
            log("  ⚠️  Captcha detectado. Resuélvelo y presiona Enter...")
            page.screenshot(path=os.path.join(LOG_DIR, "captcha.png"))
            input("  → Enter cuando lo resuelvas: ")
            page.wait_for_timeout(2000)
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)
        else:
            break

    if "sorry" in page.url.lower():
        log("  ❌ Captcha no resuelto, saltando query")
        return 0

    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(2000)

    nuevas           = 0
    scrolls_vacios   = 0

    for scroll in range(60):
        if nuevas >= max_imgs:
            break

        page.evaluate("window.scrollBy(0, 1200)")
        page.wait_for_timeout(1500)

        urls        = extraer_urls(page)
        nuevas_aqui = 0

        for u in urls:
            if nuevas >= max_imgs:
                break
            if descargar_url(u, destino, vistos, fallidos):
                nuevas     += 1
                nuevas_aqui += 1

        if nuevas_aqui > 0:
            scrolls_vacios = 0
            if nuevas % 40 == 0:
                log(f"    ...{nuevas}/{max_imgs} imágenes")
        else:
            scrolls_vacios += 1

        if scrolls_vacios >= 10:
            log(f"    Sin nuevas imágenes por 10 scrolls ({scroll+1} total)")
            break

    # Pausa aleatoria entre queries (2-4 seg) para no saturar a Google
    pausa = round(random.uniform(2.0, 4.0), 1)
    time.sleep(pausa)

    return nuevas


# ── Main ───────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("SCRAPING 12 CLASES SANRIO — MEJORADO")
    log("=" * 60)
    log(f"Destino: {RAW_DIR}")
    log(f"Clases:  {len(PERSONAJES)}")
    log(f"Queries: {len(PERSONAJES[0][1])} por clase")
    log(f"Meta:    ~{len(PERSONAJES[0][1]) * 80} imgs brutas por clase")
    log(f"Filtros: min {MIN_BYTES//1000}KB, min {MIN_DIMENSION}px, ratio<{MAX_RATIO}")
    log("")

    # Verificar que Brave esté cerrado
    import subprocess
    r = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq brave.exe"],
        capture_output=True, text=True
    )
    if "brave.exe" in r.stdout:
        log("⚠️  Brave está abierto. Ciérralo antes de continuar.")
        input("Presiona Enter después de cerrar Brave...")

    total_global = 0

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=BRAVE_PROFILE,
            executable_path=BRAVE_PATH,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            no_viewport=True,
            locale="en-US",
        )
        page = context.pages[0] if context.pages else context.new_page()

        log("Brave abierto. Si hay captcha en la primera búsqueda, resuélvelo.")
        log("El resto avanza solo.\n")

        for nombre, queries in PERSONAJES:
            destino = os.path.join(RAW_DIR, nombre)
            os.makedirs(destino, exist_ok=True)

            # Cargar hashes existentes para no duplicar si se re-ejecuta
            vistos   = set()
            fallidos = set()
            for f in os.listdir(destino):
                vistos.add(os.path.splitext(f)[0])

            antes = len(os.listdir(destino))
            log(f">>> {nombre.upper()} ({antes} existentes)")

            total_personaje = 0
            for q in queries:
                n = buscar_en_google(
                    page, q, destino, vistos, fallidos, max_imgs=80
                )
                total_personaje += n

            despues = len(os.listdir(destino))
            log(f">>> {nombre}: {despues} imágenes ({total_personaje} nuevas)\n")
            total_global += total_personaje

            # Pausa entre personajes (5-8 seg)
            pausa = round(random.uniform(5.0, 8.0), 1)
            log(f"  Pausa {pausa}s antes del siguiente personaje...")
            time.sleep(pausa)

        context.close()

    log("=" * 60)
    log("COMPLETADO")
    log(f"Total imágenes descargadas: {total_global}")
    log("")
    log("Resumen por clase:")
    for nombre, _ in PERSONAJES:
        ruta = os.path.join(RAW_DIR, nombre)
        if os.path.exists(ruta):
            n = len(os.listdir(ruta))
            barra = '#' * max(1, n // 10)
            log(f"  {nombre:20s}: {n:4d}  {barra}")
    log("")
    log("Siguiente paso:")
    log("  python scraping/preprocesar_raw.py  ← NO olvides actualizar BASE")
    log("  python scraping/split_12clases.py")
    log("=" * 60)


if __name__ == "__main__":
    main()
