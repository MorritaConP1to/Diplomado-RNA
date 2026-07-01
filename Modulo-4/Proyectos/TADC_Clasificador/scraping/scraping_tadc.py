#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scraping TADC desde Google Images usando el perfil real de Brave.
Con tu sesion de Google, no hay captcha.

INSTRUCCIONES:
  1. CIERRA Brave completamente (importante!)
  2. Ejecuta: python scraping/scraping_tadc.py
  3. Se abrira Brave con tu perfil real
  4. Si aparece captcha en la primera busqueda, resuelvelo
  5. El resto avanza solo
"""
import os, time, hashlib, requests, shutil, urllib.parse
from datetime import datetime
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'dataset', 'raw')
LOG_DIR = os.path.join(BASE_DIR, 'scraping', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

BRAVE_PATH = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
BRAVE_PROFILE = os.path.expandvars(
    r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\User Data"
)

PERSONAJES = [
    ('pomni', ['The Amazing Digital Circus Pomni figure',
               'The Amazing Digital Circus Pomni plush',
               'Pomni peluche',
               'Pomni action figure toy']),
    ('jax', ['The Amazing Digital Circus Jax figure',
             'The Amazing Digital Circus Jax plush',
             'Jax peluche conejo',
             'Jax action figure']),
    ('ragatha', ['The Amazing Digital Circus Ragatha figure',
                 'The Amazing Digital Circus Ragatha plush',
                 'Ragatha doll',
                 'Ragatha action figure']),
    ('caine', ['The Amazing Digital Circus Caine figure',
               'The Amazing Digital Circus Caine plush',
               'Caine ringmaster',
               'Caine action figure']),
    ('zooble', ['The Amazing Digital Circus Zooble figure',
                'The Amazing Digital Circus Zooble plush',
                'Zooble peluche',
                'Zooble action figure']),
    ('gangle', ['The Amazing Digital Circus Gangle figure',
                'The Amazing Digital Circus Gangle plush',
                'Gangle mask',
                'Gangle action figure']),
    ('kinger', ['The Amazing Digital Circus Kinger figure',
                'The Amazing Digital Circus Kinger plush',
                'Kinger chess',
                'Kinger action figure']),
    ('kaufmo', ['The Amazing Digital Circus Kaufmo figure',
                'The Amazing Digital Circus Kaufmo plush',
                'Kaufmo clown',
                'Kaufmo action figure']),
    ('bubble', ['The Amazing Digital Circus Bubble figure',
                'The Amazing Digital Circus Bubble plush',
                'Bubble peluche',
                'Bubble action figure']),
    ('gummigoo', ['The Amazing Digital Circus Gummigoo figure',
                  'The Amazing Digital Circus Gummigoo plush',
                  'Gummigoo alligator toy',
                  'Gummigoo action figure']),
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    with open(os.path.join(LOG_DIR, "scraping.log"), "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def extraer_urls(page):
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


def descargar_url(url, destino, vistos, fallidos):
    if url in fallidos:
        return False
    try:
        r = requests.get(url, timeout=15, headers={
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"),
        })
        if r.status_code != 200 or len(r.content) < 3000:
            fallidos.add(url)
            return False
        h = hashlib.md5(r.content).hexdigest()
        if h in vistos:
            return False
        ct = r.headers.get("content-type", "")
        ext = "jpg" if "jpeg" in ct else "png" if "png" in ct else "jpg"
        with open(os.path.join(destino, f"{h}.{ext}"), "wb") as f:
            f.write(r.content)
        vistos.add(h)
        return True
    except:
        fallidos.add(url)
        return False


def buscar_en_google(page, query, destino, vistos, fallidos, max_imgs=80):
    os.makedirs(destino, exist_ok=True)
    q = urllib.parse.quote(query)
    url = f"https://www.google.com/search?q={q}&tbm=isch"

    log(f"  Google: {query[:60]}...")
    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
    except:
        log("  Timeout, continuando...")
    page.wait_for_timeout(4000)

    # Captura captcha si aparece
    for _ in range(3):
        if "sorry" in page.url.lower() or "captcha" in page.url.lower():
            log("  ⚠️ Captcha! Resuelvelo y presiona Enter...")
            page.screenshot(path=os.path.join(LOG_DIR, "captcha.png"))
            input("  → Presiona Enter cuando resuelvas: ")
            page.wait_for_timeout(2000)
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)
        else:
            break

    if "sorry" in page.url.lower():
        log("  ❌ Captcha no resuelto, saltando")
        return 0

    # Esperar a que cargue la red
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except:
        pass
    page.wait_for_timeout(2000)

    nuevas = 0
    sin_resultados = 0
    scrolls_sin_imagen = 0

    for scroll in range(50):
        if nuevas >= max_imgs:
            break

        page.evaluate("window.scrollBy(0, 1200)")
        page.wait_for_timeout(1500)

        urls = extraer_urls(page)
        log(f"    scroll {scroll+1}: {len(urls)} URLs encontradas")
        nuevas_aqui = 0

        for u in urls:
            if nuevas >= max_imgs:
                break
            if descargar_url(u, destino, vistos, fallidos):
                nuevas += 1
                nuevas_aqui += 1

        if nuevas_aqui > 0:
            sin_resultados = 0
            scrolls_sin_imagen = 0
            if nuevas % 20 == 0:
                log(f"    ...{nuevas}/{max_imgs}")
        else:
            sin_resultados += 1
            scrolls_sin_imagen += 1

        if scrolls_sin_imagen >= 3 and len(urls) > 0:
            log(f"    URLs encontradas pero sin nuevas descargas, bajando calidad...")
            continue

        if sin_resultados >= 10:
            log(f"    Sin mas resultados ({scroll+1} scrolls)")
            break

    if nuevas == 0:
        fname = "".join(c for c in query[:30] if c.isalnum() or c in ' _-')[:25]
        page.screenshot(path=os.path.join(LOG_DIR, f"debug_{fname}.png"))
        debug_info = page.evaluate("""
            () => ({
                img_count: document.querySelectorAll('img').length,
                title: document.title,
                url: window.location.href.substring(0, 100),
                has_results: document.title.includes('resultados') || document.title.includes('Images'),
            })
        """)
        log(f"    DEBUG: {debug_info}")

    return nuevas


def main():
    log("=" * 60)
    log("SCRAPING TADC — GOOGLE IMAGES + BRAVE")
    log("=" * 60)
    log(f"Brave: {BRAVE_PATH}")
    log(f"Perfil: {BRAVE_PROFILE}")

    # Checar si Brave ya esta abierto
    import subprocess
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq brave.exe"],
                      capture_output=True, text=True)
    if "brave.exe" in r.stdout:
        log("⚠️  Brave esta abierto. CIERRALO antes de continuar.")
        input("Presiona Enter despues de cerrar Brave...")

    # Preguntar por raw/
    if os.path.exists(RAW_DIR):
        r = input("Borrar dataset/raw/ y empezar de cero? (s/N): ").strip().lower()
        if r == "s":
            shutil.rmtree(RAW_DIR)
            log("raw/ borrado!")

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
        page = context.pages[0]
        if not page:
            page = context.new_page()

        log("\nBrave abierto con tu perfil. Google confia en tu sesion.")
        log("Si la primera busqueda pide captcha, resuelvelo una vez.\n")

        for nombre, queries in PERSONAJES:
            destino = os.path.join(RAW_DIR, nombre)
            os.makedirs(destino, exist_ok=True)

            antes = len(os.listdir(destino))
            vistos = set()
            for f in os.listdir(destino):
                vistos.add(os.path.splitext(f)[0])
            fallidos = set()

            log(f">>> {nombre.upper()}")
            total_personaje = 0
            for i, q in enumerate(queries):
                n = buscar_en_google(page, q, destino, vistos, fallidos, max_imgs=60)
                total_personaje += n
                time.sleep(1.5)

            despues = len(os.listdir(destino))
            log(f">>> {nombre}: {despues} imagenes ({total_personaje} nuevas)\n")

        context.close()

    log("=" * 60)
    log("COMPLETADO!")
    for nombre, _ in PERSONAJES:
        ruta = os.path.join(RAW_DIR, nombre)
        if os.path.exists(ruta):
            total = len(os.listdir(ruta))
            log(f"  {nombre}: {total}")
    log("=" * 60)


if __name__ == "__main__":
    main()
