"""
curar_dataset.py — Diana Blanco (MorritaConP1to)
=================================================
Prepara el dataset para reentrenamiento mejorado:

  1. Quita 3 clases con confusion estructural irresoluble
  2. Hace cap de imagenes por clase (MAX_POR_CLASE)
  3. Re-hace el split 80/20 estratificado
  4. Genera reporte de lo que quedo

Uso:
    python curar_dataset.py

Requiere que existan:
    dataset/train/  (29 clases originales)
    dataset/test/   (29 clases originales)

Genera:
    dataset/train_v2/   (26 clases, balanceadas)
    dataset/test_v2/    (26 clases, balanceadas)
    dataset/curado_reporte.txt
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── Configuracion ──────────────────────────────────────────────
random.seed(42)

# Detectar RUTA_BASE igual que en el notebook
cwd = Path.cwd()
if cwd.name == 'notebooks':
    RUTA_BASE = cwd.parent
elif (cwd / 'dataset').is_dir():
    RUTA_BASE = cwd
else:
    RUTA_BASE = cwd

RUTA_DATASET   = RUTA_BASE / 'dataset'
RUTA_TRAIN_IN  = RUTA_DATASET / 'train'
RUTA_TEST_IN   = RUTA_DATASET / 'test'
RUTA_TRAIN_OUT = RUTA_DATASET / 'train_v2'
RUTA_TEST_OUT  = RUTA_DATASET / 'test_v2'
RUTA_REPORTE   = RUTA_DATASET / 'curado_reporte.txt'

# Clases a eliminar por confusion estructural:
#   charmmykitty  → casi identica a hello_kitty (variante con collar)
#   my_sweet_piano → casi identica a my_melody (variante con piano)
#   little_twin_stars → DOS personajes en una carpeta, modelo no sabe a quien clasificar
CLASES_ELIMINAR = {'charmmykitty', 'my_sweet_piano', 'little_twin_stars'}

# Cap de imagenes por clase en train (equilibra el dataset)
# Kuromi tenia 467, Cinnamon 474 — las bajamos al rango de las demas
MAX_POR_CLASE = 220

# Proporcion test (del total combinado train+test por clase)
TEST_RATIO = 0.20

# ── Helpers ────────────────────────────────────────────────────

def listar_imagenes(carpeta: Path) -> list[Path]:
    """Devuelve lista de imagenes validas en una carpeta."""
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    return [f for f in carpeta.iterdir()
            if f.is_file() and f.suffix.lower() in exts]

def copiar(src: Path, dst: Path):
    """Copia un archivo creando el directorio destino si no existe."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# ── Main ───────────────────────────────────────────────────────

def main():
    print()
    print('=' * 60)
    print('  CURADOR DE DATASET SANRIO — Diana Blanco')
    print('=' * 60)
    print(f'  Base:      {RUTA_BASE}')
    print(f'  Train in:  {RUTA_TRAIN_IN}')
    print(f'  Test in:   {RUTA_TEST_IN}')
    print(f'  Train out: {RUTA_TRAIN_OUT}')
    print(f'  Test out:  {RUTA_TEST_OUT}')
    print()

    # Validar que existan las carpetas de entrada
    if not RUTA_TRAIN_IN.is_dir():
        raise FileNotFoundError(f'No se encuentra {RUTA_TRAIN_IN}')
    if not RUTA_TEST_IN.is_dir():
        raise FileNotFoundError(f'No se encuentra {RUTA_TEST_IN}')

    # Detectar todas las clases disponibles
    clases_originales = sorted([
        d.name for d in RUTA_TRAIN_IN.iterdir() if d.is_dir()
    ])
    clases_validas = [c for c in clases_originales
                      if c not in CLASES_ELIMINAR]

    print(f'Clases originales:   {len(clases_originales)}')
    print(f'Clases eliminadas:   {len(CLASES_ELIMINAR)} '
          f'({", ".join(sorted(CLASES_ELIMINAR))})')
    print(f'Clases resultantes:  {len(clases_validas)}')
    print(f'Cap por clase train: {MAX_POR_CLASE} imagenes')
    print()

    # Limpiar salida si ya existe
    if RUTA_TRAIN_OUT.exists():
        shutil.rmtree(RUTA_TRAIN_OUT)
        print('  (train_v2 anterior eliminado)')
    if RUTA_TEST_OUT.exists():
        shutil.rmtree(RUTA_TEST_OUT)
        print('  (test_v2 anterior eliminado)')

    reporte_lineas = []
    reporte_lineas.append('REPORTE CURADO DATASET SANRIO')
    reporte_lineas.append('=' * 50)
    reporte_lineas.append(f'Clases eliminadas: {sorted(CLASES_ELIMINAR)}')
    reporte_lineas.append(f'Cap train: {MAX_POR_CLASE} img/clase')
    reporte_lineas.append(f'Test ratio: {TEST_RATIO:.0%}')
    reporte_lineas.append('')
    reporte_lineas.append(
        f'{"Clase":<25} {"Pool":>5} {"Train":>6} {"Test":>5} {"Total":>6}'
    )
    reporte_lineas.append('-' * 50)

    total_train = 0
    total_test  = 0

    print(f'  {"Clase":<25} {"Pool":>5} {"Train":>6} {"Test":>5}')
    print(f'  {"-"*50}')

    for clase in clases_validas:
        # Juntar train + test originales para re-hacer el split limpio
        imgs_train = listar_imagenes(RUTA_TRAIN_IN / clase)
        imgs_test  = listar_imagenes(RUTA_TEST_IN / clase) \
                     if (RUTA_TEST_IN / clase).is_dir() else []

        pool = imgs_train + imgs_test
        random.shuffle(pool)

        # Aplicar cap al pool total (no solo al train)
        # Calculamos el equivalente: si MAX_POR_CLASE es el techo de train (80%),
        # el pool max es MAX_POR_CLASE / 0.8
        pool_max = int(MAX_POR_CLASE / (1 - TEST_RATIO))
        if len(pool) > pool_max:
            pool = pool[:pool_max]

        # Split estratificado
        n_test  = max(1, round(len(pool) * TEST_RATIO))
        n_train = len(pool) - n_test

        imgs_out_test  = pool[:n_test]
        imgs_out_train = pool[n_test:]

        # Copiar a destino
        for img in imgs_out_train:
            copiar(img, RUTA_TRAIN_OUT / clase / img.name)
        for img in imgs_out_test:
            copiar(img, RUTA_TEST_OUT / clase / img.name)

        total_train += n_train
        total_test  += n_test

        print(f'  {clase:<25} {len(pool):>5} {n_train:>6} {n_test:>5}')
        reporte_lineas.append(
            f'{clase:<25} {len(pool):>5} {n_train:>6} {n_test:>5} {len(pool):>6}'
        )

    print(f'  {"-"*50}')
    print(f'  {"TOTAL":<25} {"":>5} {total_train:>6} {total_test:>5}')
    print()

    reporte_lineas.append('-' * 50)
    reporte_lineas.append(
        f'{"TOTAL":<25} {"":>5} {total_train:>6} {total_test:>5} '
        f'{total_train + total_test:>6}'
    )
    reporte_lineas.append('')
    reporte_lineas.append(f'train_v2/: {total_train} imagenes')
    reporte_lineas.append(f'test_v2/:  {total_test} imagenes')

    # Guardar reporte
    RUTA_REPORTE.write_text('\n'.join(reporte_lineas), encoding='utf-8')

    print('=' * 60)
    print(f'  ✅ Dataset curado listo')
    print(f'  Train: {total_train} imagenes en {len(clases_validas)} clases')
    print(f'  Test:  {total_test} imagenes en {len(clases_validas)} clases')
    print(f'  Reporte guardado en: {RUTA_REPORTE}')
    print('=' * 60)
    print()
    print('  Siguiente paso:')
    print('  Abre Transfer_Learning_Sanrio_v3.ipynb y ejecuta')
    print('  (ya apunta a dataset/train_v2 y dataset/test_v2)')
    print()

if __name__ == '__main__':
    main()
