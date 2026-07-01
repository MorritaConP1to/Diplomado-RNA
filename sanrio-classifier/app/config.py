"""
Configuración centralizada del Clasificador Sanrio.

Todas las rutas, thresholds y límites en un solo lugar.
Fácil de ajustar sin modificar la lógica de negocio.

Variables clave:
  CONF_THRESHOLD: 0.30 → si la máxima confianza es menor, se rechaza
  IMG_MIN/MAX_SIZE: 32-4000px → ventana de tamaños aceptables
  ALLOWED_EXTENSIONS/MIMES: solo imágenes JPG, PNG, WebP
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

CLASES_JSON = os.path.join(MODELS_DIR, 'clases_sanrio.json')
MODELO_ONNX = os.path.join(MODELS_DIR, 'tl_sanrio_int8.onnx')

IMG_SIZE = 224
CONF_THRESHOLD = 0.30
IMG_MIN_SIZE = 32
IMG_MAX_SIZE = 4000
MIN_FILE_SIZE = 5120
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_MIMES = {'image/jpeg', 'image/png', 'image/webp'}

RATE_LIMIT = "100/minute"
