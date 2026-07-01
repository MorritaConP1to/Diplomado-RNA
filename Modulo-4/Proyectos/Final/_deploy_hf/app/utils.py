"""
Utilidades de validación y preprocesamiento de imágenes.

Flujo de defensa:
  1. validar_archivo: extensión, tipo MIME, tamaño mínimo
  2. validar_imagen: integridad PIL, dimensiones 32-4000px
  3. preprocess_image: resize 224x224 + normalización ImageNet

Cada función retorna (bool, mensaje) para manejo de errores consistente.
"""

import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from app.config import ALLOWED_EXTENSIONS, ALLOWED_MIMES, IMG_MIN_SIZE, IMG_MAX_SIZE, MIN_FILE_SIZE

def validar_archivo(nombre: str, content_type: str, tamaño: int = -1) -> tuple[bool, str]:
    ext = os.path.splitext(nombre)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Formato no soportado: {ext}. Usa: {', '.join(ALLOWED_EXTENSIONS)}"
    if content_type and content_type not in ALLOWED_MIMES:
        return False, f"Tipo MIME no válido: {content_type}"
    if tamaño >= 0 and tamaño < MIN_FILE_SIZE:
        return False, "La imagen es demasiado pequeña o está vacía"
    return True, ""

def validar_imagen(ruta: str) -> tuple[bool, str]:
    try:
        with Image.open(ruta) as img:
            img.verify()
        with Image.open(ruta) as img:
            img.load()
            w, h = img.size
    except (UnidentifiedImageError, Exception):
        return False, "La imagen está corrupta o no es válida"
    if w < IMG_MIN_SIZE or h < IMG_MIN_SIZE:
        return False, f"La imagen es muy pequeña ({w}x{h}). Mínimo: {IMG_MIN_SIZE}px"
    if w > IMG_MAX_SIZE or h > IMG_MAX_SIZE:
        return False, f"La imagen es muy grande ({w}x{h}). Máximo: {IMG_MAX_SIZE}px"
    return True, ""

def preprocess_image(ruta: str) -> np.ndarray:
    """
    Preprocesa imagen para ONNX inference sin torch/torchvision.
    Resize 224x224 + normalizacion ImageNet usando PIL + numpy.
    """
    from PIL import Image
    img = Image.open(ruta).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr
