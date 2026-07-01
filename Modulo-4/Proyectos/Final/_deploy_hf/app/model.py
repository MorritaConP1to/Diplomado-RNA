"""
Clasificador ONNX para personajes Sanrio.

SanrioClassifier:
  - Carga modelo ONNX INT8 (11.4 MB) en CPU con onnxruntime
  - Preprocesa imagen: resize a 224x224, normalización ImageNet
  - Softmax manual para obtener probabilidades calibradas
  - Top-3 con nombres human-readable desde NOMBRES_MOSTRAR
  - Threshold de confianza: <30% → mensaje "no parece Sanrio"
  - 30 clases mapeadas desde clases_sanrio.json

Uso:
  from app.model import clasificador
  clasificador.cargar()
  resultado = clasificador.predecir(tensor, threshold=0.30)
"""

import json
import os
import numpy as np
import onnxruntime as ort
from app.config import MODELS_DIR, CLASES_JSON, MODELO_ONNX

# Nombres para mostrar (human-readable)
# Misma clave que en clases_sanrio.json para mapping consistente
NOMBRES_MOSTRAR = {
    "badtz_maru": "Badtz-Maru",
    "chococat": "Chococat",
    "cinnamon": "Cinnamoroll",
    "hello_kitty": "Hello Kitty",
    "keroppi": "Keroppi",
    "kuromi": "Kuromi",
    "my_melody": "My Melody",
    "pochacco": "Pochacco",
    "pompompurin": "Pompompurin",
    "tuxedo_sam": "Tuxedo Sam",
    "pekkle": "Pekkle",
    "hangyodon": "Hangyodon",
    "little_twin_stars": "Little Twin Stars",
    "cogimyun": "Cogimyun",
    "my_sweet_piano": "My Sweet Piano",
    "hanamaruobake": "Hanamaruobake",
    "wish_me_mell": "Wish Me Mell",
    "usahana": "Usahana",
    "gaopowerroo": "Gaopowerroo",
    "kuririn": "Corocorokuririn",
    "gudetama": "Gudetama",
    "aggretsuko": "Aggretsuko",
    "kirimichan": "Kirimichan",
    "marroncream": "Marroncream",
    "marumofubiyori": "Marumofubiyori",
    "charmmykitty": "Charmmy Kitty",
    "dear_daniel": "Dear Daniel",
    "sugarbunnies": "Sugarbunnies",
    "yoshikitty": "Yoshikitty",
    "hello_mimmy": "Hello Mimmy",
}


class SanrioClassifier:
    def __init__(self):
        self.session = None
        self.clases = []
        self.input_name = None
        self.cargado = False

    def cargar(self):
        if not os.path.exists(MODELO_ONNX):
            return False
        if not os.path.exists(CLASES_JSON):
            return False

        with open(CLASES_JSON, 'r') as f:
            data = json.load(f)
        self.clases = data if isinstance(data, list) else data['classes']

        self.session = ort.InferenceSession(
            MODELO_ONNX,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.cargado = True
        return True

    def predecir(self, input_tensor: np.ndarray, threshold: float = 0.30):
        if not self.cargado:
            raise RuntimeError("Modelo no cargado")

        outputs = self.session.run(None, {self.input_name: input_tensor.astype(np.float32)})[0]
        exp = np.exp(outputs - outputs.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs = probs[0]

        idxs = np.argsort(probs)[::-1]
        top3 = []
        for i in idxs[:3]:
            clase = self.clases[i]
            top3.append({
                'clase': clase,
                'nombre_mostrar': NOMBRES_MOSTRAR.get(clase, clase),
                'confianza': round(float(probs[i] * 100), 2),
            })

        mejor = top3[0]
        if mejor['confianza'] < threshold * 100:
            return {
                'exito': True,
                'prediccion': None,
                'confianza': None,
                'top_3': top3,
                'mensaje': "No parece un personaje Sanrio conocido, o no estoy segura.",
            }

        return {
            'exito': True,
            'prediccion': mejor['clase'],
            'nombre_mostrar': mejor['nombre_mostrar'],
            'confianza': mejor['confianza'],
            'top_3': top3,
            'mensaje': None,
        }


clasificador = SanrioClassifier()
