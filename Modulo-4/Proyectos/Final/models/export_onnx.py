#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exporta modelo ResNet18 Transfer Learning (.pth) a ONNX con cuantización dinámica.
El ONNX corre con onnxruntime (CPU) y pesa ~12MB vs 43MB del .pth.

Uso:
    python models/export_onnx.py

Requiere: torch, torchvision, onnx, onnxruntime
"""
import os, sys, json, torch
import torch.nn as nn
from torchvision import models

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

SRC = os.path.join(BASE, 'modelos')
PESOS_PATH = os.path.join(SRC, 'tl_sanrio_final.pth')
CLASES_PATH = os.path.join(SRC, 'clases_sanrio.json')
ONNX_PATH = os.path.join(BASE, 'models', 'tl_sanrio.onnx')
ONNX_INT8_PATH = os.path.join(BASE, 'models', 'tl_sanrio_int8.onnx')


def get_num_classes():
    with open(CLASES_PATH, 'r') as f:
        data = json.load(f)
    return data['num_classes'], data['classes']


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def export():
    print(f"Cargando pesos: {PESOS_PATH}")
    num_clases, clases = get_num_classes()
    print(f"Numero de clases: {num_clases}")

    model = build_model(num_clases)
    state = torch.load(PESOS_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    print(f"Exportando a ONNX FP32: {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=['input'],
        output_names=['output'],
        opset_version=18,
        dynamo=False,
    )
    print(f"ONNX FP32 exportado: {ONNX_PATH}")

    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX FP32 verificado")

    # Cuantización dinámica a INT8
    print(f"\nExportando a ONNX INT8: {ONNX_INT8_PATH}")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(ONNX_PATH, ONNX_INT8_PATH, weight_type=QuantType.QInt8)
        print(f"ONNX INT8 exportado: {ONNX_INT8_PATH}")
    except Exception as e:
        print(f"Nota: Cuantización INT8 no disponible: {e}")
        print("Usando ONNX FP32 como respaldo...")
        import shutil
        shutil.copy(ONNX_PATH, ONNX_INT8_PATH)

    size_pth = os.path.getsize(PESOS_PATH) / 1_000_000
    size_fp32 = os.path.getsize(ONNX_PATH) / 1_000_000
    size_int8 = os.path.getsize(ONNX_INT8_PATH) / 1_000_000
    print(f"\nTamaños:")
    print(f"  .pth original:      {size_pth:.1f} MB")
    print(f"  ONNX FP32:          {size_fp32:.1f} MB")
    print(f"  ONNX INT8:          {size_int8:.1f} MB")

    # Test inference con INT8
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_INT8_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy.numpy().astype(np.float32)})
    probs = torch.nn.functional.softmax(torch.tensor(output[0]), dim=1).numpy()
    top1 = int(probs[0].argmax())
    print(f"\nInferencia de prueba: {clases[top1]} ({probs[0][top1]*100:.1f}%)")
    print(f"Exportacion completada!")


if __name__ == '__main__':
    export()
