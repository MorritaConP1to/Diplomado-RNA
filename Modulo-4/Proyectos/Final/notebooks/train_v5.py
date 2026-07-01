"""
Transfer_Learning_Sanrio_v5 — Standalone Training Script
ResNet18 + 2-layer head + Focal Loss + ONNX INT8 via onnxruntime

Optimizaciones respecto a v4 (88.54%):
- Focal Loss gamma=2.0 (enfoca en hard examples)
- Fase 3: 30 epocas (vs 20) para mejor fine-tuning
- Dropout 0.5 (vs 0.3) mas regularizacion
- Sin MixUp (empeoraba en v5 experimental)

Run: python train_v5.py
"""
import os, sys, time, json, warnings, gc, copy, math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from functools import partial
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

warnings.filterwarnings('ignore')

AUTORA  = 'Diana Blanco — MorritaConP1to'
VERSION = 'v5-ResNet18-12clases-FocalLoss'
SEMILLA = 42
INICIO  = time.time()
torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)

DATASET_VERSION = 'v4'

# Path detection
cwd = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(cwd) == 'notebooks':
    RUTA_BASE = os.path.dirname(cwd)
elif os.path.isdir(os.path.join(cwd, 'dataset')):
    RUTA_BASE = cwd
else:
    RUTA_BASE = cwd

RUTA_DATASET = os.path.join(RUTA_BASE, 'dataset')
RUTA_MODELOS = os.path.join(RUTA_BASE, 'modelos')
RUTA_MODELS  = os.path.join(RUTA_BASE, 'models')
TRAIN_DIR    = os.path.join(RUTA_DATASET, f'train_{DATASET_VERSION}')
TEST_DIR     = os.path.join(RUTA_DATASET, f'test_{DATASET_VERSION}')
os.makedirs(RUTA_MODELOS, exist_ok=True)
os.makedirs(RUTA_MODELS,  exist_ok=True)

CONFIG = {
    'BATCH_SIZE':      16,
    'EPOCHS_FASE1':    10,
    'EPOCHS_FASE2':    15,
    'EPOCHS_FASE3':    30,          # Aumentado: 20 -> 30
    'LR_FASE1':        0.01,
    'LR_FASE2':        0.001,
    'LR_FASE3':        0.0005,
    'DROPOUT':         0.5,          # Aumentado: 0.3 -> 0.5
    'PATIENCE':        5,
    'MOMENTUM':        0.9,
    'WEIGHT_DECAY':    1e-4,
    'GRAD_CLIP':       1.0,
    'LABEL_SMOOTHING': 0.1,
    'FOCAL_GAMMA':     2.0,          # Nuevo: Focal Loss gamma
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f'No se encuentra: {TRAIN_DIR}')

print(f'Device:          {device}')
if device.type == 'cuda':
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
print(f'Dataset version: {DATASET_VERSION}')
print(f'Train dir:       {TRAIN_DIR}')
print(f'Test dir:        {TEST_DIR}')
print(f'CONFIG:          {json.dumps(CONFIG, indent=2)}')

# ── Dataset ──
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.10, scale=(0.02, 0.15)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform_train)
test_dataset  = ImageFolder(root=TEST_DIR,  transform=transform_test)
clases        = train_dataset.classes
NUM_CLASES    = len(clases)

print(f'Clases detectadas ({NUM_CLASES}): {clases}')

ruta_clases = os.path.join(RUTA_MODELOS, f'clases_sanrio_{DATASET_VERSION}.json')
with open(ruta_clases, 'w', encoding='utf-8') as f:
    json.dump(clases, f, ensure_ascii=False, indent=2)
print(f'Clases guardadas: {ruta_clases}')

conteos = [len([f for f in os.listdir(os.path.join(TRAIN_DIR, c)) if os.path.isfile(os.path.join(TRAIN_DIR, c, f))]) for c in clases]
total_imgs = sum(conteos)
pesos_clase = torch.tensor([total_imgs / max(n, 1) for n in conteos], dtype=torch.float).to(device)

print('\nDistribucion train:')
for c, n in sorted(zip(clases, conteos), key=lambda x: x[1]):
    barra = '#' * max(1, n // 8)
    warning = '  MENOS IMAGENES' if n < 200 else ''
    print(f'  {c:20s}: {n:4d} img  {barra}{warning}')
print(f'\n  Total: {total_imgs}')

sampler = WeightedRandomSampler(
    weights=[pesos_clase[train_dataset.targets[i]].item() for i in range(len(train_dataset))],
    num_samples=len(train_dataset), replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'],
                          sampler=sampler, num_workers=0,
                          pin_memory=(device.type == 'cuda'))
test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'],
                         shuffle=False, num_workers=0,
                         pin_memory=(device.type == 'cuda'))
print(f'Train batches: {len(train_loader)} | Test batches: {len(test_loader)}')

# ── Focal Loss ──
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight,
                                  label_smoothing=self.label_smoothing,
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ── Model ──
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
for param in model.parameters():
    param.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=CONFIG['DROPOUT']),
    nn.Linear(in_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(256, NUM_CLASES),
)
model = model.to(device)

criterion = FocalLoss(gamma=CONFIG['FOCAL_GAMMA'],
                      weight=pesos_clase,
                      label_smoothing=CONFIG['LABEL_SMOOTHING'])

total_p = sum(p.numel() for p in model.parameters())
entrena_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nParametros totales:     {total_p:,}')
print(f'Parametros entrenables: {entrena_p:,} ({100*entrena_p/total_p:.1f}%)')
print(f'Clases de salida:       {NUM_CLASES}')
print(f'Loss function:          FocalLoss(gamma={CONFIG["FOCAL_GAMMA"]})')

# ── Training function ──
def entrenar(modelo, loader, test_loader, criterion,
             optimizer, scheduler, num_epochs, device, fase):
    historial = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    mejor_loss = float('inf')
    mejor_state = None
    sin_mejora = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        modelo.train()
        tl = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(modelo(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), CONFIG['GRAD_CLIP'])
            optimizer.step()
            tl += loss.item()
        tl /= len(loader)

        modelo.eval()
        vl = 0.0
        ok = 0
        tot = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = modelo(imgs)
                vl += criterion(out, labels).item()
                _, p = torch.max(out, 1)
                tot += labels.size(0)
                ok += (p == labels).sum().item()
        vl /= len(test_loader)
        acc = 100.0 * ok / tot

        historial['train_loss'].append(tl)
        historial['test_loss'].append(vl)
        historial['test_acc'].append(acc)

        print(f'{fase} [{epoch+1:2d}/{num_epochs}]  '
              f'T:{tl:.4f}  V:{vl:.4f}  Acc:{acc:.2f}%  '
              f'LR:{optimizer.param_groups[0]["lr"]:.6f}  {time.time()-t0:.0f}s')

        if scheduler:
            scheduler.step()

        if vl < mejor_loss:
            mejor_loss = vl
            mejor_state = copy.deepcopy(modelo.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= CONFIG['PATIENCE']:
                print(f'  Early stopping en epoca {epoch+1}')
                break

    modelo.load_state_dict(mejor_state)
    print(f'  Mejor estado restaurado (val_loss={mejor_loss:.4f})')
    return historial

# ── PHASE 1 ──
print('\n' + '='*65)
print('FASE 1 — SOLO CABEZA (backbone congelado)')
print('='*65)

opt1 = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG['LR_FASE1'], momentum=CONFIG['MOMENTUM'],
    weight_decay=CONFIG['WEIGHT_DECAY'], nesterov=True
)
sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=CONFIG['EPOCHS_FASE1'])
hist1 = entrenar(model, train_loader, test_loader, criterion,
                 opt1, sch1, CONFIG['EPOCHS_FASE1'], device, 'Fase1')

# ── PHASE 2 ──
print('\n' + '='*65)
print('FASE 2 — CAPA 4 + CABEZA (layer4 descongelado)')
print('='*65)

for param in model.layer4.parameters():
    param.requires_grad = True

opt2 = optim.SGD([
    {'params': model.layer4.parameters(), 'lr': CONFIG['LR_FASE2']},
    {'params': model.fc.parameters(), 'lr': CONFIG['LR_FASE2']},
], momentum=CONFIG['MOMENTUM'], weight_decay=CONFIG['WEIGHT_DECAY'], nesterov=True)
sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CONFIG['EPOCHS_FASE2'])

entrena_f2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parametros entrenables: {entrena_f2:,} ({100*entrena_f2/total_p:.1f}%)\n')

hist2 = entrenar(model, train_loader, test_loader, criterion,
                 opt2, sch2, CONFIG['EPOCHS_FASE2'], device, 'Fase2')

# ── PHASE 3 ──
print('\n' + '='*65)
print('FASE 3 — CAPAS 3 Y 4 + CABEZA (layer3+layer4 descongelados)')
print(f'         {CONFIG["EPOCHS_FASE3"]} epocas, CosineAnnealingLR(T_max={CONFIG["EPOCHS_FASE3"]})')
print('='*65)

for param in model.layer3.parameters():
    param.requires_grad = True

opt3 = optim.SGD([
    {'params': model.layer3.parameters(), 'lr': CONFIG['LR_FASE3'] * 0.5},
    {'params': model.layer4.parameters(), 'lr': CONFIG['LR_FASE3']},
    {'params': model.fc.parameters(),      'lr': CONFIG['LR_FASE3']},
], momentum=CONFIG['MOMENTUM'], weight_decay=CONFIG['WEIGHT_DECAY'], nesterov=True)
sch3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=CONFIG['EPOCHS_FASE3'])

entrena_f3 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parametros entrenables: {entrena_f3:,} ({100*entrena_f3/total_p:.1f}%)\n')

hist3 = entrenar(model, train_loader, test_loader, criterion,
                 opt3, sch3, CONFIG['EPOCHS_FASE3'], device, 'Fase3')

historial = {
    'train_loss': hist1['train_loss'] + hist2['train_loss'] + hist3['train_loss'],
    'test_loss':  hist1['test_loss']  + hist2['test_loss']  + hist3['test_loss'],
    'test_acc':   hist1['test_acc']   + hist2['test_acc']   + hist3['test_acc'],
}
mejor_acc = max(historial['test_acc'])
print(f'\nMejor accuracy: {mejor_acc:.2f}%')

# ── Save model ──
ruta_pth = os.path.join(RUTA_MODELOS, f'tl_sanrio_v5_final.pth')
torch.save(model.state_dict(), ruta_pth)
print(f'Pesos guardados: {ruta_pth}')

# ── ONNX Export ──
print('\n--- Exportando ONNX ---')
try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import shutil

    ruta_fp32 = os.path.join(RUTA_MODELS, 'tl_sanrio_v5_fp32.onnx')
    ruta_int8 = os.path.join(RUTA_MODELS, 'tl_sanrio_v5_int8.onnx')
    ruta_int8_final = os.path.join(RUTA_MODELS, 'tl_sanrio_int8.onnx')

    model.eval()
    model.cpu()
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(model, dummy, ruta_fp32,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=17, export_params=True)
    mb_fp32 = os.path.getsize(ruta_fp32) / 1e6
    print(f'ONNX FP32: {mb_fp32:.1f} MB -> {ruta_fp32}')

    quantize_dynamic(ruta_fp32, ruta_int8, weight_type=QuantType.QUInt8)
    mb_int8 = os.path.getsize(ruta_int8) / 1e6
    print(f'ONNX INT8: {mb_int8:.1f} MB -> {ruta_int8}')
    print(f'Reduccion: {mb_fp32/mb_int8:.1f}x vs FP32')

    shutil.copy2(ruta_int8, ruta_int8_final)
    print(f'Copiado a: {ruta_int8_final}')

    src_cls = os.path.join(RUTA_MODELOS, f'clases_sanrio_{DATASET_VERSION}.json')
    dst_cls = os.path.join(RUTA_MODELS, 'clases_sanrio.json')
    shutil.copy2(src_cls, dst_cls)
    print(f'clases copiadas a: {dst_cls}')

    session = ort.InferenceSession(ruta_int8)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: np.zeros((1, 3, 224, 224), dtype=np.float32)})
    print(f'Validacion INT8 exitosa: output shape {result[0].shape}')

    model.to(device)

except Exception as e:
    print(f'Error ONNX: {e}')
    import traceback
    traceback.print_exc()

# ── Experiment log (preserving v4) ──
ruta_exp = os.path.join(RUTA_MODELOS, 'experimentos_tl.json')
if os.path.exists(ruta_exp):
    with open(ruta_exp, 'r') as f:
        experimentos = json.load(f)
else:
    experimentos = []

nuevo_exp = {
    'version': VERSION,
    'dataset': DATASET_VERSION,
    'fecha': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'arquitectura': 'ResNet18',
    'num_clases': NUM_CLASES,
    'clases': clases,
    'mejor_accuracy': round(mejor_acc, 2),
    'accuracy_final': round(historial['test_acc'][-1], 2),
    'config': CONFIG,
    'dataset_stats': {'train': total_imgs, 'test': len(test_dataset)},
}

# Solo eliminar duplicados de la MISMA version, no de v4
experimentos = [e for e in experimentos if e.get('version') != VERSION]
experimentos.append(nuevo_exp)

with open(ruta_exp, 'w', encoding='utf-8') as f:
    json.dump(experimentos, f, ensure_ascii=False, indent=2)
print('Experimento registrado en experimentos_tl.json')

# ── Summary ──
fin = time.time()
hh, rem = divmod(int(fin - INICIO), 3600)
mm, ss = divmod(rem, 60)

print('\n' + '='*62)
print(f'  Hecho con carino — {AUTORA}')
print(f'  Diplomado RNA * Modulo 4 * UAEM * Julio 2026')
print()
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  PyTorch: {torch.__version__}')
print()
print(f'  Version:          {VERSION}')
print(f'  Dataset:          {DATASET_VERSION} ({NUM_CLASES} clases)')
print(f'  Tiempo total:     {hh}h {mm:02d}m {ss:02d}s')
print(f'  Mejor accuracy:   {mejor_acc:.2f}%')
print(f'  Baseline v4:      88.54%')
print(f'  Mejora vs v4:     {mejor_acc - 88.54:+.2f}%')
print(f'  Fecha:            {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print()
print('  Para deploy:')
print('     models/tl_sanrio_int8.onnx')
print('     models/clases_sanrio.json')
print()
print('  Optimizaciones v5:')
print('     + FocalLoss(gamma=2.0) en vez de CrossEntropy simple')
print('     + Fase 3: 30 epocas (vs 20 en v4)')
print('     + Dropout 0.5 (vs 0.3 en v4)')
print('     - Sin MixUp (empeoraba accuracy)')
print('='*62)
