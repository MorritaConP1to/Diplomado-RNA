# TADC Clasificador — The Amazing Digital Circus

Clasificador multiclase de personajes de The Amazing Digital Circus usando:
- **CNN desde cero**
- **Transfer Learning con ResNet18**

## Flujo de trabajo

### 1. Scraping — Descargar imagenes

```cmd
conda activate diplomado-redes
cd D:\Diplomado-RNA\Modulo-4\Proyectos\TADC_Clasificador
python scraping/scraping_tadc.py
```

### 2. Preprocesar y dividir

```cmd
python scraping/preprocesar_raw.py
python scraping/split_multiclase.py
```

### 3. Abrir notebooks en VSCode

- `CNN_Multiclase.ipynb`
- `Transfer_Learning_Multiclase.ipynb`

## Personajes

pomni, jax, ragatha, caine, zooble, gangle, kinger, kaufmo, bubble, gummigoo

## Entregables

1. `CNN_Multiclase.ipynb`
2. `Transfer_Learning_Multiclase.ipynb`
