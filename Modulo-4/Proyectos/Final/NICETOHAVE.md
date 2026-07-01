# NICETOHAVE.md — Mejoras Futuras

Ordenadas por impacto/esfuerzo.

## Alta prioridad, bajo esfuerzo

- [ ] **30 clases completas**: scraper para 18 personajes faltantes + fine-tune. El slot en `personajes.json` ya existe.
- [ ] **Ensemble v4 + v5**: promediar predicciones de ambos modelos para ganar ~2% extra.
- [ ] **Test con imágenes reales de cámara**: probar robustness con fotos de juguetes/merch.
- [ ] **Añadir hello_mimmy**: solo falta scrapping (difícil de encontrar imágenes).

## Media prioridad, esfuerzo medio

- [ ] **Data augmentation más agresiva**: AutoAugment/TrivialAugment en vez de manual.
- [ ] **Test-time augmentation (TTA)**: promediar predicciones de 5 crops para estabilidad.
- [ ] **Cache de inferencia**: almacenar hash de imagen → predicción para evitar re-procesar la misma imagen.
- [ ] **Gradio demo**: interfaz alternativa más pulida para compartir.
- [ ] **Gemini con memoria**: mantener historial de chat para respuestas contextuales.

## Baja prioridad, alto esfuerzo

- [ ] **EfficientNet-Lite en vez de ResNet18**: más rápido en CPU, mejor para HF Spaces.
- [ ] **Quantization-aware training**: mejor calidad INT8 que post-training quantization.
- [ ] **WebSocket para streaming de chat**: mejor UX que polling.
- [ ] **Dark/light mode toggle** en frontend.
- [ ] **PWA**: service worker para offline, instalable en mobile.

## Ideas locas

- [ ] **CLIP zero-shot**: clasificar cualquier personaje Sanrio sin reentrenar.
- [ ] **Generar imágenes Sanrio con Stable Diffusion**: "dibuja a Kuromi comiendo pizza".
- [ ] **API pública con documentación Swagger**: para que otros devs usen el clasificador.
