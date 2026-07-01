# META.md — Checklist de Rúbrica (Proyecto Final)

## A. Núcleo Clasificador (25 pts)
- [x] Modelo funcional con ResNet18 Transfer Learning
- [x] 12 clases limpias (dataset v4)
- [x] Accuracy: 87.50% (v5) / 88.54% (v4 base)
- [x] ONNX INT8 exportado y validado (11.4 MB)
- [x] Top-3 predictions con confianza
- [x] Threshold <30% → "No parece Sanrio conocido"

## B. Integración Web (20 pts)
- [x] FastAPI backend (3 endpoints: /, /health, /predict, /chat)
- [x] Frontend vanilla HTML/CSS/JS
- [x] Comunicación Frontend ⇄ Backend con fetch API
- [x] Subida de imagen drag-drop + file input
- [x] Resultados en tiempo real con barras de confianza

## C. Estética UI/UX Única (15 pts)
- [x] Diseño pastel goth Sanrio (NO plantilla genérica)
- [x] Paleta oscura con acentos púrpura/rosa
- [x] Tipografía personalizada (Playfair Display + Quicksand)
- [x] Identidad visual original

## D. Despliegue Hosting (15 pts)
- [ ] URL pública funcional en HuggingFace Spaces
- [ ] Dockerfile listo (port 7860)
- [ ] requirements-deploy.txt optimizado (sin torch)

## E. Reporte Técnico + Código (25 pts)
- [ ] PDF con justificación del stack tecnológico
- [ ] Código comentado (docstrings en todos los módulos)
- [ ] Explicación de decisiones técnicas
- [ ] Análisis de defensas implementadas

## Extra: Chatbot Gemini (+10 pts)
- [x] Integración con API Gemini
- [x] System prompt restrictivo (solo Sanrio)
- [x] Anti-jailbreak (DAN, "ignore previous", roleplay)
- [x] Manejo graceful sin API key

## Defensas (requisitos del profesor)
- [x] Imagen no-Sanrio → threshold <30% → rechazo
- [x] Imagen corrupta → PIL verify → error message
- [x] Tamaño fuera de rango → validación 32-4000px
- [x] Chatbot jailbreak → system prompt + rejection template
- [x] API key solo en backend (variable de entorno)
- [x] Rate limiting (slowapi: 100 req/min)

## Pendientes antes de entregar
- [ ] Desplegar en HF Spaces
- [ ] Escribir reporte PDF
- [ ] Probar /chat con GEMINI_API_KEY real
