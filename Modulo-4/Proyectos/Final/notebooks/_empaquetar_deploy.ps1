# Empaquetar archivos para deploy en Hugging Face Spaces
# Ejecutar desde Modulo-4/Proyectos/Final/

$dest = "_deploy_hf"

# Crear directorios
New-Item -ItemType Directory -Path "$dest/app/static" -Force | Out-Null
New-Item -ItemType Directory -Path "$dest/models" -Force | Out-Null

# App Python
Copy-Item -Path "app/*.py" -Destination "$dest/app/" -Force

# Frontend
Copy-Item -Path "app/static/index.html" -Destination "$dest/app/static/" -Force
Copy-Item -Path "app/static/personajes.json" -Destination "$dest/app/static/" -Force

# Modelo
Copy-Item -Path "models/tl_sanrio_int8.onnx" -Destination "$dest/models/" -Force
Copy-Item -Path "models/clases_sanrio.json" -Destination "$dest/models/" -Force

# Deploy config
Copy-Item -Path "Dockerfile" -Destination "$dest/" -Force
Copy-Item -Path "requirements-deploy.txt" -Destination "$dest/" -Force
Copy-Item -Path ".env.example" -Destination "$dest/" -Force
@"
*
!app/
!models/
!requirements-deploy.txt
!.env.example
!Dockerfile
"@ | Out-File -FilePath "$dest/.dockerignore" -Encoding ASCII

Write-Host "Paquete de deploy creado en: $dest"
Write-Host "Tamanio: $(Get-ChildItem $dest -Recurse | Measure-Object Length -Sum | Select-Object -ExpandProperty Sum)/1KB KB"
