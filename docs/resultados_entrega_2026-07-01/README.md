# Resultados de la entrega (corrida del 01/07/2026)

Resultados finales reportados en el documento de tesis (E3). Corrida
determinista (semilla 42) sobre el dataset definitivo: 3 escenas PerúSAT-1
(Callao 06/02/2022 + 2 escenas de Lima central 29/11/2025), 2,603 tiles,
16,596 delitos del 2025. Modelo seleccionado: **ResNet-18**
(Acc 0.772 / F1-macro 0.704 / F1-ponderado 0.782 en test).

Esta carpeta versiona los artefactos livianos citados en la tesis. Los
artefactos pesados (modelos `.pth`, tiles GeoTIFF, los 372 overlays Grad-CAM
individuales) no se versionan; ver la sección "Reproducir" más abajo o el
Release del repositorio.

## Contenido

| Carpeta | Contenido | Uso en la tesis |
|---|---|---|
| `evaluacion/` | Reporte, métricas JSON, matriz de confusión y predicciones por tile de ResNet-18 | Tablas 25-27, Figura 10 |
| `comparacion/` | Reporte comparativo de las 3 arquitecturas + evaluaciones de ResNet-50 y EfficientNet-B0 | Tabla 28, Figura 11 |
| `entrenamiento/` | Logs y curvas de entrenamiento de las 3 arquitecturas | Figura 9, Anexo F |
| `gradcam/` | Montajes representativos por categoría y resumen cuantitativo | Figuras 12-14, Tabla 29 (R6/R7) |
| `zonificacion/` | Mapa, reporte de coincidencia espacial y capa GeoJSON de predicciones | Figura 15, Tabla 30 (R8) |
| `etiquetado/` | Reporte de etiquetado, umbrales persistidos, histograma y etiquetas por tile | Tablas 16-18, Figura 7 (R3) |

## Reproducir

Los resultados se regeneran de extremo a extremo con el pipeline del
repositorio (`python scripts/panel.py` o los scripts 01-10 en orden), con la
configuración versionada en `configs/config.yaml`. Los umbrales de etiquetado
(`etiquetado/umbrales_globales.json`) y la grilla global fija garantizan que
corridas con datos adicionales (nuevas escenas o delitos de otros años) sean
comparables celda a celda con estos resultados.
