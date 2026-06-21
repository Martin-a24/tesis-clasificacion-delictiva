# Mapa de resultados (outputs del pipeline)

Qué genera cada script y para qué sirve al redactar la tesis. Todos los outputs
caen en `data/labels/`, `data/splits/`, `models/` y `results/`.

## Objetivo 1 — Dataset PeruSAT-1 preprocesado y etiquetado

| Script | Output | Uso en la tesis |
|---|---|---|
| 01 | `data/processed/delitos_limpios/delitos_lima_limpio.{csv,geojson}` | Fuente de delitos depurada; capa de puntos para QGIS |
| 02 | `imagenes_pansharpened/*.TIF` + `pansharpening_log.txt` | Insumo; evidencia de R2 (refinamiento) |
| 02b | `mosaico/mosaico.vrt` | Mosaico unico (R2.1, sin tiles duplicados) |
| 03 | `tiles/*.tif` + `tiles_metadata.csv` | Dataset de tiles; tabla de metadatos (R2) |
| 04 | `data/labels/tiles_labeled.{csv,geojson}`, `umbrales_globales.json`, `etiquetado_reporte.txt`, `distribucion_densidad.png` | Esquema de etiquetado (R3): umbrales, distribucion bajo/medio/alto, figura |
| 05 | `data/splits/{train,val,test}.csv` + `splits_reporte.txt` | Split espacial; tabla de distribucion de clases por split |

## Objetivo 2 — Modelo CNN de clasificacion y evaluacion

| Script | Output | Uso en la tesis |
|---|---|---|
| 06 | `models/{arch}_best.pth`, `_final.pth`; `results/training_log_{arch}_*.csv`, `training_curves_{arch}_*.png` | Modelo entrenado; **curvas de loss/accuracy/F1** (figura) |
| 07 | `results/evaluation_{arch}_*.txt`, `confusion_matrix_{arch}_*.png`, `predictions_{arch}_*.csv`, `metrics_{arch}_*.json` | **Metricas finales** (accuracy, F1 macro/weighted por clase); **matriz de confusion** (figura) |
| 08 (opcional) | `results/comparisons/comparison_*.{csv,png}` | **Tabla/figura comparativa** entre arquitecturas |

## Objetivo 3 — Interpretabilidad y zonificacion

| Script | Output | Uso en la tesis |
|---|---|---|
| 09 | `results/gradcam/<categoria>/*.png`, `montage_<categoria>.png`, `gradcam_resumen.csv` | **R6**: mapas Grad-CAM por categoria; montajes representativos (figuras) |
| (manual) | `docs/R7_tabla_correspondencias.md` | **R7**: tabla Grad-CAM vs CPTED/RAT (se llena con 09) |
| 10 | `data/labels/tiles_predicciones.geojson`, `results/zonificacion/mapa_zonificacion.png`, `coincidencia_reporte.txt` | **R8**: capa QGIS de zonificacion; mapa; coincidencia con delitos |

## Qué traer a la PC para redactar

Todo lo de arriba (menos los `.pth` pesados) se baja con un solo comando:

```bash
./scripts/sync.sh pull-entregables    # -> ~/tesis-entregables/
```

Eso incluye `data/labels/` (etiquetas + predicciones para QGIS) y `results/`
completo (curvas, matriz de confusion, Grad-CAM, zonificacion, comparativas).

## Checklist de figuras/tablas para la tesis

- [ ] Distribucion de clases (`distribucion_densidad.png`, `splits_reporte.txt`)
- [ ] Curvas de entrenamiento (`training_curves_*.png`)
- [ ] Matriz de confusion (`confusion_matrix_*.png`)
- [ ] Metricas por clase (`evaluation_*.txt` / `metrics_*.json`)
- [ ] (Opcional) Comparativa de arquitecturas (`comparison_*.png`)
- [ ] Grad-CAM por categoria (`montage_<categoria>.png`)
- [ ] Tabla de coherencia R7 (`docs/R7_tabla_correspondencias.md`)
- [ ] Mapa de zonificacion en QGIS (`tiles_predicciones.geojson`) + coincidencia
