# Tesis: Clasificación de Zonas de Riesgo Delictivo

## Pipeline de ejecución

1. `scripts/01_limpiar_datos_delictivos.py` - Consolida CSVs del MININTER
2. `scripts/02_pansharpening.py` - Aplica pansharpening a imágenes PerúSAT-1
3. `scripts/03_generar_tiles.py` - Segmenta en tiles 512x512
4. `scripts/04_etiquetar_tiles.py` - Cruza con delitos y etiqueta por riesgo
5. `scripts/05_construir_splits.py` - Genera train/val/test
6. `scripts/06_entrenar_modelo.py` - Entrena CNN
7. `scripts/07_evaluar_modelo.py` - Evalúa con métricas
8. `scripts/08_gradcam.py` - Genera mapas de interpretabilidad