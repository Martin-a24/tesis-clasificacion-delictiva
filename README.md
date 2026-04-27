# Tesis: Clasificacion de Zonas de Riesgo Delictivo


**Autor:** Martin Adrian Sayago Subelete
**Asesor:** Ferdinand Edgardo Pineda Ancco

## Resumen

Modelo de redes neuronales convolucionales (CNN) que clasifica zonas de
Lima Metropolitana y Callao segun su nivel de riesgo de delitos patrimoniales,
a partir del analisis de imagenes satelitales del entorno urbano (PeruSAT-1)
combinadas con datos delictivos georreferenciados del MININTER. La interpretabilidad
del modelo se aborda mediante Grad-CAM y se valida frente a la literatura de
criminologia ambiental.

## Estructura del proyecto

```
tesis-clasificacion-delictiva/
├── data/                  # Datos crudos y procesados (excluido de Git)
├── scripts/               # Pipeline de procesamiento numerado
├── configs/               # Configuracion (config.yaml)
├── notebooks/             # Exploracion en Jupyter
├── models/                # Modelos entrenados
├── results/               # Metricas y figuras
├── docs/                  # Documentacion adicional
├── environment.yml        # Dependencias conda
└── SETUP.md               # Guia de instalacion
```

## Instalacion

Ver SETUP.md para la guia completa de instalacion en PC y servidor.

Resumen:

```bash
conda env create -f environment.yml
conda activate tesis
```

## Pipeline

1. **01_limpiar_datos_delictivos.py** - Consolida CSVs del MININTER en un
   dataset filtrado de delitos patrimoniales georreferenciados.
2. **02_pansharpening.py** - Aplica refinamiento pancromatico a imagenes
   PeruSAT-1 (resolucion 2.8m -> 0.7m, 4 bandas).
3. **03_generar_tiles.py** - Segmenta imagenes en tiles de 512x512 px,
   descartando bordes y zonas de agua via NDWI.
4. **04_etiquetar_tiles.py** - Cruza tiles con delitos y asigna nivel de
   riesgo (bajo/medio/alto) por percentiles de densidad.
5. **05_construir_splits.py** - (Pendiente) Genera train/val/test.
6. **06_entrenar_modelo.py** - (Pendiente) Entrena CNN.
7. **07_evaluar_modelo.py** - (Pendiente) Evalua metricas.
8. **08_gradcam.py** - (Pendiente) Genera mapas de interpretabilidad.

## Datos

Los datos pesados (imagenes satelitales, modelos entrenados, tiles) no se
versionan en Git. Solo el codigo se versiona via Git.

