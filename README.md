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
├── data/
│   ├── raw/                       # Insumos (no versionados, ver "Datos")
│   │   ├── delitos/               #   CSVs MININTER (SI versionados)
│   │   ├── imagenes_perusat/
│   │   │   ├── ESPECTRAL/         #   <- colocar IMG_PER1_*_MS_*.TIF
│   │   │   └── PANCROMATICA/      #   <- colocar IMG_PER1_*_P_*.TIF
│   │   ├── limites/               #   geojson de Lima (se generan por script)
│   │   └── worldpop/              #   per_ppp_2020.tif (opcional)
│   ├── processed/                 # Salidas de 02/02b/03 (auto-generadas)
│   ├── labels/                    # Salidas de 04
│   └── splits/                    # Salidas de 05
├── scripts/               # Pipeline de procesamiento numerado
├── configs/               # Configuracion (config.yaml)
├── notebooks/             # Exploracion en Jupyter
├── models/                # Modelos entrenados (no versionados)
├── results/               # Metricas y figuras (no versionados)
├── docs/                  # Documentacion adicional
├── environment.yml        # Dependencias conda
└── SETUP.md               # Guia de instalacion
```

Las carpetas de insumos llevan un `.gitkeep` para que la estructura exista al
clonar; los archivos pesados (imagenes, modelos, tiles) no se versionan.

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
3. **02b_construir_mosaico.py** - Une las escenas pansharpened en un mosaico
   virtual (VRT) para que cada ubicacion exista una sola vez (evita tiles
   duplicados en el solape entre escenas).
4. **03_generar_tiles.py** - Segmenta el mosaico en tiles de 512x512 px
   alineados a una grilla global fija (cell_id estable), descartando bordes,
   agua (NDWI) y zonas no urbanas.
5. **04_etiquetar_tiles.py** - Cruza tiles con delitos y asigna nivel de
   riesgo (bajo/medio/alto) por percentiles de densidad.
6. **05_construir_splits.py** - Genera train/val/test con split estratificado
   y agrupado por cell_id (evita fuga espacial).
7. **06_entrenar_modelo.py** - Entrena la CNN (ResNet/EfficientNet/ViT).
8. **07_evaluar_modelo.py** - Evalua metricas sobre el test.
9. **08_comparar_arquitecturas.py** - Compara varias arquitecturas.

Scripts auxiliares:
- **descargar_limites_lima.py** - Descarga/genera los limites de Lima en
  `data/raw/limites/`.
- **limpiar_salidas.py** - Borra las salidas generadas para re-correr el
  pipeline desde cero (conserva `data/raw/`). Usar `--dry-run` para previsualizar.

## Datos

Solo el codigo se versiona en Git. Los datos pesados (imagenes satelitales,
WorldPop, tiles, modelos, resultados) no se suben. Tras **clonar**, prepara los
insumos en `data/raw/` asi:

| Carpeta | Que colocar | Como obtenerlo |
|---|---|---|
| `data/raw/delitos/` | CSVs del MININTER | Ya vienen versionados |
| `data/raw/imagenes_perusat/ESPECTRAL/` | `IMG_PER1_*_MS_*.TIF` (multiespectral) | Imagenes PeruSAT-1 (no publicas) |
| `data/raw/imagenes_perusat/PANCROMATICA/` | `IMG_PER1_*_P_*.TIF` (pancromatica) | Imagenes PeruSAT-1 (no publicas) |
| `data/raw/limites/` | `lima_metropolitana.geojson` | `python scripts/descargar_limites_lima.py` |
| `data/raw/worldpop/` | `per_ppp_2020.tif` (opcional) | WorldPop (solo si `normalizar_poblacion: true`) |

Las demas carpetas (`data/processed/`, `data/labels/`, `data/splits/`, `models/`,
`results/`) se crean automaticamente al ejecutar los scripts.

