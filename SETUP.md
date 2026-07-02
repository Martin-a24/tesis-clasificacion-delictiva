# Guia de instalacion

Instrucciones para dejar el proyecto listo en una PC o en un servidor.

## 1. Requisitos

- **Conda** (Miniconda o Anaconda). Se usa porque GDAL/rasterio/geopandas se
  instalan mucho mas facil por conda-forge que por pip.
- **Git**.
- **GPU NVIDIA** (opcional): acelera el entrenamiento (scripts 06/08). Sin GPU el
  pipeline funciona igual en CPU (mas lento); ajusta `entrenamiento.device: cpu`
  en `configs/config.yaml`.

## 2. Clonar y crear el entorno

```bash
git clone https://github.com/Martin-a24/tesis-clasificacion-delictiva
cd tesis-clasificacion-delictiva
conda env create -f environment.yml
conda activate tesis
```

Verificar que todo importa:

```bash
python -c "import torch, rasterio, geopandas, sklearn; from osgeo import gdal; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gdal', gdal.__version__)"
```

## 3. Preparar los datos

Los datos pesados no se versionan. Tras clonar, coloca los insumos en
`data/raw/` segun la tabla **"Datos"** del [README](README.md):

- Imagenes PeruSAT-1 (MS + PAN) en `data/raw/imagenes_perusat/{ESPECTRAL,PANCROMATICA}/`.
- Limites de Lima: `python scripts/descargar_limites_lima.py`.
- Delitos: ya vienen de ejemplo en `data/raw/delitos/` (reemplazables).
- WorldPop (opcional): solo si activas `etiquetado.normalizar_poblacion`.

## 4. Operar el proyecto

Todo se maneja desde el panel de control:

```bash
python scripts/panel.py          # menu: estado, correr pasos, configurar, ver inputs
python scripts/panel.py status   # solo el estado del pipeline
```

## 5. Servidor separado (opcional)

Si editas el codigo en una PC y ejecutas en un servidor, ver la seccion
**"Sincronizacion con el servidor"** del [README](README.md) (`scripts/sync.sh`
con rsync + SSH). Ajusta en `config.yaml` lo especifico de la maquina
(`entrenamiento.device`, `num_workers`, `batch_size`).

## Notas

- Todos los parametros estan en `configs/config.yaml`; ninguno esta hardcodeado.
- Los scripts se ejecutan desde la **raiz del proyecto** (`python scripts/XX.py`),
  no hace falta entrar a `scripts/`.
