# Plan R2.1: manejo de solape entre escenas, grilla global y split espacial

Documento de trabajo para ejecutar con Claude Code sobre el repo
`tesis-clasificacion-delictiva`. Objetivo: que el pipeline sea correcto y
escalable cuando llegan varias imagenes PeruSAT-1 que se solapan
geograficamente.

## 1. Problema que se resuelve

El teselado actual (script 03) recorre cada imagen con ventana deslizante sin
superposicion. Eso evita solape **dentro** de una imagen, pero no entre
escenas distintas. Cuando dos escenas cubren el mismo sector (caso real
observado en QGIS, escenas `..._000054` y `..._000444` de la misma fecha), se
generan dos tiles del mismo lugar fisico con nombres distintos
(`tile_fecha_row_col`). Consecuencias:

1. La misma ubicacion puede caer en train y en test a la vez (fuga espacial),
   inflando las metricas.
2. La verificacion de no-overlap del script 05 solo compara `tile_name`, asi
   que no detecta este caso.
3. El etiquetado (script 04) asigna casi la misma etiqueta a ubicaciones
   duplicadas, distorsionando balance de clases y percentiles.

## 2. Decision de diseno

Tres piezas:

1. **Mosaico virtual (VRT)** de todas las escenas pansharpened, con
   `nodata = 0` para que los bordes negros rotados no tapen pixeles validos de
   la escena vecina en el solape. El mosaico hace que cada ubicacion exista una
   sola vez.
2. **Grilla global fija** anclada a un origen en EPSG:32718 definido en config,
   de modo que el `cell_id` de cada tile sea estable y no dependa de la escena
   ni cambie cuando lleguen mas imagenes. Las ventanas de teselado se alinean a
   multiplos de 512 px desde ese ancla.
3. **Split espacial agrupado** en el script 05: ningun `cell_id` puede quedar
   repartido entre train/val/test. La verificacion se hace sobre `cell_id`, no
   sobre `tile_name`.

Ademas, teselado en paralelo (multiprocessing) para responder a la sugerencia
del validador sobre escalabilidad.

## 3. Cambios por archivo

### 3.1 `configs/config.yaml`

Agregar bloque de mosaico y de grilla global, y parametros de split agrupado:

```yaml
mosaico:
  enabled: true
  crs: "EPSG:32718"
  src_nodata: 0
  vrt_nodata: 0
  # orden de prioridad en zonas de solape: la ultima escena listada queda
  # "arriba". Ordenar por calidad (menos nubes / menos borde) o dejar "auto"
  # para ordenar por valid_ratio descendente calculado en 02.
  prioridad: "auto"
  output: "data/processed/mosaico/mosaico.vrt"

grilla_global:
  # ancla fija (esquina NO) en EPSG:32718. Elegir un punto al NO del area de
  # estudio y NO cambiarlo nunca, para que cell_id sea estable entre corridas.
  origin_x: 240000.0
  origin_y: 8740000.0
  # paso en metros = tile_size_px * resolucion (512 * 0.7 = 358.4)
  step_m: 358.4

splits:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
  stratify_by: "nivel_riesgo"
  group_by: "cell_id"        # NUEVO: agrupar por ubicacion para split espacial
```

### 3.2 Nuevo script `scripts/02b_construir_mosaico.py`

Construye el VRT a partir de las pansharpened. Boceto:

```python
# Lista de escenas pansharpened ordenadas por calidad (mejor al final, queda
# arriba en el solape). Para "auto", ordenar por % de pixeles validos asc.
from osgeo import gdal

gdal.BuildVRT(
    str(OUTPUT_VRT),
    [str(p) for p in escenas_ordenadas],
    options=gdal.BuildVRTOptions(
        srcNodata=CFG["mosaico"]["src_nodata"],
        VRTNodata=CFG["mosaico"]["vrt_nodata"],
        resolution="highest",
    ),
)
```

Notas:
- El VRT es virtual, no copia datos, asi que es barato reconstruirlo cada vez
  que llegan escenas nuevas.
- Verificar que todas las escenas comparten CRS EPSG:32718 antes de unir; si
  alguna difiere, reproyectar (gdalwarp) primero.

### 3.3 `scripts/03_generar_tiles.py`

Cambios:

1. **Entrada**: en vez de iterar `INPUT_DIR.glob("*.TIF")`, abrir el mosaico
   VRT (`CFG["mosaico"]["output"]`). Si `mosaico.enabled` es false, mantener el
   comportamiento por imagen (compatibilidad).
2. **Ventanas alineadas a grilla global**: calcular el offset para que las
   ventanas caigan en multiplos de 512 px desde el ancla fija, no desde el
   pixel (0,0) del raster:

   ```python
   step_px = TILE_SIZE
   step_m  = CFG["grilla_global"]["step_m"]
   Ox = CFG["grilla_global"]["origin_x"]
   Oy = CFG["grilla_global"]["origin_y"]

   left, top = src.transform.c, src.transform.f
   res = src.transform.a
   # primer indice de celda global que cae dentro del raster
   ix0 = int(np.ceil((left - Ox) / step_m))
   iy0 = int(np.ceil((Oy - top) / step_m))
   # offset en pixeles dentro del raster para alinear al ancla
   off_x = int(round((Ox + ix0 * step_m - left) / res))
   off_y = int(round((top - (Oy - iy0 * step_m)) / res))
   ```

   Recorrer las ventanas desde `off_x, off_y` con paso `TILE_SIZE`, y asignar a
   cada tile su indice de celda global `(ix, iy)`.
3. **Nombre y metadatos**: `tile_name = f"tile_{ix:05d}_{iy:05d}.tif"` y nueva
   columna `cell_id = f"{ix}_{iy}"`. Mantener `center_x`, `center_y`,
   `valid_ratio`, `water_ratio`, `urban_overlap`, `crs`, `res_m`, `n_bands`.
   `source_image` pasa a ser informativo (queda "mosaico").
4. **Deduplicacion**: como se tesela el mosaico (grilla unica), no se generan
   duplicados. Mantener igual una verificacion: si `cell_id` se repite, abortar
   con error.
5. **Paralelizacion**: repartir las filas de la grilla entre procesos con
   `multiprocessing.Pool` o `concurrent.futures.ProcessPoolExecutor`. Cada
   worker abre su propio handle de rasterio (no compartir el dataset entre
   procesos).

### 3.4 `scripts/04_etiquetar_tiles.py`

Sin cambios de logica, pero arrastrar la columna `cell_id` al
`tiles_labeled.csv` (agregarla a `cols_csv`). El etiquetado por grilla virtual
de 358 m sigue valido.

### 3.5 `scripts/05_construir_splits.py`

1. Reemplazar el split estratificado aleatorio por un split **estratificado y
   agrupado** por `cell_id`:

   ```python
   from sklearn.model_selection import StratifiedGroupKFold

   groups = df["cell_id"].values
   y = df[STRATIFY_BY].values
   # 1ra particion: separar test (~15%)
   sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
   train_val_idx, test_idx = next(sgkf.split(df, y, groups))
   # 2da particion sobre train_val para sacar val (~15% del total)
   ...
   ```

   (Con un solo conjunto de escenas y pocas celdas repetidas el efecto es
   pequeno, pero deja el pipeline blindado para cuando crezca el dataset.)
2. Cambiar la verificacion: comparar conjuntos de `cell_id`, no de `tile_name`:

   ```python
   train_g = set(df_train["cell_id"]); val_g = set(df_val["cell_id"])
   test_g  = set(df_test["cell_id"])
   assert not (train_g & val_g)
   assert not (train_g & test_g)
   assert not (val_g  & test_g)
   ```

## 4. Diagnostico previo (correr antes de tocar nada)

Sobre el `tiles_metadata.csv` actual, medir cuanto solape real hay:

```python
import pandas as pd
df = pd.read_csv("data/processed/tiles/tiles_metadata.csv")
cell = 358.4
df["cell_id"] = ((df.center_y/cell).round().astype(int).astype(str) + "_"
                 + (df.center_x/cell).round().astype(int).astype(str))
rep = df.groupby("cell_id")["source_image"].nunique()
print("Ubicaciones en >1 escena:", (rep > 1).sum(), "de", df.cell_id.nunique())
```

Si sale 0, el solape aun no afecta y basta con dejar el blindaje (mosaico +
split agrupado) documentado. Si sale > 0, el cambio es necesario.

## 5. Verificacion de aceptacion

- [ ] El VRT abre en QGIS y en el solape no se ven bordes negros tapando datos.
- [ ] `tiles_metadata.csv` no tiene `cell_id` repetidos.
- [ ] Reprocesar agregando una escena nueva no cambia el `cell_id` de los tiles
      ya existentes (grilla estable).
- [ ] `splits_reporte.txt`: los tres overlaps de `cell_id` son 0.
- [ ] Distribucion de clases por split sigue estratificada (proporciones
      similares a la global).
- [ ] Tiempo de teselado baja con la version paralela respecto a la secuencial.

## 6. Impacto en la validacion R2

El informe firmado sigue valido a nivel de imagen. Registrar este cambio como
version R2.1 del protocolo (mosaico + grilla global + split espacial),
comentarlo al asesor y, si se desea, anexar una nota corta al informe. No es
obligatorio re-firmar: la extension refuerza el protocolo, no contradice lo
validado. Conviene corregir la justificacion de la Tabla 2 para que diga que el
0% de solape garantiza tiles sin pixeles compartidos, y que la garantia contra
fuga de datos en train/test la provee el split espacial agrupado del script 05.
