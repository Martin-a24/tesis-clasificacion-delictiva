#!/usr/bin/env python3
"""
03_generar_tiles.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Segmenta el mosaico pansharpened en tiles de 512x512 pixeles, aplicando filtros:
  1. Pixeles validos: descarta tiles con muchos pixeles cero (bordes).
  2. Filtro de agua (NDWI): descarta tiles que sean mayoritariamente mar.
  3. Filtro de nubes (opcional).
  4. Filtro geografico: solo conserva tiles dentro del area urbana de Lima.

A diferencia de la version anterior (que recorria cada imagen por separado y
podia generar tiles duplicados en el solape entre escenas), ahora:
  - Tesela un MOSAICO VIRTUAL (VRT) unico, asi cada ubicacion fisica existe una
    sola vez (sin duplicados entre escenas).
  - Alinea las ventanas a una GRILLA GLOBAL FIJA anclada en config, de modo que
    el cell_id de cada tile sea estable entre corridas y al agregar escenas.
  - Teselado en paralelo (multiprocessing) para escalar.

Si mosaico.enabled = false, mantiene el comportamiento por imagen (compat).

Entrada:
    data/processed/mosaico/mosaico.vrt   (o data/processed/imagenes_pansharpened/*.TIF en compat)

Salida:
    data/processed/tiles/tile_IX_IY.tif
    data/processed/tiles/tiles_metadata.csv

Uso:
    python scripts/03_generar_tiles.py
"""

import os
import csv
import sys
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

try:
    import rasterio
    from rasterio.windows import Window
    import geopandas as gpd
    from shapely.geometry import box
    from shapely import wkb as shapely_wkb
except ImportError:
    print("Error: librerias no instaladas. Activa el environment:")
    print("  conda activate tesis")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

INPUT_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["pansharpened"]
OUTPUT_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["tiles"]
LIMITES_PATH = PROJECT_ROOT / CFG["paths"]["raw"]["limites_urbanos"]

MOSAICO_CFG = CFG.get("mosaico", {})
USE_MOSAICO = MOSAICO_CFG.get("enabled", False)
MOSAICO_VRT = PROJECT_ROOT / MOSAICO_CFG.get("output", "data/processed/mosaico/mosaico.vrt")

GRID_CFG = CFG.get("grilla_global", {})
ORIGIN_X = GRID_CFG.get("origin_x", 240000.0)
ORIGIN_Y = GRID_CFG.get("origin_y", 8740000.0)
STEP_M = GRID_CFG.get("step_m", 358.4)

TILE_SIZE = CFG["tiles"]["size"]
MIN_VALID_RATIO = CFG["tiles"]["min_valid_ratio"]
NDWI_THRESHOLD = CFG["tiles"]["ndwi_water_threshold"]
MAX_WATER_RATIO = CFG["tiles"]["max_water_ratio"]
BAND_GREEN = CFG["tiles"]["ndwi_band_green"]
BAND_NIR = CFG["tiles"]["ndwi_band_nir"]
USE_URBAN = CFG["tiles"].get("use_urban_filter", True)
MIN_URBAN_OVERLAP = CFG["tiles"].get("min_urban_overlap", 0.5)
USE_CLOUD = CFG["tiles"].get("use_cloud_filter", False)
CLOUD_THRESHOLD = CFG["tiles"].get("cloud_brightness_threshold", 1100)
MAX_CLOUD_RATIO = CFG["tiles"].get("max_cloud_ratio", 0.8)
NUM_WORKERS = CFG["tiles"].get("num_workers", 0) or os.cpu_count()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILTROS (compartidos por ambos modos)
# ============================================================

def calcular_ratio_validos(data, nodata=0):
    if data.ndim == 3:
        mascara_nodata = np.all(data == nodata, axis=0)
    else:
        mascara_nodata = (data == nodata)
    return 1.0 - (mascara_nodata.sum() / mascara_nodata.size)


def calcular_ratio_agua(data, banda_verde_idx, banda_nir_idx, umbral_ndwi):
    if data.ndim != 3 or data.shape[0] < max(banda_verde_idx, banda_nir_idx):
        return 0.0
    verde = data[banda_verde_idx - 1].astype(np.float32)
    nir = data[banda_nir_idx - 1].astype(np.float32)
    suma = verde + nir
    suma[suma == 0] = 1
    ndwi = (verde - nir) / suma
    return (ndwi > umbral_ndwi).sum() / ndwi.size


def calcular_ratio_nubes(data, brightness_thr):
    if data.ndim != 3 or data.shape[0] < 3:
        return 0.0
    brillo = data[:3].astype(np.float32).mean(axis=0)
    return (brillo > brightness_thr).sum() / brillo.size


def calcular_overlap_urbano(tile_geom, poligono_urbano):
    if not tile_geom.intersects(poligono_urbano):
        return 0.0
    return tile_geom.intersection(poligono_urbano).area / tile_geom.area


def cargar_poligono_urbano(path, crs_destino):
    if not path.exists():
        print(f"  ERROR: No se encontro {path}")
        print(f"  Ejecuta: python scripts/descargar_limites_lima.py")
        return None
    print(f"  Cargando limites urbanos: {path.name}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs != crs_destino:
        gdf = gdf.to_crs(crs_destino)
    poligono = gdf.geometry.union_all()
    print(f"  Area urbana: {poligono.area / 1e6:.0f} km2")
    return poligono


# ============================================================
# ALINEACION A GRILLA GLOBAL
# ============================================================

def calcular_alineacion(src):
    """
    Devuelve (off_x, off_y, ix0, iy0): offset en pixeles de la primera ventana
    que cae dentro del raster alineada al ancla global, y el indice de celda
    global de esa primera ventana.
    """
    left, top = src.transform.c, src.transform.f
    res = src.transform.a
    ix0 = int(np.ceil((left - ORIGIN_X) / STEP_M))
    iy0 = int(np.ceil((ORIGIN_Y - top) / STEP_M))
    off_x = int(round((ORIGIN_X + ix0 * STEP_M - left) / res))
    off_y = int(round((top - (ORIGIN_Y - iy0 * STEP_M)) / res))
    return off_x, off_y, ix0, iy0


# ============================================================
# TESELADO GLOBAL (mosaico) - worker para multiprocessing
# ============================================================

def _procesar_filas_globales(args):
    """Procesa un bloque de filas de la grilla global. Cada worker abre su
    propio handle de rasterio (no se comparte el dataset entre procesos)."""
    (raster_path, output_dir, filas, off_x, off_y, ix0, iy0, n_cols,
     params, poligono_wkb, source_label) = args

    poligono = shapely_wkb.loads(poligono_wkb) if poligono_wkb else None
    tile_size = params["tile_size"]

    tiles_info = []
    gen = inv = agua = nube = no_urb = 0

    with rasterio.open(raster_path) as src:
        n_bandas = src.count
        res_x, _ = src.res
        crs_str = str(src.crs)
        profile_base = src.profile.copy()

        for i in filas:
            for j in range(n_cols):
                px = off_x + j * tile_size
                py = off_y + i * tile_size
                window = Window(px, py, tile_size, tile_size)
                data = src.read(window=window)

                # Filtro 1: bordes
                ratio_validos = calcular_ratio_validos(data)
                if ratio_validos < params["min_ratio"]:
                    inv += 1
                    continue

                # Filtro 2: agua
                ratio_agua = calcular_ratio_agua(
                    data, params["band_green"], params["band_nir"], params["ndwi_thr"])
                if ratio_agua > params["max_water"]:
                    agua += 1
                    continue

                # Filtro 3: nubes (opcional)
                ratio_nubes = 0.0
                if params["use_cloud"]:
                    ratio_nubes = calcular_ratio_nubes(data, params["cloud_thr"])
                    if ratio_nubes > params["max_cloud"]:
                        nube += 1
                        continue

                # Geometria del tile
                tile_transform = src.window_transform(window)
                left = tile_transform.c
                top = tile_transform.f
                right = left + tile_size * tile_transform.a
                bottom = top + tile_size * tile_transform.e
                tile_geom = box(left, bottom, right, top)

                # Filtro 4: urbano
                urban_overlap = 1.0
                if params["use_urban"] and poligono is not None:
                    urban_overlap = calcular_overlap_urbano(tile_geom, poligono)
                    if urban_overlap < params["min_overlap"]:
                        no_urb += 1
                        continue

                # Indice de celda global (estable entre corridas)
                ix = ix0 + j
                iy = iy0 + i
                cell_id = f"{ix}_{iy}"

                cx = (left + right) / 2
                cy = (top + bottom) / 2

                tile_name = f"tile_{ix:05d}_{iy:05d}.tif"
                tile_path = output_dir / tile_name

                profile = profile_base.copy()
                profile.update(
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    transform=tile_transform,
                    compress="lzw",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                )

                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(data)

                gen += 1
                tiles_info.append({
                    "tile_name": tile_name,
                    "source_image": source_label,
                    "fecha": params["fecha"],
                    "row": iy,
                    "col": ix,
                    "center_x": round(cx, 2),
                    "center_y": round(cy, 2),
                    "valid_ratio": round(ratio_validos, 4),
                    "water_ratio": round(ratio_agua, 4),
                    "cloud_ratio": round(ratio_nubes, 4),
                    "urban_overlap": round(urban_overlap, 4),
                    "crs": crs_str,
                    "res_m": round(res_x, 2),
                    "n_bands": n_bandas,
                    "cell_id": cell_id,
                })

    return tiles_info, gen, inv, agua, nube, no_urb


def teselar_mosaico(raster_path, output_dir, poligono, source_label):
    """Tesela un raster (tipicamente el VRT) sobre la grilla global, en paralelo."""
    with rasterio.open(raster_path) as src:
        ancho, alto = src.width, src.height
        res_x, _ = src.res
        n_bandas = src.count
        off_x, off_y, ix0, iy0 = calcular_alineacion(src)

    print(f"    Dimensiones: {ancho}x{alto} px, {n_bandas} bandas, {res_x:.2f} m/px")
    n_cols = (ancho - off_x) // TILE_SIZE
    n_rows = (alto - off_y) // TILE_SIZE
    if n_cols <= 0 or n_rows <= 0:
        print("    ADVERTENCIA: el raster no cubre ninguna celda completa de la grilla.")
        return [], 0, 0, 0, 0, 0
    print(f"    Grilla global alineada: {n_cols} x {n_rows} = {n_cols * n_rows} ventanas "
          f"(ancla ix0={ix0}, iy0={iy0})")

    params = {
        "tile_size": TILE_SIZE,
        "min_ratio": MIN_VALID_RATIO,
        "band_green": BAND_GREEN,
        "band_nir": BAND_NIR,
        "ndwi_thr": NDWI_THRESHOLD,
        "max_water": MAX_WATER_RATIO,
        "use_cloud": USE_CLOUD,
        "cloud_thr": CLOUD_THRESHOLD,
        "max_cloud": MAX_CLOUD_RATIO,
        "use_urban": USE_URBAN,
        "min_overlap": MIN_URBAN_OVERLAP,
        "fecha": "mosaico",
    }
    poligono_wkb = poligono.wkb if poligono is not None else None

    n_workers = max(1, min(NUM_WORKERS, n_rows))
    bloques = [b for b in np.array_split(np.arange(n_rows), n_workers) if len(b) > 0]
    print(f"    Procesando con {n_workers} proceso(s)...")

    args_list = [
        (str(raster_path), output_dir, [int(i) for i in bloque],
         off_x, off_y, ix0, iy0, n_cols, params, poligono_wkb, source_label)
        for bloque in bloques
    ]

    todos = []
    gen = inv = agua = nube = no_urb = 0
    if n_workers == 1:
        resultados = [_procesar_filas_globales(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            resultados = list(ex.map(_procesar_filas_globales, args_list))

    for info, g, i, a, n, u in resultados:
        todos.extend(info)
        gen += g; inv += i; agua += a; nube += n; no_urb += u

    return todos, gen, inv, agua, nube, no_urb


# ============================================================
# TESELADO POR IMAGEN (compat: mosaico.enabled = false)
# ============================================================

def extraer_fecha(nombre):
    partes = nombre.replace(".TIF", "").replace(".tif", "").split("_")
    if len(partes) >= 3:
        return partes[2][:8]
    return "unknown"


def procesar_imagen(imagen_path, output_dir):
    """Modo compat: tesela una imagen alineada a la grilla global. Conserva la
    nomenclatura legacy tile_fecha_row_col pero agrega cell_id estable."""
    fecha = extraer_fecha(imagen_path.name)
    poligono = cargar_poligono_urbano(LIMITES_PATH, _crs_imagen(imagen_path)) if USE_URBAN else None
    tiles_info = []
    gen = inv = agua = nube = no_urb = 0

    with rasterio.open(imagen_path) as src:
        n_bandas = src.count
        ancho, alto = src.width, src.height
        res_x, _ = src.res
        off_x, off_y, ix0, iy0 = calcular_alineacion(src)

        print(f"    Dimensiones: {ancho}x{alto} px, {n_bandas} bandas, {res_x:.2f} m/px")
        n_cols = (ancho - off_x) // TILE_SIZE
        n_rows = (alto - off_y) // TILE_SIZE
        print(f"    Grilla: {n_cols} x {n_rows} = {n_cols * n_rows} tiles")

        for i in range(n_rows):
            for j in range(n_cols):
                window = Window(off_x + j * TILE_SIZE, off_y + i * TILE_SIZE,
                                TILE_SIZE, TILE_SIZE)
                data = src.read(window=window)

                ratio_validos = calcular_ratio_validos(data)
                if ratio_validos < MIN_VALID_RATIO:
                    inv += 1
                    continue

                ratio_agua = calcular_ratio_agua(data, BAND_GREEN, BAND_NIR, NDWI_THRESHOLD)
                if ratio_agua > MAX_WATER_RATIO:
                    agua += 1
                    continue

                ratio_nubes = 0.0
                if USE_CLOUD:
                    ratio_nubes = calcular_ratio_nubes(data, CLOUD_THRESHOLD)
                    if ratio_nubes > MAX_CLOUD_RATIO:
                        nube += 1
                        continue

                tile_transform = src.window_transform(window)
                left = tile_transform.c
                top = tile_transform.f
                right = left + TILE_SIZE * tile_transform.a
                bottom = top + TILE_SIZE * tile_transform.e
                tile_geom = box(left, bottom, right, top)

                urban_overlap = 1.0
                if USE_URBAN and poligono is not None:
                    urban_overlap = calcular_overlap_urbano(tile_geom, poligono)
                    if urban_overlap < MIN_URBAN_OVERLAP:
                        no_urb += 1
                        continue

                ix = ix0 + j
                iy = iy0 + i
                cell_id = f"{ix}_{iy}"
                cx = (left + right) / 2
                cy = (top + bottom) / 2

                tile_name = f"tile_{fecha}_{iy:03d}_{ix:03d}.tif"
                tile_path = output_dir / tile_name

                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    width=TILE_SIZE,
                    height=TILE_SIZE,
                    transform=tile_transform,
                    compress="lzw",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                )
                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(data)

                gen += 1
                tiles_info.append({
                    "tile_name": tile_name,
                    "source_image": imagen_path.name,
                    "fecha": fecha,
                    "row": iy,
                    "col": ix,
                    "center_x": round(cx, 2),
                    "center_y": round(cy, 2),
                    "valid_ratio": round(ratio_validos, 4),
                    "water_ratio": round(ratio_agua, 4),
                    "cloud_ratio": round(ratio_nubes, 4),
                    "urban_overlap": round(urban_overlap, 4),
                    "crs": str(src.crs),
                    "res_m": round(res_x, 2),
                    "n_bands": n_bandas,
                    "cell_id": cell_id,
                })

    return tiles_info, gen, inv, agua, nube, no_urb


def _crs_imagen(path):
    with rasterio.open(path) as src:
        return src.crs


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SEGMENTACION EN TILES")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Tile:   {TILE_SIZE}x{TILE_SIZE} px")
    print(f"  Modo:   {'MOSAICO VRT + grilla global' if USE_MOSAICO else 'por imagen (compat)'}")
    print(f"\n  Filtros activados:")
    print(f"    Validez minima:  {MIN_VALID_RATIO * 100:.0f}%")
    print(f"    Max agua:        {MAX_WATER_RATIO * 100:.0f}% (NDWI > {NDWI_THRESHOLD})")
    if USE_CLOUD:
        print(f"    Max nubes:       {MAX_CLOUD_RATIO * 100:.0f}%")
    else:
        print(f"    Filtro nubes:    DESACTIVADO")
    if USE_URBAN:
        print(f"    Min urbano:      {MIN_URBAN_OVERLAP * 100:.0f}%")
    else:
        print(f"    Filtro urbano:   DESACTIVADO")

    todos = []
    tot_gen = tot_inv = tot_agua = tot_nube = tot_no_urb = 0

    if USE_MOSAICO:
        # ---- Modo mosaico ----
        if not MOSAICO_VRT.exists():
            print(f"\n  ERROR: No existe el mosaico {MOSAICO_VRT}")
            print(f"  Ejecuta primero: python scripts/02b_construir_mosaico.py")
            sys.exit(1)
        print(f"  Input:  {MOSAICO_VRT}")
        print(f"  Grilla global: ancla=({ORIGIN_X:.0f}, {ORIGIN_Y:.0f}), paso={STEP_M} m")

        poligono = None
        if USE_URBAN:
            with rasterio.open(MOSAICO_VRT) as src:
                crs_destino = src.crs
            poligono = cargar_poligono_urbano(LIMITES_PATH, crs_destino)
            if poligono is None:
                sys.exit(1)

        print(f"\n  [1/1] {MOSAICO_VRT.name}")
        todos, tot_gen, tot_inv, tot_agua, tot_nube, tot_no_urb = teselar_mosaico(
            MOSAICO_VRT, OUTPUT_DIR, poligono, "mosaico")
        print(f"    Validos:               {tot_gen}")
        print(f"    Descartados borde:     {tot_inv}")
        print(f"    Descartados agua:      {tot_agua}")
        if USE_CLOUD:
            print(f"    Descartados nubes:     {tot_nube}")
        if USE_URBAN:
            print(f"    Descartados no urbano: {tot_no_urb}")
    else:
        # ---- Modo compat por imagen ----
        if not INPUT_DIR.exists():
            print(f"\n  ERROR: No se encontro {INPUT_DIR}")
            sys.exit(1)
        imagenes = sorted(INPUT_DIR.glob("*.TIF")) + sorted(INPUT_DIR.glob("*.tif"))
        if not imagenes:
            print(f"\n  No se encontraron imagenes en {INPUT_DIR}")
            sys.exit(1)
        print(f"  Input:  {INPUT_DIR}")
        print(f"\n  Imagenes encontradas: {len(imagenes)}")

        for i, img in enumerate(imagenes, 1):
            print(f"\n  [{i}/{len(imagenes)}] {img.name}")
            info, gen, inv, agua, nube, no_urb = procesar_imagen(img, OUTPUT_DIR)
            todos.extend(info)
            tot_gen += gen; tot_inv += inv; tot_agua += agua
            tot_nube += nube; tot_no_urb += no_urb
            print(f"    Validos:               {gen}")
            print(f"    Descartados borde:     {inv}")
            print(f"    Descartados agua:      {agua}")
            if USE_CLOUD:
                print(f"    Descartados nubes:     {nube}")
            if USE_URBAN:
                print(f"    Descartados no urbano: {no_urb}")

    # ---- Deduplicacion: ningun cell_id puede repetirse ----
    cell_ids = [t["cell_id"] for t in todos]
    repetidos = [c for c, n in Counter(cell_ids).items() if n > 1]
    if repetidos:
        print(f"\n  ERROR: {len(repetidos)} cell_id repetidos (ej: {repetidos[:5]}).")
        print(f"  En modo mosaico esto no deberia ocurrir. Abortando para evitar fuga espacial.")
        sys.exit(1)

    if todos:
        meta_path = OUTPUT_DIR / "tiles_metadata.csv"
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=todos[0].keys())
            writer.writeheader()
            writer.writerows(todos)

    total = tot_gen + tot_inv + tot_agua + tot_nube + tot_no_urb
    tasa = tot_gen / max(1, total) * 100

    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"{'='*60}")
    print(f"  Tiles validos:              {tot_gen}")
    print(f"  cell_id unicos:             {len(set(cell_ids))}")
    print(f"  Descartados (bordes):       {tot_inv}")
    print(f"  Descartados (agua/mar):     {tot_agua}")
    if USE_CLOUD:
        print(f"  Descartados (nubes):        {tot_nube}")
    if USE_URBAN:
        print(f"  Descartados (no urbano):    {tot_no_urb}")
    print(f"  Total potencial:            {total}")
    print(f"  Tasa aprovechamiento:       {tasa:.1f}%")

    if todos:
        res = todos[0]["res_m"]
        metros = TILE_SIZE * res
        print(f"\n  Tamano por tile:  {TILE_SIZE}x{TILE_SIZE} px")
        print(f"  Resolucion:       {res} m/px")
        print(f"  Cobertura:        ~{metros:.0f}x{metros:.0f} m")
        print(f"\n  Tiles por fuente:")
        for img, count in sorted(Counter(t["source_image"] for t in todos).items()):
            print(f"    {img}: {count}")

    print(f"\n  Tiles en:     {OUTPUT_DIR}")
    print(f"  Metadatos en: {OUTPUT_DIR / 'tiles_metadata.csv'}")
    print(f"\n  SIGUIENTE PASO: python scripts/04_etiquetar_tiles.py")
    print(f"{'='*60}")
