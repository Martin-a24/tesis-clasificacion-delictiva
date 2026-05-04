#!/usr/bin/env python3
"""
03_generar_tiles.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Segmenta imagenes pansharpened en tiles de 512x512 pixeles, aplicando tres filtros:
  1. Pixeles validos: descarta tiles con muchos pixeles cero (bordes).
  2. Filtro de agua (NDWI): descarta tiles que sean mayoritariamente mar.
  3. Filtro geografico: solo conserva tiles dentro del area urbana de Lima Metropolitana.

Entrada:
    data/processed/imagenes_pansharpened/*.TIF

Salida:
    data/processed/tiles/tile_YYYYMMDD_row_col.tif
    data/processed/tiles/tiles_metadata.csv

Uso:
    python scripts/03_generar_tiles.py
"""

import csv
import sys
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
 
try:
    import rasterio
    from rasterio.windows import Window
    import geopandas as gpd
    from shapely.geometry import box
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
 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
def extraer_fecha(nombre):
    partes = nombre.replace(".TIF", "").replace(".tif", "").split("_")
    if len(partes) >= 3:
        return partes[2][:8]
    return "unknown"
 
 
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
    poligono = gdf.geometry.unary_union
    print(f"  Area urbana: {poligono.area / 1e6:.0f} km2")
    return poligono
 
 
def procesar_imagen(imagen_path, output_dir, tile_size, min_ratio,
                     band_green, band_nir, ndwi_thr, max_water,
                     use_cloud, cloud_thr, max_cloud,
                     use_urban, poligono, min_overlap):
 
    fecha = extraer_fecha(imagen_path.name)
    tiles_info = []
    gen = inv = agua = nube = no_urb = 0
 
    with rasterio.open(imagen_path) as src:
        n_bandas = src.count
        ancho, alto = src.width, src.height
        res_x, _ = src.res
 
        print(f"    Dimensiones: {ancho}x{alto} px, {n_bandas} bandas, {res_x:.2f} m/px")
 
        n_cols = ancho // tile_size
        n_rows = alto // tile_size
        print(f"    Grilla: {n_cols} x {n_rows} = {n_cols * n_rows} tiles")
 
        for row in range(n_rows):
            for col in range(n_cols):
                window = Window(col * tile_size, row * tile_size, tile_size, tile_size)
                data = src.read(window=window)
 
                # Filtro 1: bordes
                ratio_validos = calcular_ratio_validos(data)
                if ratio_validos < min_ratio:
                    inv += 1
                    continue
 
                # Filtro 2: agua
                ratio_agua = calcular_ratio_agua(data, band_green, band_nir, ndwi_thr)
                if ratio_agua > max_water:
                    agua += 1
                    continue
 
                # Filtro 3: nubes (opcional)
                ratio_nubes = 0.0
                if use_cloud:
                    ratio_nubes = calcular_ratio_nubes(data, cloud_thr)
                    if ratio_nubes > max_cloud:
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
                if use_urban and poligono is not None:
                    urban_overlap = calcular_overlap_urbano(tile_geom, poligono)
                    if urban_overlap < min_overlap:
                        no_urb += 1
                        continue
 
                # Guardar tile
                cx = (left + right) / 2
                cy = (top + bottom) / 2
 
                tile_name = f"tile_{fecha}_{row:03d}_{col:03d}.tif"
                tile_path = output_dir / tile_name
 
                profile = src.profile.copy()
                profile.update(
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
                    "source_image": imagen_path.name,
                    "fecha": fecha,
                    "row": row,
                    "col": col,
                    "center_x": round(cx, 2),
                    "center_y": round(cy, 2),
                    "valid_ratio": round(ratio_validos, 4),
                    "water_ratio": round(ratio_agua, 4),
                    "cloud_ratio": round(ratio_nubes, 4),
                    "urban_overlap": round(urban_overlap, 4),
                    "crs": str(src.crs),
                    "res_m": round(res_x, 2),
                    "n_bands": n_bandas,
                })
 
    return tiles_info, gen, inv, agua, nube, no_urb
 
 
if __name__ == "__main__":
    print("=" * 60)
    print("  SEGMENTACION EN TILES")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Tile:   {TILE_SIZE}x{TILE_SIZE} px")
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
 
    if not INPUT_DIR.exists():
        print(f"\n  ERROR: No se encontro {INPUT_DIR}")
        sys.exit(1)
 
    imagenes = sorted(INPUT_DIR.glob("*.TIF")) + sorted(INPUT_DIR.glob("*.tif"))
    if not imagenes:
        print(f"\n  No se encontraron imagenes en {INPUT_DIR}")
        sys.exit(1)
 
    print(f"\n  Imagenes encontradas: {len(imagenes)}")
 
    # Cargar poligono urbano una sola vez
    poligono = None
    if USE_URBAN:
        with rasterio.open(imagenes[0]) as src:
            crs_destino = src.crs
        poligono = cargar_poligono_urbano(LIMITES_PATH, crs_destino)
        if poligono is None:
            sys.exit(1)
 
    todos = []
    tot_gen = tot_inv = tot_agua = tot_nube = tot_no_urb = 0
 
    for i, img in enumerate(imagenes, 1):
        print(f"\n  [{i}/{len(imagenes)}] {img.name}")
        info, gen, inv, agua, nube, no_urb = procesar_imagen(
            img, OUTPUT_DIR, TILE_SIZE, MIN_VALID_RATIO,
            BAND_GREEN, BAND_NIR, NDWI_THRESHOLD, MAX_WATER_RATIO,
            USE_CLOUD, CLOUD_THRESHOLD, MAX_CLOUD_RATIO,
            USE_URBAN, poligono, MIN_URBAN_OVERLAP
        )
        todos.extend(info)
        tot_gen += gen
        tot_inv += inv
        tot_agua += agua
        tot_nube += nube
        tot_no_urb += no_urb
        print(f"    Validos:               {gen}")
        print(f"    Descartados borde:     {inv}")
        print(f"    Descartados agua:      {agua}")
        if USE_CLOUD:
            print(f"    Descartados nubes:     {nube}")
        if USE_URBAN:
            print(f"    Descartados no urbano: {no_urb}")
 
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
    print(f"  Imagenes procesadas:        {len(imagenes)}")
    print(f"  Tiles validos:              {tot_gen}")
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
        print(f"\n  Tiles por imagen:")
        for img, count in sorted(Counter(t["source_image"] for t in todos).items()):
            print(f"    {img}: {count}")
 
    print(f"\n  Tiles en:     {OUTPUT_DIR}")
    print(f"  Metadatos en: {OUTPUT_DIR / 'tiles_metadata.csv'}")
    print(f"\n  SIGUIENTE PASO: python scripts/04_etiquetar_tiles.py")
    print(f"{'='*60}")
 
