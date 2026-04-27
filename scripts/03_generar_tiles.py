#!/usr/bin/env python3
"""
03_generar_tiles.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Segmenta imagenes pansharpened en tiles de 512x512 pixeles, aplicando dos filtros:
  1. Pixeles validos: descarta tiles con muchos pixeles cero (bordes).
  2. Filtro de agua (NDWI): descarta tiles que sean mayoritariamente mar.

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
except ImportError:
    print("Error: rasterio no instalado.")
    print("  conda activate tesis")
    sys.exit(1)


# Cargar configuracion
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

INPUT_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["pansharpened"]
OUTPUT_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["tiles"]

TILE_SIZE = CFG["tiles"]["size"]
MIN_VALID_RATIO = CFG["tiles"]["min_valid_ratio"]
NDWI_THRESHOLD = CFG["tiles"]["ndwi_water_threshold"]
MAX_WATER_RATIO = CFG["tiles"]["max_water_ratio"]
BAND_GREEN = CFG["tiles"]["ndwi_band_green"]
BAND_NIR = CFG["tiles"]["ndwi_band_nir"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extraer_fecha(nombre):
    partes = nombre.replace(".TIF", "").replace(".tif", "").split("_")
    if len(partes) >= 3:
        return partes[2][:8]
    return "unknown"


def calcular_ratio_validos(data, nodata=0):
    """Ratio de pixeles que NO son nodata en todas las bandas."""
    if data.ndim == 3:
        mascara_nodata = np.all(data == nodata, axis=0)
    else:
        mascara_nodata = (data == nodata)
    return 1.0 - (mascara_nodata.sum() / mascara_nodata.size)


def calcular_ratio_agua(data, banda_verde_idx, banda_nir_idx, umbral_ndwi):
    """
    Calcula ratio de pixeles que son agua segun NDWI.
    NDWI = (Verde - NIR) / (Verde + NIR)
    Valores altos (> umbral) indican agua.
    Bandas: 1-indexed en el config, 0-indexed en numpy.
    """
    if data.ndim != 3 or data.shape[0] < max(banda_verde_idx, banda_nir_idx):
        return 0.0

    verde = data[banda_verde_idx - 1].astype(np.float32)
    nir = data[banda_nir_idx - 1].astype(np.float32)

    suma = verde + nir
    suma[suma == 0] = 1  # evitar division por cero
    ndwi = (verde - nir) / suma

    es_agua = ndwi > umbral_ndwi
    return es_agua.sum() / es_agua.size


def procesar_imagen(imagen_path, output_dir, tile_size, min_ratio,
                     band_green, band_nir, ndwi_thr, max_water):
    fecha = extraer_fecha(imagen_path.name)
    tiles_info = []
    generados = 0
    descartados_invalido = 0
    descartados_agua = 0

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
                window = Window(
                    col_off=col * tile_size,
                    row_off=row * tile_size,
                    width=tile_size,
                    height=tile_size,
                )

                data = src.read(window=window)

                # Filtro 1: validez (no bordes)
                ratio_validos = calcular_ratio_validos(data)
                if ratio_validos < min_ratio:
                    descartados_invalido += 1
                    continue

                # Filtro 2: agua (no mar)
                ratio_agua = calcular_ratio_agua(
                    data, band_green, band_nir, ndwi_thr
                )
                if ratio_agua > max_water:
                    descartados_agua += 1
                    continue

                # Pasa los filtros: guardar
                tile_transform = src.window_transform(window)
                cx = tile_transform.c + (tile_size / 2) * tile_transform.a
                cy = tile_transform.f + (tile_size / 2) * tile_transform.e

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

                generados += 1
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
                    "crs": str(src.crs),
                    "res_m": round(res_x, 2),
                    "n_bands": n_bandas,
                })

    return tiles_info, generados, descartados_invalido, descartados_agua


if __name__ == "__main__":

    print("=" * 60)
    print("  SEGMENTACION EN TILES")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Tile:   {TILE_SIZE}x{TILE_SIZE} px")
    print(f"  Min validos: {MIN_VALID_RATIO * 100:.0f}%")
    print(f"  Max agua:    {MAX_WATER_RATIO * 100:.0f}% (NDWI > {NDWI_THRESHOLD})")

    if not INPUT_DIR.exists():
        print(f"\n  ERROR: No se encontro {INPUT_DIR}")
        print(f"  Ejecuta primero: python scripts/02_pansharpening.py")
        sys.exit(1)

    imagenes = sorted(INPUT_DIR.glob("*.TIF")) + sorted(INPUT_DIR.glob("*.tif"))

    if not imagenes:
        print(f"\n  No se encontraron imagenes en {INPUT_DIR}")
        sys.exit(1)

    print(f"\n  Imagenes encontradas: {len(imagenes)}")

    todos = []
    tot_gen = tot_inv = tot_agua = 0

    for i, img in enumerate(imagenes, 1):
        print(f"\n  [{i}/{len(imagenes)}] {img.name}")
        info, gen, inv, agua = procesar_imagen(
            img, OUTPUT_DIR, TILE_SIZE, MIN_VALID_RATIO,
            BAND_GREEN, BAND_NIR, NDWI_THRESHOLD, MAX_WATER_RATIO
        )
        todos.extend(info)
        tot_gen += gen
        tot_inv += inv
        tot_agua += agua
        print(f"    Validos:           {gen}")
        print(f"    Descartados borde: {inv}")
        print(f"    Descartados agua:  {agua}")

    # Guardar metadatos
    if todos:
        meta_path = OUTPUT_DIR / "tiles_metadata.csv"
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=todos[0].keys())
            writer.writeheader()
            writer.writerows(todos)

    total = tot_gen + tot_inv + tot_agua
    tasa = tot_gen / max(1, total) * 100

    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"{'='*60}")
    print(f"  Imagenes procesadas:        {len(imagenes)}")
    print(f"  Tiles validos:              {tot_gen}")
    print(f"  Descartados (bordes):       {tot_inv}")
    print(f"  Descartados (agua/mar):     {tot_agua}")
    print(f"  Total potencial:            {total}")
    print(f"  Tasa aprovechamiento:       {tasa:.1f}%")

    if todos:
        res = todos[0]["res_m"]
        metros = TILE_SIZE * res
        print(f"\n  Tamano por tile:  {TILE_SIZE}x{TILE_SIZE} px")
        print(f"  Resolucion:       {res} m/px")
        print(f"  Cobertura:        ~{metros:.0f}x{metros:.0f} m")

    if todos:
        print(f"\n  Tiles por imagen:")
        for img, count in sorted(Counter(t["source_image"] for t in todos).items()):
            print(f"    {img}: {count}")

    print(f"\n  Tiles en:     {OUTPUT_DIR}")
    print(f"  Metadatos en: {OUTPUT_DIR / 'tiles_metadata.csv'}")
    print(f"\n  SIGUIENTE PASO: python scripts/04_etiquetar_tiles.py")
    print(f"{'='*60}")
