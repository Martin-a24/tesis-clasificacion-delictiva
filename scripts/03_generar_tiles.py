#!/usr/bin/env python3


import os
import csv
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    print("Error: rasterio no instalado.")
    print("  pip install rasterio numpy")
    sys.exit(1)


DEFAULT_INPUT = Path.home() / "tesis_output" / "pansharpened"
DEFAULT_OUTPUT = Path.home() / "tesis_output" / "tiles"
TILE_SIZE = 512
MIN_VALID_RATIO = 0.95


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentar imagenes en tiles de 512x512 px"
    )
    parser.add_argument("-i", "--input", type=str, default=str(DEFAULT_INPUT),
                        help="Carpeta con imagenes pansharpened (.TIF)")
    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Carpeta de salida para los tiles")
    parser.add_argument("-t", "--tile-size", type=int, default=TILE_SIZE,
                        help="Tamano del tile en pixeles (default: 512)")
    parser.add_argument("-m", "--min-valid", type=float, default=MIN_VALID_RATIO,
                        help="Ratio minimo de pixeles validos (default: 0.95)")
    return parser.parse_args()


def extraer_fecha(nombre_archivo):
    partes = nombre_archivo.replace(".TIF", "").replace(".tif", "").split("_")
    if len(partes) >= 3:
        return partes[2][:8]
    return "unknown"


def tile_es_valido(data, min_ratio, nodata_value=0):
    if data.ndim == 3:
        mascara_cero = np.all(data == nodata_value, axis=0)
    else:
        mascara_cero = (data == nodata_value)
    ratio_valido = 1.0 - (mascara_cero.sum() / mascara_cero.size)
    return ratio_valido >= min_ratio, ratio_valido


def procesar_imagen(imagen_path, output_dir, tile_size, min_ratio):
    fecha = extraer_fecha(imagen_path.name)
    tiles_info = []
    tiles_generados = 0
    tiles_descartados = 0

    with rasterio.open(imagen_path) as src:
        n_bandas = src.count
        ancho = src.width
        alto = src.height
        res_x, res_y = src.res

        print(f"    Dimensiones: {ancho} x {alto} px, {n_bandas} bandas")
        print(f"    Resolucion: {res_x:.2f} m/px")

        n_cols = ancho // tile_size
        n_rows = alto // tile_size

        print(f"    Grilla: {n_cols} x {n_rows} = {n_cols * n_rows} tiles potenciales")

        for row in range(n_rows):
            for col in range(n_cols):
                window = Window(
                    col_off=col * tile_size,
                    row_off=row * tile_size,
                    width=tile_size,
                    height=tile_size
                )

                data = src.read(window=window)
                es_valido, ratio = tile_es_valido(data, min_ratio)

                if not es_valido:
                    tiles_descartados += 1
                    continue

                tile_transform = src.window_transform(window)
                center_x = tile_transform.c + (tile_size / 2) * tile_transform.a
                center_y = tile_transform.f + (tile_size / 2) * tile_transform.e

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

                tiles_generados += 1
                tiles_info.append({
                    "tile_name": tile_name,
                    "source_image": imagen_path.name,
                    "fecha": fecha,
                    "row": row,
                    "col": col,
                    "center_x": round(center_x, 2),
                    "center_y": round(center_y, 2),
                    "valid_ratio": round(ratio, 4),
                    "crs": str(src.crs),
                    "res_m": round(res_x, 2),
                    "n_bands": n_bandas,
                })

    return tiles_info, tiles_generados, tiles_descartados


if __name__ == "__main__":

    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    tile_size = args.tile_size
    min_ratio = args.min_valid

    print("=" * 60)
    print("  SEGMENTACION EN TILES")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  Input:      {input_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Tile size:  {tile_size} x {tile_size} px")
    print(f"  Min valid:  {min_ratio * 100:.0f}%")

    if not input_dir.exists():
        print(f"\n  ERROR: No se encontro {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Buscar imagenes TIF
    imagenes = sorted(
        list(input_dir.glob("*.TIF")) + list(input_dir.glob("*.tif"))
    )

    if not imagenes:
        print(f"\n  No se encontraron imagenes en {input_dir}")
        sys.exit(1)

    print(f"\n  Imagenes encontradas: {len(imagenes)}")
    for img in imagenes:
        size_mb = img.stat().st_size / (1024 * 1024)
        print(f"    {img.name} ({size_mb:.0f} MB)")

    # Procesar
    todos_metadatos = []
    total_generados = 0
    total_descartados = 0

    for i, img_path in enumerate(imagenes, 1):
        print(f"\n  [{i}/{len(imagenes)}] {img_path.name}")
        info, generados, descartados = procesar_imagen(
            img_path, output_dir, tile_size, min_ratio
        )
        todos_metadatos.extend(info)
        total_generados += generados
        total_descartados += descartados
        print(f"    Tiles validos:      {generados}")
        print(f"    Tiles descartados:  {descartados}")

    # Guardar metadatos CSV
    metadata_path = output_dir / "tiles_metadata.csv"
    if todos_metadatos:
        fieldnames = todos_metadatos[0].keys()
        with open(metadata_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(todos_metadatos)

    # Resumen
    total_potencial = total_generados + total_descartados
    tasa = total_generados / max(1, total_potencial) * 100

    print(f"\n{'=' * 60}")
    print(f"  RESUMEN")
    print(f"{'=' * 60}")
    print(f"  Imagenes procesadas:     {len(imagenes)}")
    print(f"  Tiles generados:         {total_generados}")
    print(f"  Tiles descartados:       {total_descartados}")
    print(f"  Total potencial:         {total_potencial}")
    print(f"  Tasa de aprovechamiento: {tasa:.1f}%")

    if todos_metadatos and todos_metadatos[0]["res_m"] > 0:
        res = todos_metadatos[0]["res_m"]
        metros = tile_size * res
        print(f"  Tamano por tile:   {tile_size}x{tile_size} px")
        print(f"  Resolucion:        {res} m/px")
        print(f"  Area por tile:     ~{metros:.0f} x {metros:.0f} m")

    print(f"  Tiles en:      {output_dir}")
    print(f"  Metadatos en:  {metadata_path}")

    if todos_metadatos:
        from collections import Counter
        conteo = Counter(m["source_image"] for m in todos_metadatos)
        print(f"\n  Tiles por imagen fuente:")
        for img, count in sorted(conteo.items()):
            print(f"    {img}: {count}")

    print(f"\n  SIGUIENTE PASO: python3 04_etiquetar_tiles.py")
    print(f"{'=' * 60}")