#!/usr/bin/env python3
"""
02b_construir_mosaico.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Construye un mosaico virtual (VRT) a partir de todas las escenas pansharpened.
El VRT no copia datos (es barato y rapido de reconstruir cuando llegan escenas
nuevas) y hace que cada ubicacion fisica exista una sola vez: en las zonas de
solape entre escenas no se generan tiles duplicados.

Detalles:
  - nodata = 0 para que los bordes negros rotados de una escena no tapen los
    pixeles validos de la escena vecina en el solape.
  - prioridad "auto": ordena las escenas por % de pixeles validos ascendente,
    de modo que la escena con mas pixeles validos quede "arriba" en el solape.
  - Verifica que todas las escenas compartan el CRS configurado (EPSG:32718).

Entrada:
    data/processed/imagenes_pansharpened/*.TIF

Salida:
    data/processed/mosaico/mosaico.vrt

Uso:
    python scripts/02b_construir_mosaico.py
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    import rasterio
    from osgeo import gdal
except ImportError:
    print("Error: librerias no instaladas. Activa el environment:")
    print("  conda activate tesis")
    sys.exit(1)

gdal.UseExceptions()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

INPUT_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["pansharpened"]
OUTPUT_VRT = PROJECT_ROOT / CFG["mosaico"]["output"]

CRS_ESPERADO = CFG["mosaico"]["crs"]
SRC_NODATA = CFG["mosaico"]["src_nodata"]
VRT_NODATA = CFG["mosaico"]["vrt_nodata"]
PRIORIDAD = CFG["mosaico"].get("prioridad", "auto")


def estimar_valid_ratio(path, max_lado=2048):
    """Estima el % de pixeles validos (no nodata) leyendo a resolucion reducida."""
    with rasterio.open(path) as src:
        escala = max(1, int(max(src.width, src.height) / max_lado))
        out_h = max(1, src.height // escala)
        out_w = max(1, src.width // escala)
        data = src.read(out_shape=(src.count, out_h, out_w))
    mascara_nodata = np.all(data == SRC_NODATA, axis=0)
    return 1.0 - (mascara_nodata.sum() / mascara_nodata.size)


def verificar_crs(escenas):
    """Comprueba que todas las escenas comparten el CRS esperado."""
    problemas = []
    for p in escenas:
        with rasterio.open(p) as src:
            crs = src.crs
        if crs is None or crs.to_string() != CRS_ESPERADO:
            problemas.append((p.name, str(crs)))
    return problemas


if __name__ == "__main__":
    print("=" * 60)
    print("  CONSTRUCCION DE MOSAICO VIRTUAL (VRT)")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_VRT}")
    print(f"  CRS esperado: {CRS_ESPERADO}")
    print(f"  Prioridad en solape: {PRIORIDAD}")

    if not INPUT_DIR.exists():
        print(f"\n  ERROR: No existe {INPUT_DIR}")
        print(f"  Ejecuta primero: python scripts/02_pansharpening.py")
        sys.exit(1)

    escenas = sorted(INPUT_DIR.glob("*.TIF")) + sorted(INPUT_DIR.glob("*.tif"))
    if not escenas:
        print(f"\n  ERROR: No se encontraron escenas pansharpened en {INPUT_DIR}")
        print(f"  Ejecuta primero: python scripts/02_pansharpening.py")
        sys.exit(1)

    print(f"\n  Escenas pansharpened encontradas: {len(escenas)}")

    # Verificar CRS comun
    problemas = verificar_crs(escenas)
    if problemas:
        print("\n  ERROR: hay escenas con CRS distinto al esperado:")
        for nombre, crs in problemas:
            print(f"    - {nombre}: {crs}")
        print(f"\n  Reproyecta a {CRS_ESPERADO} con gdalwarp antes de construir el mosaico.")
        sys.exit(1)

    # Ordenar escenas: la ultima de la lista queda "arriba" en el solape.
    if PRIORIDAD == "auto":
        print("\n  Calculando % de pixeles validos por escena (orden 'auto')...")
        ratios = {}
        for p in escenas:
            vr = estimar_valid_ratio(p)
            ratios[p] = vr
            print(f"    {p.name}: {vr * 100:.1f}% validos")
        # ascendente: menor valid_ratio primero (abajo), mayor al final (arriba)
        escenas_ordenadas = sorted(escenas, key=lambda p: ratios[p])
    else:
        escenas_ordenadas = escenas

    print("\n  Orden de apilado (de abajo hacia arriba en el solape):")
    for i, p in enumerate(escenas_ordenadas, 1):
        print(f"    {i}. {p.name}")

    OUTPUT_VRT.parent.mkdir(parents=True, exist_ok=True)

    print("\n  Construyendo VRT...")
    vrt = gdal.BuildVRT(
        str(OUTPUT_VRT),
        [str(p) for p in escenas_ordenadas],
        options=gdal.BuildVRTOptions(
            srcNodata=SRC_NODATA,
            VRTNodata=VRT_NODATA,
            resolution="highest",
        ),
    )
    if vrt is None:
        print("  ERROR: gdal.BuildVRT no pudo construir el mosaico.")
        sys.exit(1)
    vrt.FlushCache()
    vrt = None  # cierra el dataset y escribe el .vrt a disco

    # Reportar dimensiones del mosaico resultante
    with rasterio.open(OUTPUT_VRT) as src:
        ancho, alto = src.width, src.height
        res_x, _ = src.res
        bounds = src.bounds
        n_bandas = src.count

    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Escenas unidas:   {len(escenas_ordenadas)}")
    print(f"  Dimensiones VRT:  {ancho}x{alto} px, {n_bandas} bandas, {res_x:.2f} m/px")
    print(f"  Extension (m):    X[{bounds.left:.0f}, {bounds.right:.0f}] "
          f"Y[{bounds.bottom:.0f}, {bounds.top:.0f}]")
    print(f"  nodata:           {VRT_NODATA}")
    print(f"  VRT en:           {OUTPUT_VRT}")
    print("\n  Verifica en QGIS que en el solape no haya bordes negros tapando datos.")
    print(f"\n  SIGUIENTE PASO: python scripts/03_generar_tiles.py")
    print("=" * 60)
