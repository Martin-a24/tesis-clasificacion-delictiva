#!/usr/bin/env python3
"""
04_etiquetar_tiles.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Etiqueta cada tile con un nivel de riesgo (bajo, medio, alto) basado en
la densidad de delitos que ocurren dentro de su area.

Pipeline:
  1. Carga tiles_metadata.csv y reconstruye geometrias de los tiles.
  2. Carga delitos_lima_limpio.geojson.
  3. Reproyecta delitos al CRS de los tiles.
  4. Realiza spatial join: cuenta delitos por tile.
  5. (Opcional) Normaliza por poblacion usando WorldPop.
  6. Asigna categorias por percentiles.
  7. Guarda tiles_labeled.csv con la etiqueta por tile.

Entrada:
    data/processed/tiles/tiles_metadata.csv
    data/processed/tiles/*.tif
    data/processed/delitos_limpios/delitos_lima_limpio.geojson

Salida:
    data/labels/tiles_labeled.csv
    data/labels/tiles_labeled.geojson      (para visualizar en QGIS)
    data/labels/etiquetado_reporte.txt
    data/labels/distribucion_densidad.png

Uso:
    python scripts/04_etiquetar_tiles.py
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box
except ImportError:
    print("Error: dependencias geoespaciales no instaladas.")
    print("  conda activate tesis")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Cargar configuracion
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

TILES_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["tiles"]
DELITOS_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["delitos"]
LABELS_DIR = PROJECT_ROOT / CFG["paths"]["labels"]

NIVELES = CFG["etiquetado"]["niveles"]
P_BAJO = CFG["etiquetado"]["percentil_bajo"]
P_ALTO = CFG["etiquetado"]["percentil_alto"]
NORMALIZAR = CFG["etiquetado"]["normalizar_poblacion"]
WORLDPOP_PATH = PROJECT_ROOT / CFG["etiquetado"]["worldpop_path"]

LABELS_DIR.mkdir(parents=True, exist_ok=True)


def cargar_tiles_geo():
    """
    Carga tiles_metadata.csv y construye GeoDataFrame con la geometria de cada tile.
    """
    meta_path = TILES_DIR / "tiles_metadata.csv"
    if not meta_path.exists():
        print(f"  ERROR: No existe {meta_path}")
        print(f"  Ejecuta primero: python scripts/03_generar_tiles.py")
        sys.exit(1)

    df = pd.read_csv(meta_path)
    print(f"  Tiles cargados desde metadata: {len(df)}")

    # Reconstruir geometria leyendo cada tile (asegura precision)
    geoms = []
    crs_tiles = None

    for _, row in df.iterrows():
        tile_path = TILES_DIR / row["tile_name"]
        if not tile_path.exists():
            geoms.append(None)
            continue
        with rasterio.open(tile_path) as src:
            geoms.append(box(*src.bounds))
            if crs_tiles is None:
                crs_tiles = src.crs

    df["geometry"] = geoms
    df = df[df["geometry"].notna()].copy()
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs_tiles)
    print(f"  Geometrias construidas: {len(gdf)}")
    print(f"  CRS de tiles: {crs_tiles}")
    return gdf


def cargar_delitos(crs_destino):
    """Carga delitos y reproyecta al CRS de los tiles."""
    delitos_path = DELITOS_DIR / "delitos_lima_limpio.geojson"
    if not delitos_path.exists():
        print(f"  ERROR: No existe {delitos_path}")
        sys.exit(1)

    delitos = gpd.read_file(delitos_path)
    print(f"  Delitos cargados: {len(delitos)} (CRS original: {delitos.crs})")

    delitos = delitos.to_crs(crs_destino)
    print(f"  Reproyectados a: {delitos.crs}")

    return delitos


def contar_delitos_por_tile(tiles_gdf, delitos_gdf):
    """
    Spatial join: cuenta cuantos delitos caen en cada tile.
    """
    # Hacer el join: cada delito recibe el indice del tile que lo contiene
    join = gpd.sjoin(delitos_gdf, tiles_gdf[["tile_name", "geometry"]],
                      how="inner", predicate="within")

    # Contar delitos por tile
    conteos = join.groupby("tile_name").size().reset_index(name="n_delitos")

    # Merge con todos los tiles (los que no tienen delitos quedan en 0)
    tiles_con_conteo = tiles_gdf.merge(conteos, on="tile_name", how="left")
    tiles_con_conteo["n_delitos"] = tiles_con_conteo["n_delitos"].fillna(0).astype(int)

    return tiles_con_conteo


def estimar_poblacion(tiles_gdf, worldpop_path):
    """
    Para cada tile, suma la poblacion estimada de los pixeles WorldPop dentro del tile.
    Si no hay WorldPop disponible, retorna None (usa conteo bruto).
    """
    if not worldpop_path.exists():
        print(f"  WorldPop no encontrado en {worldpop_path}")
        print(f"  Usando conteo bruto sin normalizacion por poblacion.")
        return None

    print(f"  Cargando WorldPop: {worldpop_path}")

    poblaciones = []
    with rasterio.open(worldpop_path) as src:
        wp_crs = src.crs
        # Reproyectar tiles al CRS de WorldPop si es diferente
        if tiles_gdf.crs != wp_crs:
            tiles_proj = tiles_gdf.to_crs(wp_crs)
        else:
            tiles_proj = tiles_gdf

        for _, row in tiles_proj.iterrows():
            try:
                from rasterio.mask import mask
                out_image, _ = mask(src, [row.geometry], crop=True)
                pop = float(np.nansum(out_image[out_image >= 0]))
                poblaciones.append(pop)
            except Exception:
                poblaciones.append(0.0)

    return poblaciones


def calcular_densidad(tiles_gdf, poblaciones=None):
    """
    Calcula densidad delictiva.
    Si hay poblaciones: delitos / poblacion * 1000 (delitos por mil habitantes).
    Si no: usa conteo bruto.
    """
    if poblaciones is not None:
        tiles_gdf["poblacion"] = poblaciones
        # Evitar division por cero: tiles sin poblacion -> densidad = 0
        # (asumimos que zonas sin poblacion no tienen riesgo medible)
        tiles_gdf["densidad"] = np.where(
            tiles_gdf["poblacion"] > 0,
            tiles_gdf["n_delitos"] / tiles_gdf["poblacion"] * 1000,
            0
        )
        print(f"  Densidad calculada (delitos por mil habitantes)")
    else:
        tiles_gdf["poblacion"] = np.nan
        tiles_gdf["densidad"] = tiles_gdf["n_delitos"].astype(float)
        print(f"  Densidad calculada (conteo bruto de delitos)")

    return tiles_gdf


def asignar_etiquetas(tiles_gdf, niveles, p_bajo, p_alto):
    """
    Asigna nivel de riesgo por percentiles.
    Tiles con densidad 0 van a "bajo" automaticamente.
    """
    densidades = tiles_gdf["densidad"].values

    # Calcular umbrales solo con tiles que tienen actividad delictiva
    densidades_no_cero = densidades[densidades > 0]

    if len(densidades_no_cero) < 10:
        print(f"  ADVERTENCIA: Solo {len(densidades_no_cero)} tiles con delitos.")
        print(f"  Resultado puede no ser representativo.")

    if len(densidades_no_cero) > 0:
        umbral_medio = np.percentile(densidades_no_cero, p_bajo)
        umbral_alto = np.percentile(densidades_no_cero, p_alto)
    else:
        umbral_medio = 0
        umbral_alto = 0

    print(f"\n  Umbrales calculados (sobre tiles con delitos):")
    print(f"    Percentil {p_bajo}: {umbral_medio:.4f}")
    print(f"    Percentil {p_alto}: {umbral_alto:.4f}")

    def clasificar(d):
        if d == 0:
            return niveles[0]  # bajo
        elif d < umbral_medio:
            return niveles[0]  # bajo
        elif d < umbral_alto:
            return niveles[1]  # medio
        else:
            return niveles[2]  # alto

    tiles_gdf["nivel_riesgo"] = tiles_gdf["densidad"].apply(clasificar)

    return tiles_gdf, umbral_medio, umbral_alto


def generar_reporte(tiles_gdf, umbral_medio, umbral_alto, niveles, output_path):
    """Genera reporte de texto del etiquetado."""
    distribucion = tiles_gdf["nivel_riesgo"].value_counts()
    total = len(tiles_gdf)

    lines = []
    lines.append("=" * 60)
    lines.append("  REPORTE DE ETIQUETADO DE TILES")
    lines.append(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total de tiles etiquetados: {total}")
    lines.append("")
    lines.append("Distribucion por nivel de riesgo:")
    for nivel in niveles:
        n = distribucion.get(nivel, 0)
        pct = n / total * 100 if total else 0
        lines.append(f"  {nivel:<10}: {n:>5} ({pct:.1f}%)")
    lines.append("")
    lines.append("Umbrales utilizados (sobre densidades > 0):")
    lines.append(f"  bajo:  densidad < {umbral_medio:.4f}")
    lines.append(f"  medio: {umbral_medio:.4f} <= densidad < {umbral_alto:.4f}")
    lines.append(f"  alto:  densidad >= {umbral_alto:.4f}")
    lines.append("")
    lines.append("Estadisticas de delitos por tile:")
    lines.append(f"  Min:    {tiles_gdf['n_delitos'].min()}")
    lines.append(f"  Max:    {tiles_gdf['n_delitos'].max()}")
    lines.append(f"  Media:  {tiles_gdf['n_delitos'].mean():.2f}")
    lines.append(f"  Mediana: {tiles_gdf['n_delitos'].median()}")
    lines.append(f"  Tiles sin delitos: {(tiles_gdf['n_delitos'] == 0).sum()}")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)


def generar_grafico(tiles_gdf, umbral_medio, umbral_alto, output_path):
    """Genera histograma de distribucion de densidad."""
    fig, ax = plt.subplots(figsize=(10, 5))

    densidades = tiles_gdf["densidad"].values
    densidades_plot = densidades[densidades > 0]  # excluir ceros para visualizacion

    ax.hist(densidades_plot, bins=30, color="#3498db", edgecolor="white")
    ax.axvline(umbral_medio, color="orange", linestyle="--",
                label=f"P{P_BAJO} = {umbral_medio:.3f}")
    ax.axvline(umbral_alto, color="red", linestyle="--",
                label=f"P{P_ALTO} = {umbral_alto:.3f}")

    ax.set_xlabel("Densidad delictiva (delitos por tile o por mil hab.)")
    ax.set_ylabel("Numero de tiles")
    ax.set_title("Distribucion de densidad delictiva (tiles con al menos un delito)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  ETIQUETADO DE TILES POR NIVEL DE RIESGO DELICTIVO")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    # 1. Cargar tiles
    print(f"\n  [1/5] Cargando tiles desde {TILES_DIR}")
    tiles = cargar_tiles_geo()

    # 2. Cargar delitos y reproyectar
    print(f"\n  [2/5] Cargando delitos desde {DELITOS_DIR}")
    delitos = cargar_delitos(tiles.crs)

    # 3. Spatial join
    print(f"\n  [3/5] Calculando delitos por tile (spatial join)")
    tiles = contar_delitos_por_tile(tiles, delitos)
    n_con_delitos = (tiles["n_delitos"] > 0).sum()
    print(f"  Tiles con al menos 1 delito: {n_con_delitos} de {len(tiles)}")

    # 4. Densidad (con o sin normalizacion)
    print(f"\n  [4/5] Calculando densidad delictiva")
    if NORMALIZAR:
        poblaciones = estimar_poblacion(tiles, WORLDPOP_PATH)
    else:
        poblaciones = None
        print(f"  Normalizacion por poblacion: DESACTIVADA (config.yaml)")
    tiles = calcular_densidad(tiles, poblaciones)

    # 5. Etiquetar
    print(f"\n  [5/5] Asignando niveles de riesgo por percentiles")
    tiles, u_medio, u_alto = asignar_etiquetas(tiles, NIVELES, P_BAJO, P_ALTO)

    # Guardar resultados
    print(f"\n{'='*60}")
    print(f"  GUARDANDO RESULTADOS")
    print(f"{'='*60}")

    # CSV
    cols_csv = [
        "tile_name", "source_image", "fecha", "row", "col",
        "center_x", "center_y", "n_delitos", "poblacion",
        "densidad", "nivel_riesgo"
    ]
    csv_path = LABELS_DIR / "tiles_labeled.csv"
    tiles[cols_csv].to_csv(csv_path, index=False)
    print(f"  CSV:     {csv_path}")

    # GeoJSON para QGIS
    geojson_path = LABELS_DIR / "tiles_labeled.geojson"
    tiles[cols_csv + ["geometry"]].to_file(geojson_path, driver="GeoJSON")
    print(f"  GeoJSON: {geojson_path}")

    # Reporte
    reporte_path = LABELS_DIR / "etiquetado_reporte.txt"
    print(f"\n  Reporte: {reporte_path}\n")
    generar_reporte(tiles, u_medio, u_alto, NIVELES, reporte_path)

    # Grafico
    grafico_path = LABELS_DIR / "distribucion_densidad.png"
    generar_grafico(tiles, u_medio, u_alto, grafico_path)
    print(f"\n  Grafico: {grafico_path}")

    print(f"\n  SIGUIENTE PASO: Validar visualmente en QGIS")
    print(f"  Abre {geojson_path} y simboliza por nivel_riesgo")
    print(f"{'='*60}")
