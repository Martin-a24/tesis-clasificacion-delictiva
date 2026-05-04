#!/usr/bin/env python3
"""
04_etiquetar_tiles.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Etiqueta cada tile con un nivel de riesgo (bajo, medio, alto) usando
umbrales globales calculados sobre una grilla virtual de toda Lima Metropolitana.

Soporta dos metodos via config.yaml:
  normalizar_poblacion: false  -> conteo bruto
  normalizar_poblacion: true   -> densidad normalizada por WorldPop
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box
except ImportError:
    print("conda activate tesis")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

TILES_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["tiles"]
DELITOS_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["delitos"]
LABELS_DIR = PROJECT_ROOT / CFG["paths"]["labels"]
LIMITES_PATH = PROJECT_ROOT / CFG["paths"]["raw"]["limites_urbanos"]

NIVELES = CFG["etiquetado"]["niveles"]
P_BAJO = CFG["etiquetado"]["percentil_bajo"]
P_ALTO = CFG["etiquetado"]["percentil_alto"]
NORMALIZAR = CFG["etiquetado"]["normalizar_poblacion"]
WORLDPOP_PATH = PROJECT_ROOT / CFG["etiquetado"]["worldpop_path"]
GRID_SIZE_M = CFG["etiquetado"].get("global_grid_tile_size_m", 358)

LABELS_DIR.mkdir(parents=True, exist_ok=True)


def construir_grilla_virtual(poligono, tile_size_m, crs):
    minx, miny, maxx, maxy = poligono.bounds
    print(f"  Construyendo grilla virtual de {tile_size_m}m sobre area urbana")

    tiles = []
    n_cols = int((maxx - minx) / tile_size_m) + 1
    n_rows = int((maxy - miny) / tile_size_m) + 1

    for row in range(n_rows):
        for col in range(n_cols):
            x0 = minx + col * tile_size_m
            y0 = miny + row * tile_size_m
            geom = box(x0, y0, x0 + tile_size_m, y0 + tile_size_m)
            if geom.intersects(poligono):
                if geom.intersection(poligono).area / geom.area >= 0.5:
                    tiles.append({
                        "virtual_id": f"v_{row:03d}_{col:03d}",
                        "row": row,
                        "col": col,
                        "geometry": geom
                    })

    print(f"  Grilla virtual final: {len(tiles)} tiles")
    return gpd.GeoDataFrame(tiles, crs=crs)


def calcular_densidades(grilla, delitos_gdf, normalizar, worldpop_path):
    """Calcula conteo y opcionalmente densidad normalizada."""
    id_col = "virtual_id" if "virtual_id" in grilla.columns else "tile_name"

    print("  Spatial join: delitos en tiles...")
    join = gpd.sjoin(delitos_gdf, grilla[[id_col, "geometry"]],
                      how="inner", predicate="within")
    conteos = join.groupby(id_col).size().reset_index(name="n_delitos")
    grilla = grilla.merge(conteos, on=id_col, how="left")
    grilla["n_delitos"] = grilla["n_delitos"].fillna(0).astype(int)

    n_con_delitos = (grilla["n_delitos"] > 0).sum()
    print(f"  Tiles con al menos 1 delito: {n_con_delitos} de {len(grilla)}")

    if normalizar and worldpop_path.exists():
        print(f"  Calculando poblacion (WorldPop)...")
        poblaciones = []
        with rasterio.open(worldpop_path) as src:
            wp_crs = src.crs
            grilla_proj = grilla.to_crs(wp_crs) if grilla.crs != wp_crs else grilla
            for _, row in grilla_proj.iterrows():
                try:
                    out_img, _ = rio_mask(src, [row.geometry], crop=True)
                    pop = float(np.nansum(out_img[out_img >= 0]))
                    poblaciones.append(pop)
                except Exception:
                    poblaciones.append(0.0)
        grilla["poblacion"] = poblaciones
        grilla["densidad"] = np.where(
            grilla["poblacion"] > 0,
            grilla["n_delitos"] / grilla["poblacion"] * 1000,
            0
        )
        print(f"  Metodo: densidad normalizada (delitos/mil hab)")
    else:
        grilla["poblacion"] = np.nan
        grilla["densidad"] = grilla["n_delitos"].astype(float)
        print(f"  Metodo: conteo bruto de delitos")

    return grilla


def calcular_umbrales(grilla, p_bajo, p_alto):
    densidades = grilla["densidad"].values
    no_cero = densidades[densidades > 0]
    if len(no_cero) == 0:
        return 0.0, 0.0
    return float(np.percentile(no_cero, p_bajo)), float(np.percentile(no_cero, p_alto))


def cargar_tiles_reales(tiles_dir):
    meta_path = tiles_dir / "tiles_metadata.csv"
    if not meta_path.exists():
        print(f"  ERROR: No existe {meta_path}")
        sys.exit(1)

    df = pd.read_csv(meta_path)
    print(f"  Tiles reales: {len(df)}")

    geoms = []
    crs_tiles = None
    for _, row in df.iterrows():
        tile_path = tiles_dir / row["tile_name"]
        if tile_path.exists():
            with rasterio.open(tile_path) as src:
                geoms.append(box(*src.bounds))
                if crs_tiles is None:
                    crs_tiles = src.crs
        else:
            geoms.append(None)

    df["geometry"] = geoms
    df = df[df["geometry"].notna()].copy()
    return gpd.GeoDataFrame(df, geometry="geometry", crs=crs_tiles)


def aplicar_umbrales(grilla, umbral_medio, umbral_alto, niveles):
    def clasificar(d):
        if d == 0 or d < umbral_medio:
            return niveles[0]
        elif d < umbral_alto:
            return niveles[1]
        else:
            return niveles[2]
    grilla["nivel_riesgo"] = grilla["densidad"].apply(clasificar)
    return grilla


def generar_reporte(tiles, grilla_virtual, umbral_medio, umbral_alto,
                     niveles, normalizar, output_path):
    distribucion = tiles["nivel_riesgo"].value_counts()
    total = len(tiles)
    metodo = "Densidad normalizada por poblacion (WorldPop)" if normalizar else "Conteo bruto de delitos"
    referencia = "Maharana et al. (2017)" if normalizar else "Almanie et al. (2018)"

    lines = []
    lines.append("=" * 60)
    lines.append("  REPORTE DE ETIQUETADO DE TILES")
    lines.append(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"METODO: {metodo}")
    lines.append(f"Referencia: {referencia}")
    lines.append("")
    lines.append("METODOLOGIA: Umbrales globales sobre grilla virtual de Lima")
    lines.append(f"  Grilla virtual total: {len(grilla_virtual)} tiles")
    lines.append(f"  Tiles virtuales con delitos: {(grilla_virtual['n_delitos'] > 0).sum()}")
    lines.append(f"  Tiles virtuales sin delitos: {(grilla_virtual['n_delitos'] == 0).sum()}")
    lines.append("")

    n_delitos_virt = grilla_virtual["n_delitos"].values
    n_no_cero = n_delitos_virt[n_delitos_virt > 0]
    if len(n_no_cero) > 0:
        lines.append("ESTADISTICAS DE DELITOS POR TILE VIRTUAL (con al menos 1 delito):")
        lines.append(f"  Min:    {n_no_cero.min():.0f}")
        lines.append(f"  Max:    {n_no_cero.max():.0f}")
        lines.append(f"  Media:  {n_no_cero.mean():.2f}")
        lines.append(f"  Mediana: {np.median(n_no_cero):.0f}")
        lines.append(f"  Percentil 25: {np.percentile(n_no_cero, 25):.0f}")
        lines.append(f"  Percentil 50: {np.percentile(n_no_cero, 50):.0f}")
        lines.append(f"  Percentil 75: {np.percentile(n_no_cero, 75):.0f}")
        lines.append(f"  Percentil 95: {np.percentile(n_no_cero, 95):.0f}")
        lines.append(f"  Percentil 99: {np.percentile(n_no_cero, 99):.0f}")
        lines.append("")

    lines.append("UMBRALES APLICADOS:")
    if normalizar:
        lines.append(f"  Unidades: delitos por cada 1000 habitantes")
    else:
        lines.append(f"  Unidades: numero absoluto de delitos por tile")
    lines.append(f"  bajo:  densidad < {umbral_medio:.4f}")
    lines.append(f"  medio: {umbral_medio:.4f} <= densidad < {umbral_alto:.4f}")
    lines.append(f"  alto:  densidad >= {umbral_alto:.4f}")
    lines.append("")
    lines.append(f"TILES REALES ETIQUETADOS: {total}")
    lines.append("")
    lines.append("Distribucion en tiles reales:")
    for nivel in niveles:
        n = distribucion.get(nivel, 0)
        pct = n / total * 100 if total else 0
        lines.append(f"  {nivel:<10}: {n:>5} ({pct:.1f}%)")
    lines.append("")
    lines.append("Estadisticas de delitos por tile real:")
    lines.append(f"  Min:    {tiles['n_delitos'].min()}")
    lines.append(f"  Max:    {tiles['n_delitos'].max()}")
    lines.append(f"  Media:  {tiles['n_delitos'].mean():.2f}")
    lines.append(f"  Mediana: {tiles['n_delitos'].median():.0f}")
    lines.append(f"  Tiles sin delitos: {(tiles['n_delitos'] == 0).sum()}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)


def generar_grafico(tiles, grilla_virtual, umbral_medio, umbral_alto,
                     normalizar, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titulo_unidad = "Densidad (delitos/mil hab)" if normalizar else "Numero de delitos"
    metodo_str = "normalizado por poblacion" if normalizar else "conteo bruto"

    densidades_virt = grilla_virtual["densidad"].values
    densidades_no_cero = densidades_virt[densidades_virt > 0]

    axes[0].hist(densidades_no_cero, bins=50, color="#2c3e50", alpha=0.7, edgecolor="white")
    axes[0].axvline(umbral_medio, color="orange", linestyle="--",
                     label=f"P{P_BAJO} = {umbral_medio:.2f}")
    axes[0].axvline(umbral_alto, color="red", linestyle="--",
                     label=f"P{P_ALTO} = {umbral_alto:.2f}")
    axes[0].set_xlabel(titulo_unidad)
    axes[0].set_ylabel("N° de tiles virtuales")
    axes[0].set_title(f"Distribucion GLOBAL en Lima ({metodo_str})\n{len(densidades_no_cero):,} tiles con delitos")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if len(densidades_no_cero) > 0:
        p95 = np.percentile(densidades_no_cero, 95)
        densidades_zoom = densidades_no_cero[densidades_no_cero <= p95]
        axes[1].hist(densidades_zoom, bins=30, color="#2c3e50", alpha=0.7, edgecolor="white")
        axes[1].axvline(umbral_medio, color="orange", linestyle="--",
                         label=f"P{P_BAJO} = {umbral_medio:.2f}")
        axes[1].axvline(umbral_alto, color="red", linestyle="--",
                         label=f"P{P_ALTO} = {umbral_alto:.2f}")
        axes[1].set_xlabel(titulo_unidad)
        axes[1].set_ylabel("N° de tiles virtuales")
        axes[1].set_title(f"Zoom: percentil 0-95\n(excluye outliers)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    densidades_reales = tiles["densidad"].values
    densidades_reales_no_cero = densidades_reales[densidades_reales > 0]
    if len(densidades_reales_no_cero) > 0:
        axes[2].hist(densidades_reales_no_cero, bins=20, color="#3498db",
                      alpha=0.7, edgecolor="white")
    axes[2].axvline(umbral_medio, color="orange", linestyle="--",
                     label=f"P{P_BAJO} = {umbral_medio:.2f}")
    axes[2].axvline(umbral_alto, color="red", linestyle="--",
                     label=f"P{P_ALTO} = {umbral_alto:.2f}")
    axes[2].set_xlabel(titulo_unidad)
    axes[2].set_ylabel("N° de tiles reales")
    axes[2].set_title(f"Distribucion en tiles reales (con imagen)\n{len(tiles)} tiles totales")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    metodo_str = "DENSIDAD NORMALIZADA POR POBLACION" if NORMALIZAR else "CONTEO BRUTO DE DELITOS"

    print("=" * 60)
    print("  ETIQUETADO DE TILES POR NIVEL DE RIESGO")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Metodo: {metodo_str}")
    print("=" * 60)

    if not LIMITES_PATH.exists():
        print(f"\n  ERROR: No existe {LIMITES_PATH}")
        sys.exit(1)

    if NORMALIZAR and not WORLDPOP_PATH.exists():
        print(f"\n  ERROR: WorldPop no encontrado en {WORLDPOP_PATH}")
        print(f"  Descarga WorldPop o desactiva normalizar_poblacion en config.yaml")
        sys.exit(1)

    print(f"\n  [1/5] Cargando tiles reales...")
    tiles_reales = cargar_tiles_reales(TILES_DIR)

    print(f"\n  [2/5] Cargando delitos...")
    delitos_path = DELITOS_DIR / "delitos_lima_limpio.geojson"
    delitos = gpd.read_file(delitos_path)
    print(f"  Delitos: {len(delitos)}")
    delitos = delitos.to_crs(tiles_reales.crs)

    print(f"\n  [3/5] Construyendo grilla virtual y calculando umbrales globales...")
    gdf_limites = gpd.read_file(LIMITES_PATH)
    if gdf_limites.crs != tiles_reales.crs:
        gdf_limites = gdf_limites.to_crs(tiles_reales.crs)
    poligono_urbano = gdf_limites.geometry.unary_union

    grilla_virtual = construir_grilla_virtual(poligono_urbano, GRID_SIZE_M, tiles_reales.crs)
    grilla_virtual = calcular_densidades(grilla_virtual, delitos, NORMALIZAR, WORLDPOP_PATH)
    umbral_medio, umbral_alto = calcular_umbrales(grilla_virtual, P_BAJO, P_ALTO)

    print(f"\n  Umbrales calculados:")
    if NORMALIZAR:
        print(f"    Percentil {P_BAJO}: {umbral_medio:.4f} delitos/mil hab")
        print(f"    Percentil {P_ALTO}: {umbral_alto:.4f} delitos/mil hab")
    else:
        print(f"    Percentil {P_BAJO}: {umbral_medio:.2f} delitos")
        print(f"    Percentil {P_ALTO}: {umbral_alto:.2f} delitos")

    print(f"\n  [4/5] Etiquetando tiles reales...")
    tiles_reales = calcular_densidades(tiles_reales, delitos, NORMALIZAR, WORLDPOP_PATH)
    tiles_reales = aplicar_umbrales(tiles_reales, umbral_medio, umbral_alto, NIVELES)

    print(f"\n  [5/5] Guardando resultados...")

    cols_csv = ["tile_name", "source_image", "fecha", "row", "col",
                 "center_x", "center_y", "n_delitos", "poblacion",
                 "densidad", "nivel_riesgo"]
    csv_path = LABELS_DIR / "tiles_labeled.csv"
    tiles_reales[cols_csv].to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    geojson_path = LABELS_DIR / "tiles_labeled.geojson"
    tiles_reales[cols_csv + ["geometry"]].to_file(geojson_path, driver="GeoJSON")
    print(f"  GeoJSON: {geojson_path}")

    umbrales_path = LABELS_DIR / "umbrales_globales.json"
    with open(umbrales_path, "w") as f:
        json.dump({
            "fecha": datetime.now().isoformat(),
            "metodo": "densidad_normalizada" if NORMALIZAR else "conteo_bruto",
            "referencia": "Maharana et al. (2017)" if NORMALIZAR else "Almanie et al. (2018)",
            "percentil_bajo": P_BAJO,
            "percentil_alto": P_ALTO,
            "umbral_medio": umbral_medio,
            "umbral_alto": umbral_alto,
            "normalizar_poblacion": NORMALIZAR,
            "n_tiles_virtuales": len(grilla_virtual),
            "n_tiles_virtuales_con_delitos": int((grilla_virtual["n_delitos"] > 0).sum()),
            "n_tiles_reales": len(tiles_reales),
        }, f, indent=2)
    print(f"  Umbrales: {umbrales_path}")

    reporte_path = LABELS_DIR / "etiquetado_reporte.txt"
    print()
    generar_reporte(tiles_reales, grilla_virtual, umbral_medio, umbral_alto,
                    NIVELES, NORMALIZAR, reporte_path)

    grafico_path = LABELS_DIR / "distribucion_densidad.png"
    generar_grafico(tiles_reales, grilla_virtual, umbral_medio, umbral_alto,
                     NORMALIZAR, grafico_path)
    print(f"\n  Grafico: {grafico_path}")

    print(f"\n  SIGUIENTE PASO: Validar en QGIS ({geojson_path})")
    print(f"{'='*60}")