#!/usr/bin/env python3
"""
01_limpiar_datos_delictivos.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Consolida y filtra registros delictivos descargados del MININTER.
Lee CSVs de la carpeta data/raw/delitos/ y produce CSV + GeoJSON
en data/processed/delitos_limpios/.

Uso:
    python scripts/01_limpiar_datos_delictivos.py

Configuracion: configs/config.yaml
"""

import sys
import yaml
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# Cargar configuracion
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

RAW_DIR = PROJECT_ROOT / CFG["paths"]["raw"]["delitos"]
OUTPUT_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["delitos"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BBOX = CFG["delitos"]["bbox_lima"]
UBICACIONES = [tuple(u) for u in CFG["delitos"]["ubicaciones_validas"]]
SUBTIPOS = CFG["delitos"]["subtipos"]
ESTADOS = CFG["delitos"]["estados_validos"]

COLUMNAS_OUTPUT = [
    "id_dgc_03", "lat_hecho", "long_hecho",
    "departamento_hecho", "provincia_hecho",
    "distrito_hecho", "ubigeo_hecho_delito",
    "direccion_hecho", "fecha_hora_hecho",
    "turno_hecho", "anio_hecho", "mes_hecho", "dia_hecho",
    "subtipo_hecho", "modalidad_hecho",
    "estado", "fuente"
]


def buscar_archivos(directorio):
    archivos = []
    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        archivos.extend(directorio.rglob(ext))
    return sorted(archivos)


def cargar_archivo(archivo):
    try:
        if archivo.suffix in (".xlsx", ".xls"):
            return pd.read_excel(archivo, engine="openpyxl")
        for sep in [",", "\t", ";"]:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(archivo, encoding=enc, sep=sep)
                    if len(df.columns) > 5:
                        return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
        return pd.read_csv(archivo, encoding="latin-1")
    except Exception as e:
        print(f"    Error leyendo {archivo.name}: {e}")
        return pd.DataFrame()


def cargar_todos(directorio):
    archivos = buscar_archivos(directorio)
    if not archivos:
        print(f"  No se encontraron archivos en {directorio}")
        return pd.DataFrame()

    print(f"  Archivos encontrados: {len(archivos)}")
    frames = []
    total = 0

    for arch in archivos:
        df = cargar_archivo(arch)
        if not df.empty:
            df.columns = df.columns.str.strip().str.lower()
            total += len(df)
            print(f"    OK {arch.relative_to(directorio)}: {len(df):,} registros")
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    print(f"\n  Total registros cargados: {total:,}")
    df_union = pd.concat(frames, ignore_index=True, join="outer")

    if "id_dgc_03" in df_union.columns:
        antes = len(df_union)
        df_union = df_union.drop_duplicates(subset=["id_dgc_03"])
        if antes - len(df_union) > 0:
            print(f"  Duplicados eliminados: {antes - len(df_union):,}")

    print(f"  Registros unicos: {len(df_union):,}")
    return df_union


def filtrar_ubicacion(df):
    print(f"\n{'-'*55}")
    print(f"  FILTRO 1: Lima Metropolitana + Callao")
    print(f"{'-'*55}")
    antes = len(df)

    if "departamento_hecho" in df.columns:
        df["departamento_hecho"] = df["departamento_hecho"].astype(str).str.strip().str.upper()
    if "provincia_hecho" in df.columns:
        df["provincia_hecho"] = df["provincia_hecho"].astype(str).str.strip().str.upper()

    mask = pd.Series(False, index=df.index)
    for dept, prov in UBICACIONES:
        mask |= (df["departamento_hecho"] == dept) & (df["provincia_hecho"] == prov)
    df = df[mask].copy()

    print(f"  {antes:,} -> {len(df):,} (descartados: {antes - len(df):,})")
    return df


def filtrar_delitos(df):
    print(f"\n{'-'*55}")
    print(f"  FILTRO 2: Solo {' + '.join(SUBTIPOS)}")
    print(f"{'-'*55}")
    antes = len(df)

    if "subtipo_hecho" in df.columns:
        df["subtipo_hecho"] = df["subtipo_hecho"].astype(str).str.strip().str.upper()
        df = df[df["subtipo_hecho"].isin(SUBTIPOS)].copy()

    print(f"  {antes:,} -> {len(df):,} (descartados: {antes - len(df):,})")
    return df


def filtrar_coordenadas(df):
    print(f"\n{'-'*55}")
    print(f"  FILTRO 3: Coordenadas validas")
    print(f"{'-'*55}")
    antes = len(df)

    if "estado" in df.columns:
        df["estado"] = pd.to_numeric(df["estado"], errors="coerce")
        for estado_val, count in df["estado"].value_counts().sort_index().items():
            pct = count / len(df) * 100
            etiq = {1: "OK", 2: "ubigeo pendiente", 3: "forzada a comisaria"}.get(int(estado_val), "?")
            print(f"    Estado {int(estado_val)}: {count:,} ({pct:.1f}%) - {etiq}")
        df = df[df["estado"].isin(ESTADOS)].copy()

    if len(df) == 0:
        return df

    for col in ["lat_hecho", "long_hecho"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["lat_hecho", "long_hecho"])

    df = df[
        (df["lat_hecho"] >= BBOX["lat_min"]) &
        (df["lat_hecho"] <= BBOX["lat_max"]) &
        (df["long_hecho"] >= BBOX["lon_min"]) &
        (df["long_hecho"] <= BBOX["lon_max"])
    ].copy()

    print(f"  {antes:,} -> {len(df):,} (descartados total: {antes - len(df):,})")
    return df


def generar_reporte(df):
    print(f"\n{'='*60}")
    print(f"  REPORTE FINAL")
    print(f"{'='*60}")
    print(f"\n  Total registros usables: {len(df):,}")

    if "subtipo_hecho" in df.columns:
        print(f"\n  Por tipo de delito:")
        for tipo, count in df["subtipo_hecho"].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {tipo}: {count:,} ({pct:.1f}%)")

    if "distrito_hecho" in df.columns:
        print(f"\n  Top 10 distritos:")
        for i, (dist, count) in enumerate(df["distrito_hecho"].value_counts().head(10).items(), 1):
            print(f"    {i:>2}. {dist}: {count:,}")

    print(f"\n  Rango espacial:")
    print(f"    Lat:  [{df['lat_hecho'].min():.4f}, {df['lat_hecho'].max():.4f}]")
    print(f"    Long: [{df['long_hecho'].min():.4f}, {df['long_hecho'].max():.4f}]")


if __name__ == "__main__":
    print("=" * 60)
    print("  LIMPIEZA DE DATOS DELICTIVOS - MININTER")
    print("=" * 60)
    print(f"  Input:  {RAW_DIR}")
    print(f"  Output: {OUTPUT_DIR}")

    if not RAW_DIR.exists():
        print(f"\n  ERROR: No existe {RAW_DIR}")
        print(f"  Coloca los CSVs descargados del MININTER alli.")
        sys.exit(1)

    df = cargar_todos(RAW_DIR)
    if df.empty:
        sys.exit(1)

    df = filtrar_ubicacion(df)
    if df.empty:
        sys.exit(1)
    df = filtrar_delitos(df)
    if df.empty:
        sys.exit(1)
    df = filtrar_coordenadas(df)
    if df.empty:
        sys.exit(1)

    # Guardar
    print(f"\n{'='*60}")
    print(f"  GUARDANDO RESULTADOS")
    print(f"{'='*60}")

    cols = [c for c in COLUMNAS_OUTPUT if c in df.columns]
    df_final = df[cols].copy()

    csv_path = OUTPUT_DIR / "delitos_lima_limpio.csv"
    df_final.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  CSV: {csv_path} ({len(df_final):,} registros)")

    try:
        import geopandas as gpd
        from shapely.geometry import Point
        geometry = [Point(lon, lat) for lon, lat in zip(df_final["long_hecho"], df_final["lat_hecho"])]
        gdf = gpd.GeoDataFrame(df_final, geometry=geometry, crs="EPSG:4326")
        geojson_path = OUTPUT_DIR / "delitos_lima_limpio.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"  GeoJSON: {geojson_path}")
    except Exception as e:
        print(f"  Error generando GeoJSON: {e}")

    generar_reporte(df_final)

    print(f"\n  SIGUIENTE PASO: scp el GeoJSON al servidor y ejecuta")
    print(f"  python scripts/02_pansharpening.py")
    print(f"{'='*60}")
