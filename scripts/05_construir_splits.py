#!/usr/bin/env python3
"""
05_construir_splits.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Construye los splits train/val/test del dataset etiquetado para entrenamiento
del modelo CNN. Usa split estratificado para mantener la proporcion de clases
en cada subset.

Entrada:
    data/labels/tiles_labeled.csv

Salida:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
    data/splits/splits_reporte.txt
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split, StratifiedGroupKFold
except ImportError:
    print("Error: scikit-learn no instalado.")
    print("  conda activate tesis")
    sys.exit(1)


# ============================================================
# CARGA DE CONFIGURACION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

LABELS_DIR = PROJECT_ROOT / CFG["paths"]["labels"]
SPLITS_DIR = PROJECT_ROOT / CFG["paths"]["splits"]

TRAIN_RATIO = CFG["splits"]["train_ratio"]
VAL_RATIO = CFG["splits"]["val_ratio"]
TEST_RATIO = CFG["splits"]["test_ratio"]
RANDOM_SEED = CFG["splits"]["random_seed"]
STRATIFY_BY = CFG["splits"]["stratify_by"]
GROUP_BY = CFG["splits"].get("group_by", "cell_id")

SPLITS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCIONES
# ============================================================

def cargar_dataset_etiquetado():
    """Carga el dataset etiquetado generado por el script 04."""
    csv_path = LABELS_DIR / "tiles_labeled.csv"
    if not csv_path.exists():
        print(f"  ERROR: No existe {csv_path}")
        print(f"  Ejecuta primero: python scripts/04_etiquetar_tiles.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  Dataset cargado: {len(df)} tiles")
    return df


def construir_splits_estratificados(df, train_ratio, val_ratio, test_ratio,
                                      stratify_col, random_seed):
    """
    Split estratificado aleatorio por tile (fallback cuando no hay columna de
    grupo). Mantiene la proporcion de clases pero NO garantiza separacion
    espacial: una misma ubicacion podria caer en train y test.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Los ratios deben sumar 1.0. Suma actual: {total_ratio}")

    test_val_size = val_ratio + test_ratio
    df_train, df_temp = train_test_split(
        df, test_size=test_val_size,
        stratify=df[stratify_col], random_state=random_seed)

    test_size_relative = test_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_temp, test_size=test_size_relative,
        stratify=df_temp[stratify_col], random_state=random_seed)

    return df_train, df_val, df_test


def construir_splits_agrupados(df, train_ratio, val_ratio, test_ratio,
                                stratify_col, group_col, random_seed):
    """
    Split estratificado Y agrupado por `group_col` (cell_id): ningun grupo
    (ubicacion fisica) se reparte entre train/val/test. Esto evita la fuga
    espacial (que la misma ubicacion entrene y evalue el modelo).

    Estrategia con StratifiedGroupKFold:
    1. Separar test (~test_ratio) del resto.
    2. Sobre train_val, separar val (~val_ratio del total).
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Los ratios deben sumar 1.0. Suma actual: {total_ratio}")

    df = df.reset_index(drop=True)
    y = df[stratify_col].values
    groups = df[group_col].values

    # 1ra particion: separar test. n_splits ~ 1/test_ratio (el fold de test es ~ese tamano)
    n_splits_test = max(2, round(1.0 / test_ratio))
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=random_seed)
    train_val_idx, test_idx = next(sgkf.split(df, y, groups))

    df_trainval = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # 2da particion: separar val del train_val
    val_frac = val_ratio / (train_ratio + val_ratio)
    n_splits_val = max(2, round(1.0 / val_frac))
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_seed)
    y2 = df_trainval[stratify_col].values
    g2 = df_trainval[group_col].values
    train_idx, val_idx = next(sgkf2.split(df_trainval, y2, g2))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, df_test


def calcular_distribucion(df, columna):
    """Calcula la distribucion de clases en un dataframe."""
    distribucion = df[columna].value_counts().sort_index()
    porcentajes = (distribucion / len(df) * 100).round(1)
    return distribucion, porcentajes


def imprimir_split_info(nombre, df, columna):
    """Imprime informacion de un split."""
    distribucion, porcentajes = calcular_distribucion(df, columna)
    print(f"\n  {nombre.upper()} ({len(df)} tiles):")
    for clase in distribucion.index:
        n = distribucion[clase]
        pct = porcentajes[clase]
        print(f"    {clase:<10}: {n:>4} ({pct:.1f}%)")


def generar_reporte(df_train, df_val, df_test, df_total, output_path):
    """Genera reporte de los splits creados."""
    lines = []
    lines.append("=" * 60)
    lines.append("  REPORTE DE CONSTRUCCION DE SPLITS")
    lines.append(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Random seed: {RANDOM_SEED}")
    if GROUP_COL_USED:
        lines.append(f"Estrategia: Split estratificado por '{STRATIFY_BY}' "
                     f"y agrupado por '{GROUP_COL_USED}' (split espacial)")
    else:
        lines.append(f"Estrategia: Split estratificado por '{STRATIFY_BY}' "
                     f"(SIN agrupacion espacial)")
    lines.append("")
    lines.append(f"Ratios configurados:")
    lines.append(f"  Train: {TRAIN_RATIO * 100:.0f}%")
    lines.append(f"  Val:   {VAL_RATIO * 100:.0f}%")
    lines.append(f"  Test:  {TEST_RATIO * 100:.0f}%")
    lines.append("")
    lines.append(f"Dataset total: {len(df_total)} tiles")
    lines.append("")

    for nombre, df in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
        distribucion, porcentajes = calcular_distribucion(df, STRATIFY_BY)
        lines.append(f"{nombre} ({len(df)} tiles, {len(df)/len(df_total)*100:.1f}% del total):")
        for clase in distribucion.index:
            n = distribucion[clase]
            pct = porcentajes[clase]
            lines.append(f"  {clase:<10}: {n:>4} ({pct:.1f}%)")
        lines.append("")

    # Verificacion de no overlap. Con split espacial se compara por grupo
    # (cell_id = ubicacion fisica), no por tile_name, para detectar fuga.
    verif_col = GROUP_COL_USED if GROUP_COL_USED else "tile_name"
    train_set = set(df_train[verif_col])
    val_set = set(df_val[verif_col])
    test_set = set(df_test[verif_col])

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    lines.append(f"Verificacion de splits disjuntos (por '{verif_col}'):")
    lines.append(f"  Train-Val overlap:  {len(overlap_train_val)} (debe ser 0)")
    lines.append(f"  Train-Test overlap: {len(overlap_train_test)} (debe ser 0)")
    lines.append(f"  Val-Test overlap:   {len(overlap_val_test)} (debe ser 0)")

    assert not overlap_train_val, "Fuga espacial: cell_id compartido entre train y val"
    assert not overlap_train_test, "Fuga espacial: cell_id compartido entre train y test"
    assert not overlap_val_test, "Fuga espacial: cell_id compartido entre val y test"

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  CONSTRUCCION DE SPLITS TRAIN/VAL/TEST")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    # Cargar dataset etiquetado
    print("\n  [1/3] Cargando dataset etiquetado...")
    df = cargar_dataset_etiquetado()

    # Distribucion total
    print("\n  Distribucion total del dataset:")
    distribucion, porcentajes = calcular_distribucion(df, STRATIFY_BY)
    for clase in distribucion.index:
        n = distribucion[clase]
        pct = porcentajes[clase]
        print(f"    {clase:<10}: {n:>4} ({pct:.1f}%)")

    # Verificar que hay suficientes muestras por clase
    min_samples = distribucion.min()
    if min_samples < 5:
        print(f"\n  ADVERTENCIA: La clase minoritaria solo tiene {min_samples} muestras.")
        print(f"  El split puede no ser estadisticamente significativo.")

    # Construir splits: agrupados por cell_id (espacial) si la columna existe.
    if GROUP_BY in df.columns:
        GROUP_COL_USED = GROUP_BY
        n_grupos = df[GROUP_BY].nunique()
        print(f"\n  [2/3] Construyendo splits estratificados y AGRUPADOS por "
              f"'{GROUP_BY}' ({n_grupos} grupos)...")
        df_train, df_val, df_test = construir_splits_agrupados(
            df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, STRATIFY_BY, GROUP_BY, RANDOM_SEED
        )
    else:
        GROUP_COL_USED = None
        print(f"\n  [2/3] ADVERTENCIA: no existe la columna '{GROUP_BY}' en el dataset.")
        print(f"  Usando split estratificado SIN agrupacion espacial (re-ejecuta el")
        print(f"  script 03 con el mosaico para obtener cell_id).")
        df_train, df_val, df_test = construir_splits_estratificados(
            df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, STRATIFY_BY, RANDOM_SEED
        )

    # Imprimir informacion de cada split
    imprimir_split_info("Train", df_train, STRATIFY_BY)
    imprimir_split_info("Val", df_val, STRATIFY_BY)
    imprimir_split_info("Test", df_test, STRATIFY_BY)

    # Guardar splits
    print("\n  [3/3] Guardando splits...")
    df_train.to_csv(SPLITS_DIR / "train.csv", index=False)
    df_val.to_csv(SPLITS_DIR / "val.csv", index=False)
    df_test.to_csv(SPLITS_DIR / "test.csv", index=False)
    print(f"  Train: {SPLITS_DIR / 'train.csv'}")
    print(f"  Val:   {SPLITS_DIR / 'val.csv'}")
    print(f"  Test:  {SPLITS_DIR / 'test.csv'}")

    # Generar reporte
    reporte_path = SPLITS_DIR / "splits_reporte.txt"
    print(f"\n  Reporte: {reporte_path}\n")
    generar_reporte(df_train, df_val, df_test, df, reporte_path)

    print(f"\n  SIGUIENTE PASO: python scripts/06_entrenar_modelo.py")
    print(f"{'='*60}")
