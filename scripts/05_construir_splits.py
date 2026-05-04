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
    from sklearn.model_selection import train_test_split
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
    Construye splits estratificados manteniendo la proporcion de clases.

    Estrategia:
    1. Primer split: train vs (val + test)
    2. Segundo split: val vs test
    """
    # Validar ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Los ratios deben sumar 1.0. Suma actual: {total_ratio}")

    # Primer split: separar train del resto
    test_val_size = val_ratio + test_ratio
    df_train, df_temp = train_test_split(
        df,
        test_size=test_val_size,
        stratify=df[stratify_col],
        random_state=random_seed
    )

    # Segundo split: separar val y test del temp
    test_size_relative = test_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_size_relative,
        stratify=df_temp[stratify_col],
        random_state=random_seed
    )

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
    lines.append(f"Estrategia: Split estratificado por '{STRATIFY_BY}'")
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

    # Verificacion de no overlap
    train_set = set(df_train["tile_name"])
    val_set = set(df_val["tile_name"])
    test_set = set(df_test["tile_name"])

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    lines.append("Verificacion de splits disjuntos:")
    lines.append(f"  Train-Val overlap:  {len(overlap_train_val)} (debe ser 0)")
    lines.append(f"  Train-Test overlap: {len(overlap_train_test)} (debe ser 0)")
    lines.append(f"  Val-Test overlap:   {len(overlap_val_test)} (debe ser 0)")

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

    # Construir splits estratificados
    print("\n  [2/3] Construyendo splits estratificados...")
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
