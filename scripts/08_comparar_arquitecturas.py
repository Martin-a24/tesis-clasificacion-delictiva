#!/usr/bin/env python3
"""
08_comparar_arquitecturas.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Compara multiples arquitecturas CNN entrenando y evaluando cada una
secuencialmente. Genera un reporte comparativo con todas las metricas.

Para cada arquitectura:
  1. Modifica configs/config.yaml temporalmente con la arquitectura
  2. Ejecuta scripts/06_entrenar_modelo.py
  3. Ejecuta scripts/07_evaluar_modelo.py
  4. Recolecta los resultados desde results/metrics_*.json

Al final, genera un reporte comparativo y un grafico de barras.

Uso:
    python scripts/08_comparar_arquitecturas.py

    # O con arquitecturas especificas:
    python scripts/08_comparar_arquitecturas.py --arquitecturas resnet18 efficientnet_b0
"""

import sys
import yaml
import json
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"

ARQUITECTURAS_DISPONIBLES = ["resnet18", "resnet50", "efficientnet_b0", "vit_b_16"]
ARQUITECTURAS_DEFAULT = ["resnet18", "resnet50", "efficientnet_b0"]

COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCIONES
# ============================================================

def actualizar_config_arquitectura(arquitectura):
    """
    Modifica configs/config.yaml para usar la arquitectura especificada.
    Mantiene una copia de seguridad del config original.
    """
    backup_path = CONFIG_PATH.with_suffix(".yaml.backup")
    if not backup_path.exists():
        shutil.copy(CONFIG_PATH, backup_path)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config["modelo"]["architecture"] = arquitectura

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def restaurar_config_original():
    """Restaura el config.yaml a su estado original."""
    backup_path = CONFIG_PATH.with_suffix(".yaml.backup")
    if backup_path.exists():
        shutil.copy(backup_path, CONFIG_PATH)
        backup_path.unlink()


def ejecutar_script(script_name, descripcion):
    """Ejecuta un script y retorna True si fue exitoso."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  ERROR: No existe {script_path}")
        return False

    print(f"\n  >>> Ejecutando {descripcion}...")
    inicio = time.time()

    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=False,
        text=True
    )

    duracion = time.time() - inicio
    if result.returncode == 0:
        print(f"  >>> {descripcion} completado en {duracion / 60:.1f} minutos")
        return True
    else:
        print(f"  >>> ERROR en {descripcion}")
        return False


def cargar_metricas_recientes(arquitectura):
    """
    Carga las metricas mas recientes de la arquitectura especificada
    desde results/metrics_*.json.
    """
    pattern = f"metrics_{arquitectura}_*.json"
    archivos = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not archivos:
        print(f"  ADVERTENCIA: No se encontraron metricas para {arquitectura}")
        return None

    archivo_reciente = archivos[0]
    with open(archivo_reciente) as f:
        return json.load(f)


def construir_tabla_comparativa(resultados):
    """Construye un DataFrame con metricas comparativas."""
    filas = []

    for arq, data in resultados.items():
        if data is None:
            continue

        metricas = data["modelo"]
        fila = {
            "arquitectura": arq,
            "accuracy": metricas["accuracy"],
            "f1_macro": metricas["f1_macro"],
            "f1_weighted": metricas["f1_weighted"],
        }

        for clase_metricas in metricas["per_class"]:
            clase = clase_metricas["clase"]
            fila[f"f1_{clase}"] = clase_metricas["f1"]
            fila[f"precision_{clase}"] = clase_metricas["precision"]
            fila[f"recall_{clase}"] = clase_metricas["recall"]

        filas.append(fila)

    if not filas:
        return None

    df = pd.DataFrame(filas)
    return df


def graficar_comparacion(df_comparacion, output_path):
    """Grafica metricas comparativas entre arquitecturas."""
    if df_comparacion is None or len(df_comparacion) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Metricas globales
    metricas_globales = ["accuracy", "f1_macro", "f1_weighted"]
    x = np.arange(len(df_comparacion))
    width = 0.25

    for i, metrica in enumerate(metricas_globales):
        offset = (i - 1) * width
        axes[0].bar(x + offset, df_comparacion[metrica], width, label=metrica)

    axes[0].set_xlabel("Arquitectura")
    axes[0].set_ylabel("Valor")
    axes[0].set_title("Metricas globales por arquitectura")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_comparacion["arquitectura"], rotation=15)
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    axes[0].grid(alpha=0.3, axis="y")

    # 2. F1 por clase
    cols_f1_clase = [c for c in df_comparacion.columns if c.startswith("f1_") and c not in ["f1_macro", "f1_weighted"]]
    clases = [c.replace("f1_", "") for c in cols_f1_clase]

    for i, clase in enumerate(clases):
        offset = (i - len(clases) / 2 + 0.5) * width
        axes[1].bar(x + offset, df_comparacion[f"f1_{clase}"], width, label=clase)

    axes[1].set_xlabel("Arquitectura")
    axes[1].set_ylabel("F1 score")
    axes[1].set_title("F1 score por clase")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_comparacion["arquitectura"], rotation=15)
    axes[1].legend(title="Clase")
    axes[1].set_ylim([0, 1])
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generar_reporte_comparativo(df_comparacion, resultados, output_path):
    """Genera reporte comparativo en formato texto."""
    lines = []
    lines.append("=" * 70)
    lines.append("  REPORTE COMPARATIVO DE ARQUITECTURAS")
    lines.append(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 70)
    lines.append("")

    if df_comparacion is None or len(df_comparacion) == 0:
        lines.append("No se obtuvieron resultados para ninguna arquitectura.")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        return

    # Tabla de metricas globales
    lines.append("METRICAS GLOBALES:")
    lines.append("")
    lines.append(f"  {'Arquitectura':<20} {'Accuracy':>10} {'F1 macro':>10} {'F1 weight.':>10}")
    lines.append(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    for _, row in df_comparacion.iterrows():
        lines.append(f"  {row['arquitectura']:<20} "
                      f"{row['accuracy']:>10.4f} "
                      f"{row['f1_macro']:>10.4f} "
                      f"{row['f1_weighted']:>10.4f}")
    lines.append("")

    # Mejor arquitectura
    mejor_idx = df_comparacion["f1_macro"].idxmax()
    mejor_arq = df_comparacion.loc[mejor_idx, "arquitectura"]
    mejor_f1 = df_comparacion.loc[mejor_idx, "f1_macro"]
    lines.append(f"MEJOR ARQUITECTURA (segun F1 macro): {mejor_arq} (F1 = {mejor_f1:.4f})")
    lines.append("")

    # F1 por clase
    cols_f1_clase = [c for c in df_comparacion.columns if c.startswith("f1_") and c not in ["f1_macro", "f1_weighted"]]
    clases = [c.replace("f1_", "") for c in cols_f1_clase]

    if clases:
        lines.append("F1 SCORE POR CLASE:")
        lines.append("")
        header = f"  {'Arquitectura':<20}"
        for clase in clases:
            header += f" {clase:>10}"
        lines.append(header)
        lines.append(f"  {'-' * 20}" + " ".join(["-" * 10] * len(clases)))

        for _, row in df_comparacion.iterrows():
            linea = f"  {row['arquitectura']:<20}"
            for clase in clases:
                linea += f" {row[f'f1_{clase}']:>10.4f}"
            lines.append(linea)
        lines.append("")

    # Comparacion con baselines
    if resultados:
        primera_arq = list(resultados.keys())[0]
        if resultados[primera_arq] and "baseline_aleatorio" in resultados[primera_arq]:
            lines.append("BASELINES (calculados desde el primer modelo evaluado):")
            lines.append(f"  Aleatorio    F1 macro: {resultados[primera_arq]['baseline_aleatorio']['f1_macro']:.4f}")
            lines.append(f"  Mayoritario  F1 macro: {resultados[primera_arq]['baseline_mayoritario']['f1_macro']:.4f}")
            lines.append("")

    lines.append("=" * 70)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparar arquitecturas CNN")
    parser.add_argument(
        "--arquitecturas",
        nargs="+",
        choices=ARQUITECTURAS_DISPONIBLES,
        default=ARQUITECTURAS_DEFAULT,
        help=f"Arquitecturas a comparar. Default: {ARQUITECTURAS_DEFAULT}"
    )
    parser.add_argument(
        "--solo-evaluar",
        action="store_true",
        help="Solo evaluar (no entrenar). Usa modelos previamente entrenados."
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  COMPARACION DE ARQUITECTURAS CNN")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Arquitecturas a comparar: {args.arquitecturas}")
    print(f"  Modo: {'Solo evaluar' if args.solo_evaluar else 'Entrenar + evaluar'}")
    print("=" * 70)

    inicio_total = time.time()
    resultados = {}

    try:
        for i, arq in enumerate(args.arquitecturas, 1):
            print(f"\n{'=' * 70}")
            print(f"  [{i}/{len(args.arquitecturas)}] ARQUITECTURA: {arq}")
            print(f"{'=' * 70}")

            # Actualizar config
            actualizar_config_arquitectura(arq)
            print(f"  Config actualizada con architecture = {arq}")

            # Entrenar (si corresponde)
            if not args.solo_evaluar:
                if not ejecutar_script("06_entrenar_modelo.py", "entrenamiento"):
                    print(f"  Saltando {arq} debido a error en entrenamiento")
                    continue

            # Evaluar
            if not ejecutar_script("07_evaluar_modelo.py", "evaluacion"):
                print(f"  Saltando {arq} debido a error en evaluacion")
                continue

            # Cargar metricas
            metricas = cargar_metricas_recientes(arq)
            resultados[arq] = metricas

    finally:
        # Restaurar config original
        restaurar_config_original()
        print("\n  Config original restaurado")

    # Generar reporte comparativo
    duracion_total = time.time() - inicio_total
    print(f"\n\n{'=' * 70}")
    print(f"  GENERANDO REPORTE COMPARATIVO")
    print(f"  Tiempo total: {duracion_total / 60:.1f} minutos")
    print(f"{'=' * 70}\n")

    df_comparacion = construir_tabla_comparativa(resultados)

    # Guardar reportes
    if df_comparacion is not None:
        csv_path = COMPARISONS_DIR / f"comparison_{timestamp}.csv"
        df_comparacion.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")

        reporte_path = COMPARISONS_DIR / f"comparison_{timestamp}.txt"
        generar_reporte_comparativo(df_comparacion, resultados, reporte_path)
        print(f"  Reporte: {reporte_path}")

        grafico_path = COMPARISONS_DIR / f"comparison_{timestamp}.png"
        graficar_comparacion(df_comparacion, grafico_path)
        print(f"  Grafico: {grafico_path}")

    print(f"\n{'=' * 70}")
    print("  COMPARACION COMPLETADA")
    print(f"{'=' * 70}")
