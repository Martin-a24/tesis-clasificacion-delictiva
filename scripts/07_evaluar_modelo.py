#!/usr/bin/env python3
"""
07_evaluar_modelo.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Evalua el modelo entrenado sobre el conjunto de test, generando metricas
detalladas, matriz de confusion y comparacion con baselines.

Caracteristicas:
- Carga el mejor modelo guardado durante el entrenamiento
- Calcula metricas globales y por clase (accuracy, precision, recall, F1)
- Genera matriz de confusion (texto e imagen)
- Compara con baselines: clasificador aleatorio y clasificador mayoritario
- Guarda predicciones detalladas por tile
- Genera reporte completo para incluir en el informe de la tesis

Entrada:
    data/splits/test.csv
    models/{architecture}_best.pth

Salida:
    results/evaluation_{architecture}_{timestamp}.txt
    results/confusion_matrix_{architecture}_{timestamp}.png
    results/predictions_{architecture}_{timestamp}.csv
    results/metrics_{architecture}_{timestamp}.json
"""

import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report, f1_score
    )
except ImportError as e:
    print(f"Error: dependencias no instaladas: {e}")
    print("  conda activate tesis")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Importar componentes del script de entrenamiento
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

# Cargar el modulo del script 06 dinamicamente
spec_module = import_module("06_entrenar_modelo")
TilesDataset = spec_module.TilesDataset
construir_modelo = spec_module.construir_modelo


# ============================================================
# CARGA DE CONFIGURACION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

TILES_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["tiles"]
SPLITS_DIR = PROJECT_ROOT / CFG["paths"]["splits"]
MODELS_DIR = PROJECT_ROOT / CFG["paths"]["models"]
RESULTS_DIR = PROJECT_ROOT / CFG["paths"]["results"]

NIVELES = CFG["etiquetado"]["niveles"]
NIVEL_A_IDX = {nivel: i for i, nivel in enumerate(NIVELES)}
IDX_A_NIVEL = {i: nivel for i, nivel in enumerate(NIVELES)}

ARCHITECTURE = CFG["modelo"]["architecture"]
IN_CHANNELS = CFG["modelo"]["in_channels"]
NUM_CLASSES = CFG["modelo"]["num_classes"]

BATCH_SIZE = CFG["entrenamiento"]["batch_size"]
DEVICE = CFG["entrenamiento"]["device"]
NUM_WORKERS = CFG["entrenamiento"]["num_workers"]
PIN_MEMORY = CFG["entrenamiento"]["pin_memory"]

GENERATE_CM = CFG["evaluacion"]["generate_confusion_matrix"]
COMPARE_BASELINES = CFG["evaluacion"]["compare_baselines"]
SAVE_PREDICTIONS = CFG["evaluacion"]["save_predictions"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EVALUACION
# ============================================================

def predecir_en_test(model, loader, device):
    """Realiza predicciones sobre el conjunto de test."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def calcular_metricas_completas(y_true, y_pred, niveles):
    """
    Calcula todas las metricas de clasificacion:
    - Accuracy global
    - F1 macro y ponderado
    - Precision, recall, F1 por clase
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Metricas por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(niveles)), zero_division=0
    )

    metrics_per_class = []
    for i, nivel in enumerate(niveles):
        metrics_per_class.append({
            "clase": nivel,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class": metrics_per_class,
    }


# ============================================================
# BASELINES
# ============================================================

def baseline_aleatorio(y_true, n_classes, random_seed=42):
    """Clasificador aleatorio uniforme."""
    rng = np.random.RandomState(random_seed)
    y_pred = rng.randint(0, n_classes, size=len(y_true))
    return y_pred


def baseline_mayoritario(y_true, df_train, niveles):
    """Clasificador que siempre predice la clase mayoritaria del train."""
    clase_mayoritaria = df_train["nivel_riesgo"].value_counts().idxmax()
    idx_mayoritaria = NIVEL_A_IDX[clase_mayoritaria]
    y_pred = np.full(len(y_true), idx_mayoritaria, dtype=int)
    return y_pred, clase_mayoritaria


# ============================================================
# VISUALIZACION
# ============================================================

def graficar_matriz_confusion(y_true, y_pred, niveles, output_path, titulo=""):
    """Grafica la matriz de confusion como heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(niveles)))

    # Calcular porcentajes por fila (recall por clase)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Matriz absoluta
    im0 = axes[0].imshow(cm, cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(len(niveles)))
    axes[0].set_yticks(range(len(niveles)))
    axes[0].set_xticklabels(niveles)
    axes[0].set_yticklabels(niveles)
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Real")
    axes[0].set_title("Matriz de Confusion (conteos)")

    for i in range(len(niveles)):
        for j in range(len(niveles)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Matriz normalizada
    im1 = axes[1].imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    axes[1].set_xticks(range(len(niveles)))
    axes[1].set_yticks(range(len(niveles)))
    axes[1].set_xticklabels(niveles)
    axes[1].set_yticklabels(niveles)
    axes[1].set_xlabel("Prediccion")
    axes[1].set_ylabel("Real")
    axes[1].set_title("Matriz de Confusion (% por clase real)")

    for i in range(len(niveles)):
        for j in range(len(niveles)):
            color = "white" if cm_norm[i, j] > 50 else "black"
            axes[1].text(j, i, f"{cm_norm[i, j]:.1f}%",
                          ha="center", va="center", color=color)

    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if titulo:
        fig.suptitle(titulo)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# REPORTE
# ============================================================

def generar_reporte(metrics_modelo, metrics_aleatorio, metrics_mayoritario,
                     y_true, y_pred, niveles, architecture,
                     clase_mayoritaria, output_path):
    """Genera un reporte completo de la evaluacion en formato texto."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(niveles)))

    lines = []
    lines.append("=" * 60)
    lines.append("  REPORTE DE EVALUACION DEL MODELO")
    lines.append(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"  Arquitectura: {architecture}")
    lines.append(f"  Tiles en test: {len(y_true)}")
    lines.append("=" * 60)
    lines.append("")

    # Metricas globales del modelo
    lines.append("METRICAS GLOBALES DEL MODELO:")
    lines.append(f"  Accuracy:    {metrics_modelo['accuracy']:.4f}")
    lines.append(f"  F1 macro:    {metrics_modelo['f1_macro']:.4f}")
    lines.append(f"  F1 weighted: {metrics_modelo['f1_weighted']:.4f}")
    lines.append("")

    # Metricas por clase
    lines.append("METRICAS POR CLASE:")
    lines.append(f"  {'Clase':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for m in metrics_modelo["per_class"]:
        lines.append(f"  {m['clase']:<10} "
                      f"{m['precision']:>10.4f} {m['recall']:>10.4f} "
                      f"{m['f1']:>10.4f} {m['support']:>10}")
    lines.append("")

    # Matriz de confusion
    lines.append("MATRIZ DE CONFUSION (filas=real, columnas=prediccion):")
    header = f"  {'':<10}" + " ".join(f"{n:>8}" for n in niveles)
    lines.append(header)
    for i, nivel in enumerate(niveles):
        row = f"  {nivel:<10}" + " ".join(f"{cm[i,j]:>8}" for j in range(len(niveles)))
        lines.append(row)
    lines.append("")

    # Comparacion con baselines
    if metrics_aleatorio is not None:
        lines.append("COMPARACION CON BASELINES:")
        lines.append("")
        lines.append(f"  {'Metrica':<15} {'Modelo':>10} {'Aleatorio':>10} {'Mayoritario':>12}")
        lines.append(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
        lines.append(f"  {'Accuracy':<15} "
                      f"{metrics_modelo['accuracy']:>10.4f} "
                      f"{metrics_aleatorio['accuracy']:>10.4f} "
                      f"{metrics_mayoritario['accuracy']:>12.4f}")
        lines.append(f"  {'F1 macro':<15} "
                      f"{metrics_modelo['f1_macro']:>10.4f} "
                      f"{metrics_aleatorio['f1_macro']:>10.4f} "
                      f"{metrics_mayoritario['f1_macro']:>12.4f}")
        lines.append(f"  {'F1 weighted':<15} "
                      f"{metrics_modelo['f1_weighted']:>10.4f} "
                      f"{metrics_aleatorio['f1_weighted']:>10.4f} "
                      f"{metrics_mayoritario['f1_weighted']:>12.4f}")
        lines.append("")
        lines.append(f"  Baseline mayoritario predice siempre: '{clase_mayoritaria}'")
        lines.append("")

        # Mejora relativa sobre baselines
        mejora_aleatorio = (metrics_modelo['f1_macro'] - metrics_aleatorio['f1_macro']) * 100
        mejora_mayoritario = (metrics_modelo['f1_macro'] - metrics_mayoritario['f1_macro']) * 100
        lines.append(f"  Mejora F1 macro sobre baseline aleatorio:    +{mejora_aleatorio:.1f} puntos")
        lines.append(f"  Mejora F1 macro sobre baseline mayoritario:  +{mejora_mayoritario:.1f} puntos")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  EVALUACION DE MODELO CNN")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Arquitectura: {ARCHITECTURE}")
    print("=" * 60)

    # Verificar dispositivo
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo: {device}")

    # Cargar test split
    print(f"\n  [1/6] Cargando conjunto de test...")
    test_csv = SPLITS_DIR / "test.csv"
    train_csv = SPLITS_DIR / "train.csv"
    if not test_csv.exists():
        print(f"  ERROR: No existe {test_csv}")
        sys.exit(1)

    df_test = pd.read_csv(test_csv)
    df_train = pd.read_csv(train_csv)
    print(f"  Tiles en test: {len(df_test)}")

    # Distribucion del test
    print(f"\n  Distribucion del test:")
    for nivel in NIVELES:
        n = (df_test["nivel_riesgo"] == nivel).sum()
        pct = n / len(df_test) * 100
        print(f"    {nivel:<10}: {n:>4} ({pct:.1f}%)")

    # Construir dataloader
    print(f"\n  [2/6] Construyendo dataloader...")
    test_dataset = TilesDataset(df_test, TILES_DIR, transform=None)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Cargar modelo
    print(f"\n  [3/6] Cargando modelo...")
    model_path = MODELS_DIR / f"{ARCHITECTURE}_best.pth"
    if not model_path.exists():
        print(f"  ERROR: No existe {model_path}")
        print(f"  Ejecuta primero: python scripts/06_entrenar_modelo.py")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = construir_modelo(ARCHITECTURE, NUM_CLASSES, IN_CHANNELS, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"  Modelo cargado desde epoca {checkpoint['epoch']}")
    print(f"  F1 (val) en mejor epoca: {checkpoint['val_f1']:.4f}")

    # Predicciones en test
    print(f"\n  [4/6] Evaluando modelo en test...")
    y_true, y_pred, y_probs = predecir_en_test(model, test_loader, device)

    # Metricas del modelo
    metrics_modelo = calcular_metricas_completas(y_true, y_pred, NIVELES)

    # Baselines
    metrics_aleatorio = None
    metrics_mayoritario = None
    clase_mayoritaria = None

    if COMPARE_BASELINES:
        print(f"\n  [5/6] Calculando baselines...")
        y_pred_aleatorio = baseline_aleatorio(y_true, NUM_CLASSES, random_seed=42)
        metrics_aleatorio = calcular_metricas_completas(y_true, y_pred_aleatorio, NIVELES)

        y_pred_mayoritario, clase_mayoritaria = baseline_mayoritario(y_true, df_train, NIVELES)
        metrics_mayoritario = calcular_metricas_completas(y_true, y_pred_mayoritario, NIVELES)
    else:
        print(f"\n  [5/6] Baselines deshabilitados.")

    # Guardar resultados
    print(f"\n  [6/6] Guardando reportes...")

    # Reporte de texto
    reporte_path = RESULTS_DIR / f"evaluation_{ARCHITECTURE}_{timestamp}.txt"
    print()
    generar_reporte(metrics_modelo, metrics_aleatorio, metrics_mayoritario,
                    y_true, y_pred, NIVELES, ARCHITECTURE,
                    clase_mayoritaria, reporte_path)
    print(f"\n  Reporte: {reporte_path}")

    # Matriz de confusion
    if GENERATE_CM:
        cm_path = RESULTS_DIR / f"confusion_matrix_{ARCHITECTURE}_{timestamp}.png"
        graficar_matriz_confusion(y_true, y_pred, NIVELES, cm_path,
                                    titulo=f"Matriz de Confusion - {ARCHITECTURE}")
        print(f"  Matriz de confusion: {cm_path}")

    # Predicciones detalladas
    if SAVE_PREDICTIONS:
        df_test_pred = df_test.copy()
        df_test_pred["nivel_predicho"] = [IDX_A_NIVEL[p] for p in y_pred]
        df_test_pred["correcto"] = df_test_pred["nivel_riesgo"] == df_test_pred["nivel_predicho"]
        for i, nivel in enumerate(NIVELES):
            df_test_pred[f"prob_{nivel}"] = y_probs[:, i].round(4)

        pred_path = RESULTS_DIR / f"predictions_{ARCHITECTURE}_{timestamp}.csv"
        df_test_pred.to_csv(pred_path, index=False)
        print(f"  Predicciones: {pred_path}")

    # Metricas en JSON (util para procesamiento posterior)
    metrics_path = RESULTS_DIR / f"metrics_{ARCHITECTURE}_{timestamp}.json"
    metrics_data = {
        "fecha": datetime.now().isoformat(),
        "architecture": ARCHITECTURE,
        "n_test_tiles": len(y_true),
        "modelo": metrics_modelo,
    }
    if metrics_aleatorio is not None:
        metrics_data["baseline_aleatorio"] = metrics_aleatorio
        metrics_data["baseline_mayoritario"] = metrics_mayoritario
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Metricas (JSON): {metrics_path}")

    print(f"\n  EVALUACION COMPLETADA")
    print(f"{'='*60}")
