#!/usr/bin/env python3
"""
10_zonificacion.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Objetivo 3 (R8): genera el mapa de zonificacion de riesgo delictivo a partir de
las PREDICCIONES del modelo sobre todos los tiles del area de estudio, y analiza
la coincidencia espacial entre esa clasificacion y los registros delictivos del
MININTER.

Entrada:
    models/{architecture}_best.pth      (modelo entrenado en 06)
    data/labels/tiles_labeled.geojson   (tiles con geometria, nivel real, n_delitos)
    data/splits/{train,val,test}.csv    (para marcar a que split pertenece cada tile)
    data/processed/delitos_limpios/delitos_lima_limpio.geojson

Salida:
    data/labels/tiles_predicciones.geojson      (capa para QGIS)
    results/zonificacion/mapa_zonificacion.png   (vista rapida)
    results/zonificacion/coincidencia_reporte.txt

Uso:
    python scripts/10_zonificacion.py
"""

import sys
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
    import geopandas as gpd
except ImportError as e:
    print(f"Error: dependencias no instaladas: {e}")
    print("  conda activate tesis")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Reusar 09 (que a su vez reusa 06): cargador de modelo, dataset y config.
_spec = importlib.util.spec_from_file_location("gradcam09", SCRIPTS_DIR / "09_gradcam.py")
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)
ent = gc.ent

CFG = ent.CFG
TILES_DIR = ent.TILES_DIR
SPLITS_DIR = ent.SPLITS_DIR
MODELS_DIR = ent.MODELS_DIR
RESULTS_DIR = ent.RESULTS_DIR
NIVELES = ent.NIVELES
IDX_A_NIVEL = ent.IDX_A_NIVEL
ARCHITECTURE = ent.ARCHITECTURE

LABELS_DIR = PROJECT_ROOT / CFG["paths"]["labels"]
DELITOS_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["delitos"]
ZON_DIR = RESULTS_DIR / "zonificacion"

COLORES = {"bajo": "#2ecc71", "medio": "#f1c40f", "alto": "#e74c3c"}


def split_de_cada_tile():
    mapa = {}
    for nombre in ["train", "val", "test"]:
        p = SPLITS_DIR / f"{nombre}.csv"
        if p.exists():
            for t in pd.read_csv(p)["tile_name"]:
                mapa[t] = nombre
    return mapa


def predecir_todos(model, gdf, device):
    ds = ent.TilesDataset(gdf, TILES_DIR, transform=None)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=ent.NUM_WORKERS, pin_memory=ent.PIN_MEMORY)
    preds, probs = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            out = model(imgs.to(device)).softmax(dim=1)
            preds.extend(out.argmax(dim=1).cpu().tolist())
            probs.extend(out.max(dim=1).values.cpu().tolist())
    return preds, probs


if __name__ == "__main__":
    print("=" * 60)
    print("  ZONIFICACION DE RIESGO (Objetivo 3 - R8)")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    device = torch.device(ent.DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = MODELS_DIR / f"{ARCHITECTURE}_best.pth"
    geojson = LABELS_DIR / "tiles_labeled.geojson"
    for req in [ckpt, geojson]:
        if not req.exists():
            print(f"\n  ERROR: falta {req}")
            print("  Necesitas: modelo entrenado (06) y etiquetado (04).")
            sys.exit(1)

    print(f"\n  [1/4] Cargando modelo {ARCHITECTURE} y tiles...")
    model, arch, niveles = gc.cargar_modelo(ckpt, device)
    gdf = gpd.read_file(geojson)
    print(f"  Tiles: {len(gdf)}  | dispositivo: {device}")

    print(f"\n  [2/4] Prediciendo nivel de riesgo en todos los tiles...")
    preds, probs = predecir_todos(model, gdf, device)
    gdf["nivel_pred"] = [IDX_A_NIVEL[i] for i in preds]
    gdf["prob_pred"] = np.round(probs, 4)
    gdf = gdf.rename(columns={"nivel_riesgo": "nivel_true"})
    gdf["split"] = gdf["tile_name"].map(split_de_cada_tile()).fillna("-")

    print(f"\n  [3/4] Guardando capa de zonificacion...")
    ZON_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["tile_name", "nivel_true", "nivel_pred", "prob_pred", "n_delitos", "split"]
    cols = [c for c in cols if c in gdf.columns]
    if "cell_id" in gdf.columns:
        cols.insert(1, "cell_id")
    out_geojson = LABELS_DIR / "tiles_predicciones.geojson"
    gdf[cols + ["geometry"]].to_file(out_geojson, driver="GeoJSON")
    print(f"  GeoJSON (QGIS): {out_geojson}")

    # Mapa rapido coloreado por nivel predicho + puntos de delitos
    fig, ax = plt.subplots(figsize=(10, 10))
    for niv in NIVELES:
        sub = gdf[gdf["nivel_pred"] == niv]
        if len(sub):
            sub.plot(ax=ax, color=COLORES.get(niv, "gray"), edgecolor="none",
                     label=f"{niv} ({len(sub)})")
    delitos_path = DELITOS_DIR / "delitos_lima_limpio.geojson"
    if delitos_path.exists():
        d = gpd.read_file(delitos_path).to_crs(gdf.crs)
        d.plot(ax=ax, color="black", markersize=1, alpha=0.25)
    ax.set_title("Zonificacion de riesgo (prediccion del modelo) + delitos MININTER")
    ax.legend(); ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(ZON_DIR / "mapa_zonificacion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mapa:           {ZON_DIR / 'mapa_zonificacion.png'}")

    print(f"\n  [4/4] Analisis de coincidencia espacial con delitos...")
    lines = []
    lines.append("=" * 60)
    lines.append("  COINCIDENCIA ESPACIAL: prediccion del modelo vs delitos")
    lines.append(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Delitos promedio por categoria PREDICHA (deberia crecer bajo<medio<alto):")
    if "n_delitos" in gdf.columns:
        tab = gdf.groupby("nivel_pred")["n_delitos"].agg(["count", "mean", "median", "sum"])
        for niv in NIVELES:
            if niv in tab.index:
                r = tab.loc[niv]
                lines.append(f"  {niv:<7}: tiles={int(r['count']):>4}  "
                             f"media={r['mean']:.2f}  mediana={r['median']:.0f}  "
                             f"delitos_tot={int(r['sum'])}")
        total_delitos = gdf["n_delitos"].sum()
        alto_delitos = gdf.loc[gdf["nivel_pred"] == "alto", "n_delitos"].sum()
        alto_tiles = (gdf["nivel_pred"] == "alto").mean() * 100
        if total_delitos > 0:
            lines.append("")
            lines.append(f"Los tiles predichos 'alto' son {alto_tiles:.1f}% del area y "
                         f"concentran {alto_delitos / total_delitos * 100:.1f}% de los delitos.")
    lines.append("")

    # Acuerdo prediccion vs etiqueta real (global y solo en test, mas honesto)
    if "nivel_true" in gdf.columns:
        acc_global = (gdf["nivel_pred"] == gdf["nivel_true"]).mean() * 100
        lines.append(f"Acuerdo con la etiqueta real (todos los tiles): {acc_global:.1f}%")
        sub_test = gdf[gdf["split"] == "test"]
        if len(sub_test):
            acc_test = (sub_test["nivel_pred"] == sub_test["nivel_true"]).mean() * 100
            lines.append(f"Acuerdo en el conjunto de TEST (sin fuga): {acc_test:.1f}%  "
                         f"(n={len(sub_test)})")
        lines.append("(El acuerdo global es optimista porque incluye tiles de entrenamiento.)")

    reporte = ZON_DIR / "coincidencia_reporte.txt"
    reporte.write_text("\n".join(lines))
    print()
    for ln in lines:
        print("  " + ln)
    print(f"\n  Reporte: {reporte}")
    print(f"\n  SIGUIENTE PASO: abrir tiles_predicciones.geojson en QGIS y comparar")
    print(f"  con los hotspots historicos del MININTER.")
    print("=" * 60)
