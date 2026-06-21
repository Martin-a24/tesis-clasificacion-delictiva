#!/usr/bin/env python3
"""
09_gradcam.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Objetivo 3 (R6): genera mapas de activacion Grad-CAM para los tiles del conjunto
de prueba, resaltando las regiones de la imagen mas relevantes para la
clasificacion del modelo. Sirve de insumo para el analisis de coherencia con la
criminologia ambiental (R7).

Grad-CAM se implementa con hooks de PyTorch (sin dependencias extra). La capa
objetivo es la ultima etapa convolucional (resnet: layer4; efficientnet:
features[-1]).

Entrada:
    models/{architecture}_best.pth   (modelo entrenado en 06)
    data/splits/test.csv             (split de 05)
    data/processed/tiles/*.tif

Salida:
    results/gradcam/<categoria>/<tile>__true-<t>_pred-<p>.png   (overlay por tile)
    results/gradcam/montage_<categoria>.png                     (muestras representativas)
    results/gradcam/gradcam_resumen.csv                         (apoyo cuantitativo R7)

Uso:
    python scripts/09_gradcam.py
"""

import sys
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    import rasterio  # noqa: F401  (lo usa el dataset reusado de 06)
except ImportError as e:
    print(f"Error: dependencias no instaladas: {e}")
    print("  conda activate tesis")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Reusar el dataset, el constructor de modelo y la config de 06 (misma
# normalizacion y mismas clases), cargando el modulo aunque empiece con digito.
_spec = importlib.util.spec_from_file_location("entrenar06", SCRIPTS_DIR / "06_entrenar_modelo.py")
ent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ent)

TILES_DIR = ent.TILES_DIR
SPLITS_DIR = ent.SPLITS_DIR
MODELS_DIR = ent.MODELS_DIR
RESULTS_DIR = ent.RESULTS_DIR
NIVELES = ent.NIVELES
IDX_A_NIVEL = ent.IDX_A_NIVEL
ARCHITECTURE = ent.ARCHITECTURE

GRADCAM_DIR = RESULTS_DIR / "gradcam"
MAX_MONTAJE = 12   # tiles por montaje de categoria


def cargar_modelo(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ck.get("architecture", ARCHITECTURE)
    inch = ck.get("in_channels", 4)
    ncl = ck.get("num_classes", len(NIVELES))
    niveles = ck.get("niveles", NIVELES)
    # El state_dict puede tener (o no) dropout en la cabeza; probamos opciones
    # hasta que las claves coincidan.
    ultimo_err = None
    for d in [ent.DROPOUT, 0.0, 0.2, 0.5]:
        model = ent.construir_modelo(arch, ncl, inch, False, d)
        try:
            model.load_state_dict(ck["model_state_dict"], strict=True)
            model.to(device).eval()
            return model, arch, niveles
        except RuntimeError as e:
            ultimo_err = e
    raise RuntimeError(f"No se pudo cargar el modelo: {ultimo_err}")


def capa_objetivo(model, arch):
    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"Grad-CAM no soportado para '{arch}' (usa resnet o efficientnet). "
                     f"ViT requiere otra tecnica (attention rollout).")


class GradCAM:
    """Grad-CAM con hooks sobre una capa convolucional."""
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, clase=None):
        """Grad-CAM para 'clase'; si es None usa la clase predicha. Una sola pasada."""
        self.model.zero_grad()
        out = self.model(x)                      # (1, num_classes)
        probs = out.softmax(dim=1)[0]
        if clase is None:
            clase = int(probs.argmax().item())
        out[0, clase].backward()
        pesos = self.grads.mean(dim=(2, 3), keepdim=True)      # GAP de gradientes
        cam = torch.relu((pesos * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), probs.detach().cpu().numpy(), clase


def rgb_para_mostrar(img_tensor):
    """De (4,H,W) B,G,R,NIR a RGB 0-1 con estiramiento de percentiles 2-98."""
    arr = img_tensor.numpy()
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)   # R,G,B
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def ratio_centro(cam):
    """Razon activacion media en el centro (50%) vs total. >1 = activa al centro."""
    h, w = cam.shape
    centro = cam[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float(centro.mean() / (cam.mean() + 1e-8))


def guardar_overlay(rgb, cam, titulo, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    axes[0].imshow(rgb); axes[0].set_title("Imagen (RGB)"); axes[0].axis("off")
    axes[1].imshow(rgb); axes[1].imshow(cam, cmap="jet", alpha=0.5)
    axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    fig.suptitle(titulo, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()


def guardar_montaje(items, categoria, path):
    """items: lista de (rgb, cam, subtitulo). Grilla de overlays."""
    if not items:
        return
    n = len(items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax, (rgb, cam, sub) in zip(axes, items):
        ax.imshow(rgb); ax.imshow(cam, cmap="jet", alpha=0.5)
        ax.set_title(sub, fontsize=8); ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Grad-CAM - categoria '{categoria}' ({n} muestras)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  GRAD-CAM (Objetivo 3 - R6)")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    device = torch.device(ent.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")

    ckpt = MODELS_DIR / f"{ARCHITECTURE}_best.pth"
    if not ckpt.exists():
        print(f"\n  ERROR: no existe el modelo {ckpt}")
        print(f"  Entrena primero: python scripts/06_entrenar_modelo.py")
        sys.exit(1)

    test_csv = SPLITS_DIR / "test.csv"
    if not test_csv.exists():
        print(f"\n  ERROR: no existe {test_csv} (corre 05_construir_splits.py)")
        sys.exit(1)

    print(f"\n  [1/3] Cargando modelo {ARCHITECTURE}...")
    model, arch, niveles = cargar_modelo(ckpt, device)
    cam_engine = GradCAM(model, capa_objetivo(model, arch))

    df_test = pd.read_csv(test_csv)
    dataset = ent.TilesDataset(df_test, TILES_DIR, transform=None)
    print(f"  Tiles de prueba: {len(dataset)}")

    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    for niv in NIVELES:
        (GRADCAM_DIR / niv).mkdir(exist_ok=True)

    print(f"\n  [2/3] Generando Grad-CAM por tile...")
    filas = []
    muestras = {niv: [] for niv in NIVELES}   # para montajes por categoria real
    for i in range(len(dataset)):
        img, label = dataset[i]
        true_niv = IDX_A_NIVEL[int(label)]
        x = img.unsqueeze(0).to(device)
        cam, probs, pred_idx = cam_engine(x)          # apunta a la clase predicha
        pred_niv = IDX_A_NIVEL[pred_idx]

        rgb = rgb_para_mostrar(img)
        tile_name = Path(df_test.iloc[i]["tile_name"]).stem
        marca = "ok" if pred_niv == true_niv else "x"
        titulo = f"{tile_name}  true={true_niv}  pred={pred_niv} ({probs[pred_idx]:.2f}) [{marca}]"
        out_png = GRADCAM_DIR / true_niv / f"{tile_name}__true-{true_niv}_pred-{pred_niv}.png"
        guardar_overlay(rgb, cam, titulo, out_png)

        if len(muestras[true_niv]) < MAX_MONTAJE:
            muestras[true_niv].append((rgb, cam, f"pred={pred_niv} ({probs[pred_idx]:.2f})"))

        filas.append({
            "tile_name": df_test.iloc[i]["tile_name"],
            "nivel_true": true_niv,
            "nivel_pred": pred_niv,
            "prob_pred": round(float(probs[pred_idx]), 4),
            "correcto": int(pred_niv == true_niv),
            "act_centro_ratio": round(ratio_centro(cam), 3),
        })
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(dataset)} tiles...")

    print(f"\n  [3/3] Guardando montajes y resumen...")
    for niv in NIVELES:
        guardar_montaje(muestras[niv], niv, GRADCAM_DIR / f"montage_{niv}.png")

    resumen = pd.DataFrame(filas)
    resumen.to_csv(GRADCAM_DIR / "gradcam_resumen.csv", index=False)

    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Tiles procesados: {len(resumen)}")
    print(f"  Accuracy en test: {resumen['correcto'].mean() * 100:.1f}%")
    print("  Activacion al centro (ratio medio) por categoria real:")
    for niv in NIVELES:
        sub = resumen[resumen["nivel_true"] == niv]
        if len(sub):
            print(f"    {niv:<7}: {sub['act_centro_ratio'].mean():.2f}  (n={len(sub)})")
    print(f"\n  Overlays por tile:  {GRADCAM_DIR}/<categoria>/")
    print(f"  Montajes:           {GRADCAM_DIR}/montage_<categoria>.png")
    print(f"  Resumen (R7):       {GRADCAM_DIR}/gradcam_resumen.csv")
    print(f"\n  SIGUIENTE PASO: analisis de coherencia con criminologia ambiental (R7)")
    print("=" * 60)
