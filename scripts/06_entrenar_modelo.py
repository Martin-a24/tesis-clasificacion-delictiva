#!/usr/bin/env python3
"""
06_entrenar_modelo.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Entrena un modelo CNN para clasificar tiles satelitales por nivel de riesgo
delictivo (alto, medio, bajo). Soporta multiples arquitecturas via config.yaml.

Caracteristicas:
- Arquitecturas soportadas: ResNet-18, ResNet-50, EfficientNet-B0, ViT-B/16
- Adaptacion automatica para 4 bandas (PeruSAT-1) en lugar de 3 (RGB ImageNet)
- Transfer learning desde modelos pre-entrenados
- Class weights para manejo de desbalance
- Data augmentation
- Early stopping
- Logging completo de metricas por epoca

Entrada:
    data/splits/train.csv
    data/splits/val.csv
    data/processed/tiles/*.tif

Salida:
    models/{architecture}_best.pth          (mejor modelo segun F1 en val)
    models/{architecture}_final.pth         (modelo de la ultima epoca)
    results/training_log_{timestamp}.csv    (metricas por epoca)
    results/training_curves_{timestamp}.png (curvas de entrenamiento)
"""

import sys
import yaml
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    from torchvision import transforms
    import rasterio
    from sklearn.metrics import f1_score, accuracy_score
except ImportError as e:
    print(f"Error: dependencias no instaladas: {e}")
    print("  conda activate tesis")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
PRETRAINED = CFG["modelo"]["pretrained"]
IN_CHANNELS = CFG["modelo"]["in_channels"]
NUM_CLASSES = CFG["modelo"]["num_classes"]

BATCH_SIZE = CFG["entrenamiento"]["batch_size"]
NUM_EPOCHS = CFG["entrenamiento"]["num_epochs"]
LEARNING_RATE = CFG["entrenamiento"]["learning_rate"]
WEIGHT_DECAY = CFG["entrenamiento"]["weight_decay"]
OPTIMIZER = CFG["entrenamiento"]["optimizer"]
USE_SCHEDULER = CFG["entrenamiento"]["use_scheduler"]
SCHEDULER_STEP = CFG["entrenamiento"]["scheduler_step_size"]
SCHEDULER_GAMMA = CFG["entrenamiento"]["scheduler_gamma"]
USE_EARLY_STOP = CFG["entrenamiento"]["use_early_stopping"]
EARLY_STOP_PATIENCE = CFG["entrenamiento"]["early_stopping_patience"]
USE_CLASS_WEIGHTS = CFG["entrenamiento"]["use_class_weights"]
USE_FOCAL_LOSS = CFG["entrenamiento"]["use_focal_loss"]
USE_AUGMENTATION = CFG["entrenamiento"]["use_augmentation"]
RANDOM_SEED = CFG["entrenamiento"]["random_seed"]
DEVICE = CFG["entrenamiento"]["device"]
NUM_WORKERS = CFG["entrenamiento"]["num_workers"]
PIN_MEMORY = CFG["entrenamiento"]["pin_memory"]

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def fijar_semillas(seed):
    """Fija las semillas para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET PYTORCH
# ============================================================

class TilesDataset(Dataset):
    """
    Dataset PyTorch que carga tiles satelitales desde GeoTIFF.

    Cada tile tiene 4 bandas (B, G, R, NIR) a 0.7 m/px.
    Las etiquetas se mapean a indices: bajo=0, medio=1, alto=2.
    """

    def __init__(self, df, tiles_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.tiles_dir = Path(tiles_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tile_path = self.tiles_dir / row["tile_name"]

        # Leer tile multispectral
        with rasterio.open(tile_path) as src:
            image = src.read().astype(np.float32)  # shape: (4, H, W)

        # Normalizar a [0, 1]: PeruSAT-1 usa valores de 11 bits (0-2047)
        # Se asume que los valores estan en este rango
        image = image / 2047.0
        image = np.clip(image, 0.0, 1.0)

        # Convertir a tensor PyTorch
        image = torch.from_numpy(image)

        # Aplicar transformaciones (augmentation)
        if self.transform is not None:
            image = self.transform(image)

        # Etiqueta como indice
        label = NIVEL_A_IDX[row["nivel_riesgo"]]
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def construir_transformaciones(use_augmentation):
    """
    Construye las transformaciones de data augmentation.
    Solo se aplican en train; val y test sin augmentation.
    """
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])
    else:
        train_transform = None

    val_transform = None  # sin augmentation en validacion

    return train_transform, val_transform


# ============================================================
# CONSTRUCCION DE MODELOS
# ============================================================

def construir_modelo(arquitectura, num_classes, in_channels, pretrained):
    """
    Construye el modelo CNN segun la arquitectura especificada.
    Adapta la primera capa convolucional para soportar 4 bandas
    en lugar de 3 (RGB).
    """
    if arquitectura == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model = adaptar_primera_capa_resnet(model, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arquitectura == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model = adaptar_primera_capa_resnet(model, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arquitectura == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model = adaptar_primera_capa_efficientnet(model, in_channels)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif arquitectura == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model = adaptar_primera_capa_vit(model, in_channels)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    else:
        raise ValueError(f"Arquitectura no soportada: {arquitectura}. "
                          f"Opciones: resnet18, resnet50, efficientnet_b0, vit_b_16")

    return model


def adaptar_primera_capa_resnet(model, in_channels):
    """
    Adapta la primera capa conv de ResNet para in_channels bandas.
    Inicializa los pesos de las nuevas bandas como promedio de los pesos RGB.
    """
    original_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )

    with torch.no_grad():
        if in_channels >= 3:
            # Copiar pesos RGB y replicar el promedio para las bandas extra
            new_conv.weight[:, :3] = original_conv.weight
            mean_weights = original_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, in_channels):
                new_conv.weight[:, i:i+1] = mean_weights
        else:
            new_conv.weight[:, :in_channels] = original_conv.weight[:, :in_channels]

    model.conv1 = new_conv
    return model


def adaptar_primera_capa_efficientnet(model, in_channels):
    """Adapta la primera capa de EfficientNet."""
    original_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )

    with torch.no_grad():
        if in_channels >= 3:
            new_conv.weight[:, :3] = original_conv.weight
            mean_weights = original_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, in_channels):
                new_conv.weight[:, i:i+1] = mean_weights
        else:
            new_conv.weight[:, :in_channels] = original_conv.weight[:, :in_channels]

    model.features[0][0] = new_conv
    return model


def adaptar_primera_capa_vit(model, in_channels):
    """Adapta la primera capa de Vision Transformer."""
    original_conv = model.conv_proj
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )

    with torch.no_grad():
        if in_channels >= 3:
            new_conv.weight[:, :3] = original_conv.weight
            mean_weights = original_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, in_channels):
                new_conv.weight[:, i:i+1] = mean_weights
        else:
            new_conv.weight[:, :in_channels] = original_conv.weight[:, :in_channels]

    model.conv_proj = new_conv
    return model


# ============================================================
# FUNCIONES DE PERDIDA
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss para manejo de desbalance de clases.
    Penaliza mas a las muestras dificiles.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def calcular_class_weights(df, niveles):
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.
    Util para CrossEntropyLoss en datasets desbalanceados.
    """
    counts = df["nivel_riesgo"].value_counts()
    total = len(df)
    weights = []
    for nivel in niveles:
        n = counts.get(nivel, 1)
        weights.append(total / (len(niveles) * n))
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# ENTRENAMIENTO Y VALIDACION POR EPOCA
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena el modelo durante una epoca."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted


def validate_epoch(model, loader, criterion, device):
    """Evalua el modelo en validacion."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_weighted


# ============================================================
# VISUALIZACION DE RESULTADOS
# ============================================================

def graficar_curvas_entrenamiento(history, output_path):
    """Grafica las curvas de loss, accuracy y F1 a lo largo del entrenamiento."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2c3e50")
    axes[0].plot(epochs, history["val_loss"], label="Validation", color="#e74c3c")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Curva de perdida")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", color="#2c3e50")
    axes[1].plot(epochs, history["val_acc"], label="Validation", color="#e74c3c")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Curva de accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # F1 macro
    axes[2].plot(epochs, history["train_f1"], label="Train", color="#2c3e50")
    axes[2].plot(epochs, history["val_f1"], label="Validation", color="#e74c3c")
    axes[2].set_xlabel("Epoca")
    axes[2].set_ylabel("F1 macro")
    axes[2].set_title("Curva de F1 macro")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  ENTRENAMIENTO DE MODELO CNN")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Arquitectura: {ARCHITECTURE}")
    print(f"  Pretrained: {PRETRAINED}")
    print("=" * 60)

    # Reproducibilidad
    fijar_semillas(RANDOM_SEED)

    # Verificar dispositivo
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  ADVERTENCIA: Entrenando en CPU sera muy lento.")

    # Cargar splits
    print(f"\n  [1/6] Cargando splits...")
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        print(f"  ERROR: No existen los splits. Ejecuta primero:")
        print(f"  python scripts/05_construir_splits.py")
        sys.exit(1)

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"  Train: {len(df_train)} tiles")
    print(f"  Val:   {len(df_val)} tiles")

    # Construir datasets y dataloaders
    print(f"\n  [2/6] Construyendo dataloaders...")
    train_transform, val_transform = construir_transformaciones(USE_AUGMENTATION)

    train_dataset = TilesDataset(df_train, TILES_DIR, transform=train_transform)
    val_dataset = TilesDataset(df_val, TILES_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Construir modelo
    print(f"\n  [3/6] Construyendo modelo {ARCHITECTURE}...")
    model = construir_modelo(ARCHITECTURE, NUM_CLASSES, IN_CHANNELS, PRETRAINED)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametros totales: {n_params:,}")
    print(f"  Parametros entrenables: {n_trainable:,}")

    # Configurar funcion de perdida
    print(f"\n  [4/6] Configurando funcion de perdida...")
    if USE_CLASS_WEIGHTS:
        class_weights = calcular_class_weights(df_train, NIVELES).to(device)
        print(f"  Class weights: {dict(zip(NIVELES, class_weights.cpu().numpy().round(3)))}")
    else:
        class_weights = None

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print(f"  Funcion de perdida: Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Funcion de perdida: CrossEntropyLoss" +
               (" (ponderada)" if USE_CLASS_WEIGHTS else ""))

    # Configurar optimizador
    if OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(),
                                lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Optimizador no soportado: {OPTIMIZER}")
    print(f"  Optimizador: {OPTIMIZER} (lr={LEARNING_RATE})")

    # Scheduler
    scheduler = None
    if USE_SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA
        )
        print(f"  Scheduler: StepLR (step={SCHEDULER_STEP}, gamma={SCHEDULER_GAMMA})")

    # Entrenamiento
    print(f"\n  [5/6] Iniciando entrenamiento ({NUM_EPOCHS} epocas)...")
    print(f"  Early stopping: {USE_EARLY_STOP} (paciencia={EARLY_STOP_PATIENCE})\n")

    history = {
        "epoch": [], "train_loss": [], "train_acc": [], "train_f1": [], "train_f1_w": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_f1_w": [], "lr": []
    }

    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_model_path = MODELS_DIR / f"{ARCHITECTURE}_best.pth"

    print(f"  {'Epoca':<6} {'TrainLoss':>10} {'TrainF1':>9} {'ValLoss':>10} {'ValF1':>9} {'ValAcc':>8}  {'LR':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*9} {'-'*10} {'-'*9} {'-'*8}  {'-'*10}")

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1, train_f1_w = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, val_f1_w = validate_epoch(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["train_f1_w"].append(train_f1_w)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_f1_w"].append(val_f1_w)
        history["lr"].append(current_lr)

        marca = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "architecture": ARCHITECTURE,
                "in_channels": IN_CHANNELS,
                "num_classes": NUM_CLASSES,
                "niveles": NIVELES,
            }, best_model_path)
            marca = " *"
        else:
            epochs_no_improve += 1

        print(f"  {epoch:<6} {train_loss:>10.4f} {train_f1:>9.4f} "
               f"{val_loss:>10.4f} {val_f1:>9.4f} {val_acc:>8.4f}  {current_lr:>10.2e}{marca}")

        if scheduler is not None:
            scheduler.step()

        if USE_EARLY_STOP and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping en epoca {epoch} (sin mejora en {EARLY_STOP_PATIENCE} epocas)")
            break

    total_time = time.time() - start_time
    print(f"\n  Tiempo total: {total_time / 60:.1f} minutos")
    print(f"  Mejor F1 (val): {best_val_f1:.4f}")
    print(f"  Modelo guardado: {best_model_path}")

    # Guardar modelo final
    final_model_path = MODELS_DIR / f"{ARCHITECTURE}_final.pth"
    torch.save({
        "epoch": history["epoch"][-1],
        "model_state_dict": model.state_dict(),
        "architecture": ARCHITECTURE,
        "in_channels": IN_CHANNELS,
        "num_classes": NUM_CLASSES,
        "niveles": NIVELES,
    }, final_model_path)

    # Guardar log y curvas
    print(f"\n  [6/6] Guardando logs y curvas...")
    log_path = RESULTS_DIR / f"training_log_{ARCHITECTURE}_{timestamp}.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"  Log: {log_path}")

    curvas_path = RESULTS_DIR / f"training_curves_{ARCHITECTURE}_{timestamp}.png"
    graficar_curvas_entrenamiento(history, curvas_path)
    print(f"  Curvas: {curvas_path}")

    print(f"\n  SIGUIENTE PASO: python scripts/07_evaluar_modelo.py")
    print(f"{'='*60}")
