#!/usr/bin/env python3
"""
02_pansharpening.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Aplica refinamiento pancromatico a pares de imagenes PeruSAT-1 (MS + PAN).
Usa gdal_pansharpen.py de GDAL (instalado via conda-forge).

Entrada:
    data/raw/imagenes_perusat/ESPECTRAL/IMG_PER1_..._MS_*.TIF
    data/raw/imagenes_perusat/PANCROMATICA/IMG_PER1_..._P_*.TIF

Salida:
    data/processed/imagenes_pansharpened/IMG_PER1_..._PS_*.TIF

Uso:
    python scripts/02_pansharpening.py
"""

import sys
import yaml
import subprocess
from pathlib import Path
from datetime import datetime


# Cargar configuracion
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MS_DIR = PROJECT_ROOT / CFG["paths"]["raw"]["imagenes_ms"]
PAN_DIR = PROJECT_ROOT / CFG["paths"]["raw"]["imagenes_pan"]
OUTPUT_DIR = PROJECT_ROOT / CFG["paths"]["processed"]["pansharpened"]

RESAMPLING = CFG["pansharpening"]["resampling"]
COMPRESS = CFG["pansharpening"]["compress"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extraer_clave(nombre_archivo):
    """
    Extrae clave para emparejar MS con PAN.
    IMG_PER1_20200216152934_ORT_MS_001233.TIF -> 20200216152934_ORT_001233
    IMG_PER1_20200216152934_ORT_P_001233.TIF  -> 20200216152934_ORT_001233
    """
    partes = nombre_archivo.replace(".TIF", "").split("_")
    if len(partes) >= 6:
        return f"{partes[2]}_{partes[3]}_{partes[5]}"
    return None


def buscar_pares(ms_dir, pan_dir):
    ms_files = {extraer_clave(f.name): f for f in ms_dir.glob("*.TIF")}
    pan_files = {extraer_clave(f.name): f for f in pan_dir.glob("*.TIF")}

    pares = []
    sin_par_ms = []

    for clave, ms_path in sorted(ms_files.items()):
        if clave in pan_files:
            pares.append((ms_path, pan_files[clave]))
        else:
            sin_par_ms.append(ms_path)

    return pares, sin_par_ms


def verificar_gdal():
    try:
        subprocess.run(
            ["gdal_pansharpen.py", "--help"],
            capture_output=True, text=True, timeout=10
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ejecutar_pansharpening(ms_path, pan_path, output_path):
    cmd = [
        "gdal_pansharpen.py",
        str(pan_path),
        str(ms_path),
        str(output_path),
        "-r", RESAMPLING,
        "-co", f"COMPRESS={COMPRESS}",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=256",
        "-co", "BLOCKYSIZE=256",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            return True
        print(f"    Error GDAL: {result.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"    Timeout (30 min)")
        return False


if __name__ == "__main__":

    print("=" * 60)
    print("  REFINAMIENTO PANCROMATICO - PeruSAT-1")
    print(f"  Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    if not verificar_gdal():
        print("  ERROR: gdal_pansharpen.py no disponible.")
        print("  Activa el environment conda: conda activate tesis")
        sys.exit(1)

    if not MS_DIR.exists() or not PAN_DIR.exists():
        print(f"  ERROR: No se encontraron carpetas de imagenes.")
        print(f"    MS:  {MS_DIR}")
        print(f"    PAN: {PAN_DIR}")
        sys.exit(1)

    print(f"  MS:     {MS_DIR}")
    print(f"  PAN:    {PAN_DIR}")
    print(f"  Output: {OUTPUT_DIR}")

    pares, sin_par = buscar_pares(MS_DIR, PAN_DIR)

    print(f"\n  Pares MS-PAN encontrados: {len(pares)}")
    print(f"  MS sin par PAN: {len(sin_par)}")

    if sin_par:
        print(f"\n  Excluidas (sin PAN):")
        for ms in sin_par:
            print(f"    - {ms.name}")

    if not pares:
        sys.exit(1)

    exitosos = 0
    fallidos = 0
    log_lines = []

    for i, (ms_path, pan_path) in enumerate(pares, 1):
        out_name = ms_path.name.replace("_MS_", "_PS_")
        out_path = OUTPUT_DIR / out_name

        print(f"\n  [{i}/{len(pares)}] {ms_path.name}")

        if out_path.exists():
            size_mb = out_path.stat().st_size / (1024 ** 2)
            print(f"    Ya existe ({size_mb:.0f} MB). Saltando.")
            exitosos += 1
            log_lines.append(f"SKIP,{ms_path.name},{out_name}")
            continue

        print(f"    Procesando...")
        ok = ejecutar_pansharpening(ms_path, pan_path, out_path)

        if ok and out_path.exists():
            size_mb = out_path.stat().st_size / (1024 ** 2)
            print(f"    OK ({size_mb:.0f} MB)")
            exitosos += 1
            log_lines.append(f"OK,{ms_path.name},{out_name},{size_mb:.0f}MB")
        else:
            print(f"    FALLO")
            fallidos += 1
            log_lines.append(f"FAIL,{ms_path.name},{out_name}")

    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"{'='*60}")
    print(f"  Pares procesados: {len(pares)}")
    print(f"  Exitosos:         {exitosos}")
    print(f"  Fallidos:         {fallidos}")
    print(f"  Excluidos sin PAN: {len(sin_par)}")
    print(f"  Salida en:        {OUTPUT_DIR}")

    log_path = OUTPUT_DIR / "pansharpening_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Fecha: {datetime.now().isoformat()}\n")
        f.write(f"Pares: {len(pares)}, Exitosos: {exitosos}, Fallidos: {fallidos}\n\n")
        for line in log_lines:
            f.write(line + "\n")
    print(f"  Log: {log_path}")

    print(f"\n  SIGUIENTE PASO: python scripts/03_generar_tiles.py")
    print(f"{'='*60}")
