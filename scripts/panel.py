#!/usr/bin/env python3
"""
panel.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Panel de control en terminal para el proyecto. Permite:
  - Ver el estado del pipeline (que paso esta hecho, que falta).
  - Correr cualquier paso (o todo el pipeline).
  - Cambiar parametros comunes y dejarlos como DEFAULT en configs/config.yaml.
  - Ver que input necesita cada paso y como agregar datos (imagenes, delitos).
  - Atajos para sincronizar con el servidor y limpiar salidas.

Uso:
    python scripts/panel.py            # menu interactivo
    python scripts/panel.py status     # solo imprime el estado y sale
    python scripts/panel.py run 4      # corre el paso 4 y sale
    python scripts/panel.py inputs     # imprime la guia de inputs y sale
"""

import os
import re
import sys
import glob
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Falta PyYAML. Activa el entorno: conda activate tesis")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ----- colores simples (se desactivan con NO_COLOR o si no es TTY) -----
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def c(texto, color):
    if not _USE_COLOR:
        return texto
    codes = {"verde": "32", "amarillo": "33", "rojo": "31",
             "cian": "36", "gris": "90", "bold": "1"}
    return f"\033[{codes.get(color, '0')}m{texto}\033[0m"


# ============================================================
# DEFINICION DEL PIPELINE
# ============================================================
# Cada paso: salida(s) que indican "hecho" y input(s) que necesita.
# Las rutas son relativas a la raiz del proyecto; soportan comodines (*).

def pipeline(cfg):
    usa_mosaico = cfg.get("mosaico", {}).get("enabled", False)
    return [
        {"id": "1", "label": "Limpiar delitos", "script": "01_limpiar_datos_delictivos.py",
         "out": ["data/processed/delitos_limpios/delitos_lima_limpio.geojson"],
         "needs": ["data/raw/delitos/*.csv"],
         "needs_desc": "CSVs del MININTER en data/raw/delitos/"},
        {"id": "2", "label": "Pansharpening", "script": "02_pansharpening.py",
         "out": ["data/processed/imagenes_pansharpened/*PS*.TIF",
                 "data/processed/imagenes_pansharpened/*.tif"],
         "needs": ["data/raw/imagenes_perusat/ESPECTRAL/*MS*.TIF",
                   "data/raw/imagenes_perusat/PANCROMATICA/*.TIF"],
         "needs_desc": "pares MS + PAN en data/raw/imagenes_perusat/{ESPECTRAL,PANCROMATICA}/"},
        {"id": "2b", "label": "Mosaico VRT", "script": "02b_construir_mosaico.py",
         "out": ["data/processed/mosaico/mosaico.vrt"],
         "needs": ["data/processed/imagenes_pansharpened/*PS*.TIF",
                   "data/processed/imagenes_pansharpened/*.tif"],
         "needs_desc": "imagenes pansharpened (paso 2)"},
        {"id": "3", "label": "Generar tiles", "script": "03_generar_tiles.py",
         "out": ["data/processed/tiles/tiles_metadata.csv"],
         "needs": (["data/processed/mosaico/mosaico.vrt"] if usa_mosaico
                   else ["data/processed/imagenes_pansharpened/*PS*.TIF"]),
         "needs_desc": ("mosaico.vrt (paso 2b)" if usa_mosaico else "pansharpened (paso 2)")
                       + " + data/raw/limites/lima_metropolitana.geojson"},
        {"id": "4", "label": "Etiquetar tiles", "script": "04_etiquetar_tiles.py",
         "out": ["data/labels/tiles_labeled.csv"],
         "needs": ["data/processed/tiles/tiles_metadata.csv",
                   "data/processed/delitos_limpios/delitos_lima_limpio.geojson"],
         "needs_desc": "tiles (paso 3) + delitos (paso 1) + limites de Lima"},
        {"id": "5", "label": "Construir splits", "script": "05_construir_splits.py",
         "out": ["data/splits/train.csv"],
         "needs": ["data/labels/tiles_labeled.csv"],
         "needs_desc": "tiles etiquetados (paso 4)"},
        {"id": "6", "label": "Entrenar modelo", "script": "06_entrenar_modelo.py",
         "out": ["models/*_best.pth"],
         "needs": ["data/splits/train.csv", "data/splits/val.csv"],
         "needs_desc": "splits train/val (paso 5) + tiles"},
        {"id": "7", "label": "Evaluar modelo", "script": "07_evaluar_modelo.py",
         "out": ["results/metrics_*.json"],
         "needs": ["models/*_best.pth", "data/splits/test.csv"],
         "needs_desc": "modelo entrenado (paso 6) + split test"},
        {"id": "8", "label": "Comparar arquitecturas", "script": "08_comparar_arquitecturas.py",
         "out": ["results/comparisons/comparison_*.csv"],
         "needs": ["data/splits/train.csv"],
         "needs_desc": "splits (paso 5); entrena varias arquitecturas"},
        {"id": "9", "label": "Grad-CAM (interpretab.)", "script": "09_gradcam.py",
         "out": ["results/gradcam/gradcam_resumen.csv"],
         "needs": ["models/*_best.pth", "data/splits/test.csv"],
         "needs_desc": "modelo entrenado (paso 6) + split test (Objetivo 3 / R6)"},
    ]


def _existe_alguno(patrones):
    for p in patrones:
        if glob.glob(str(PROJECT_ROOT / p)):
            return True
    return False


def estado_paso(paso):
    """Devuelve 'hecho' | 'listo' | 'bloqueado'."""
    if _existe_alguno(paso["out"]):
        return "hecho"
    if paso["needs"] is None or _existe_alguno(paso["needs"]):
        return "listo"
    return "bloqueado"


# ============================================================
# ESTADO
# ============================================================

def rama_git():
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           cwd=PROJECT_ROOT, capture_output=True, text=True)
        return r.stdout.strip() or "?"
    except Exception:
        return "?"


def imprimir_estado(cfg):
    print(c("=" * 60, "gris"))
    print(c("  PANEL DE CONTROL - Tesis Clasificacion Delictiva", "bold"))
    print(c("=" * 60, "gris"))
    env = os.environ.get("CONDA_DEFAULT_ENV", "?")
    print(f"  rama git: {c(rama_git(),'cian')}   entorno conda: {c(env,'cian')}   "
          f"mosaico: {'on' if cfg.get('mosaico',{}).get('enabled') else 'off'}")
    print()
    print(c("  PIPELINE", "bold"))
    marca = {"hecho": c("[OK]", "verde"),
             "listo": c("[->]", "amarillo"),
             "bloqueado": c("[--]", "rojo")}
    for p in pipeline(cfg):
        est = estado_paso(p)
        print(f"   {marca[est]} {p['id']:>2}. {p['label']:<24} {c(p['script'],'gris')}")
    print()
    print(f"   {c('[OK]','verde')} hecho   {c('[->]','amarillo')} listo para correr   "
          f"{c('[--]','rojo')} falta input")


# ============================================================
# CORRER PASOS
# ============================================================

def correr_paso(paso, cfg):
    est = estado_paso(paso)
    if est == "bloqueado":
        print(c(f"\n  No se puede correr el paso {paso['id']}: falta input.", "rojo"))
        print(f"  Necesita: {paso['needs_desc']}")
        return False
    script = SCRIPTS_DIR / paso["script"]
    print(c(f"\n  >>> Corriendo paso {paso['id']}: {paso['label']}  ({paso['script']})\n", "cian"))
    r = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    ok = r.returncode == 0
    print(c(f"\n  {'OK' if ok else 'FALLO'} paso {paso['id']} (codigo {r.returncode})",
            "verde" if ok else "rojo"))
    return ok


def correr_paso_por_id(idp, cfg):
    for p in pipeline(cfg):
        if p["id"] == str(idp):
            return correr_paso(p, cfg)
    print(c(f"  Paso '{idp}' no existe.", "rojo"))
    return False


def correr_todo(cfg):
    print(c("\n  Correra todos los pasos en orden, deteniendose si alguno falla.", "amarillo"))
    if input("  Confirmas? [s/N]: ").strip().lower() not in ("s", "si", "y"):
        print("  Cancelado.")
        return
    for p in pipeline(cfg):
        if not correr_paso(p, cfg):
            print(c(f"\n  Pipeline detenido en el paso {p['id']}.", "rojo"))
            return
    print(c("\n  Pipeline completo.", "verde"))


# ============================================================
# CONFIGURACION (editar config.yaml conservando comentarios)
# ============================================================

PARAMS = [
    {"sec": "mosaico", "key": "enabled", "tipo": "bool",
     "desc": "Usar mosaico VRT (recomendado para evitar tiles duplicados)"},
    {"sec": "tiles", "key": "min_urban_overlap", "tipo": "float",
     "desc": "Minimo solape urbano por tile (0-1)"},
    {"sec": "tiles", "key": "num_workers", "tipo": "int",
     "desc": "Procesos para teselado (0 = todos los CPU)"},
    {"sec": "etiquetado", "key": "percentil_bajo", "tipo": "int",
     "desc": "Percentil de corte bajo/medio"},
    {"sec": "etiquetado", "key": "percentil_alto", "tipo": "int",
     "desc": "Percentil de corte medio/alto"},
    {"sec": "modelo", "key": "architecture", "tipo": "choice",
     "choices": ["resnet18", "resnet50", "efficientnet_b0", "vit_b_16"],
     "desc": "Arquitectura CNN"},
    {"sec": "modelo", "key": "dropout", "tipo": "float",
     "desc": "Dropout antes de la capa final (0 = sin dropout)"},
    {"sec": "entrenamiento", "key": "batch_size", "tipo": "int", "desc": "Batch size"},
    {"sec": "entrenamiento", "key": "num_epochs", "tipo": "int", "desc": "Numero de epocas"},
    {"sec": "entrenamiento", "key": "learning_rate", "tipo": "float", "desc": "Learning rate"},
    {"sec": "entrenamiento", "key": "weight_decay", "tipo": "float", "desc": "Weight decay (regularizacion)"},
    {"sec": "entrenamiento", "key": "optimizer", "tipo": "choice",
     "choices": ["adam", "adamw", "sgd"], "desc": "Optimizador"},
    {"sec": "entrenamiento", "key": "label_smoothing", "tipo": "float",
     "desc": "Label smoothing (util si la etiqueta es ruidosa)"},
    {"sec": "entrenamiento", "key": "device", "tipo": "choice",
     "choices": ["cuda", "cpu"], "desc": "Dispositivo de entrenamiento"},
]


def set_config_value(section, key, raw_value, path=CONFIG_PATH):
    """Reemplaza el valor de section.key en el YAML conservando indentacion y
    comentarios. Devuelve True si lo encontro y escribio."""
    lines = path.read_text().splitlines(keepends=True)
    in_section = False
    for i, line in enumerate(lines):
        sin_nl = line.rstrip("\n")
        if re.match(r"^[A-Za-z0-9_]+:\s*$", sin_nl):           # cabecera de seccion top-level
            in_section = (sin_nl[:-1] == section)
            continue
        if in_section:
            m = re.match(r"^(\s+)" + re.escape(key) + r":\s*(.*?)(\s+#.*)?$", sin_nl)
            if m:
                indent, comentario = m.group(1), (m.group(3) or "")
                lines[i] = f"{indent}{key}: {raw_value}{comentario}\n"
                path.write_text("".join(lines))
                return True
    return False


def _formatear(tipo, valor):
    if tipo == "bool":
        return "true" if valor else "false"
    if tipo in ("int", "float"):
        return str(valor)
    return f'"{valor}"'   # choice / str


def menu_config():
    while True:
        cfg = load_cfg()
        print(c("\n  CONFIGURAR PARAMETROS (quedan como default en config.yaml)", "bold"))
        for n, p in enumerate(PARAMS, 1):
            actual = cfg.get(p["sec"], {}).get(p["key"], "?")
            extra = f"  opciones: {p['choices']}" if p["tipo"] == "choice" else ""
            print(f"   {n:>2}. {p['sec']}.{p['key']:<18} = {c(str(actual),'cian'):<14} "
                  f"{c(p['desc'],'gris')}{extra}")
        print("    0. Volver")
        sel = input("\n  Numero a cambiar: ").strip()
        if sel in ("0", "", "q"):
            return
        if not sel.isdigit() or not (1 <= int(sel) <= len(PARAMS)):
            print(c("  Opcion invalida.", "rojo")); continue
        p = PARAMS[int(sel) - 1]
        nuevo = input(f"  Nuevo valor para {p['sec']}.{p['key']} "
                      f"({p['tipo']}): ").strip()
        try:
            if p["tipo"] == "int":
                val = int(nuevo)
            elif p["tipo"] == "float":
                val = float(nuevo)
            elif p["tipo"] == "bool":
                val = nuevo.lower() in ("true", "1", "si", "s", "yes", "y")
            elif p["tipo"] == "choice":
                if nuevo not in p["choices"]:
                    print(c(f"  Debe ser uno de {p['choices']}.", "rojo")); continue
                val = nuevo
            else:
                val = nuevo
        except ValueError:
            print(c("  Valor invalido para el tipo.", "rojo")); continue
        if set_config_value(p["sec"], p["key"], _formatear(p["tipo"], val)):
            print(c(f"  Guardado: {p['sec']}.{p['key']} = {val}", "verde"))
        else:
            print(c("  No se encontro la clave en config.yaml.", "rojo"))


# ============================================================
# GUIA DE INPUTS Y COMO AGREGAR DATOS
# ============================================================

GUIA = """
  INPUTS POR PASO
  ----------------------------------------------------------
   1 Limpiar delitos     <- data/raw/delitos/*.csv  (ya versionados)
   2 Pansharpening       <- data/raw/imagenes_perusat/ESPECTRAL/*MS*.TIF
                            data/raw/imagenes_perusat/PANCROMATICA/*P*.TIF
   2b Mosaico VRT        <- pansharpened del paso 2
   3 Generar tiles       <- mosaico.vrt (o pansharpened) + limites de Lima
   4 Etiquetar tiles     <- tiles (paso 3) + delitos (paso 1) + limites
   5 Construir splits    <- tiles etiquetados (paso 4)
   6 Entrenar modelo     <- splits train/val (paso 5) + tiles
   7 Evaluar modelo      <- modelo (paso 6) + split test
   8 Comparar arquitec.  <- splits (paso 5)

  COMO AGREGAR / CAMBIAR DATOS
  ----------------------------------------------------------
  + Mas imagenes PeruSAT-1:
      copia los nuevos pares a data/raw/imagenes_perusat/ESPECTRAL/ y
      PANCROMATICA/ (o usa: ./scripts/sync.sh push-imagenes) y re-corre
      2 -> 2b -> 3 -> 4 -> 5. La grilla global mantiene cell_id estable, asi
      que los tiles previos no se re-etiquetan por ubicacion.

  + Mas delitos:
      reemplaza/agrega CSVs en data/raw/delitos/ y re-corre 1 -> 4
      (re-etiqueta con los nuevos registros).

  + Cambiar limites urbanos:
      python scripts/descargar_limites_lima.py  (genera data/raw/limites/)

  + Cambiar parametros (umbrales, arquitectura, batch, etc.):
      opcion 'c' de este panel  (se guardan como default en config.yaml)

  LIMPIAR PARA RE-CORRER
  ----------------------------------------------------------
      python scripts/limpiar_salidas.py            (conserva data/raw/)
      python scripts/limpiar_salidas.py --dry-run  (previsualiza)
"""


def menu_sync():
    print("""
  SINCRONIZAR CON EL SERVIDOR (requiere alias 'phantom' en ~/.ssh/config)
  ----------------------------------------------------------
    ./scripts/sync.sh push-imagenes      PC -> servidor: imagenes
    ./scripts/sync.sh pull-resultados    servidor -> PC: results/
    ./scripts/sync.sh pull-entregables   servidor -> ~/tesis-entregables (QGIS/tesis)
    ./scripts/sync.sh push <ruta> | pull <ruta>
  (Agrega --dry-run para previsualizar.)
""")


# ============================================================
# MENU PRINCIPAL
# ============================================================

def menu():
    while True:
        cfg = load_cfg()
        print()
        imprimir_estado(cfg)
        print(c("\n  ACCIONES", "bold"))
        print("   r N   correr paso N (ej: 'r 4', 'r 2b')      a   correr todo")
        print("   c     configurar parametros (defaults)        i   inputs / como agregar datos")
        print("   t     sincronizar con servidor                l   limpiar salidas")
        print("   q     salir")
        sel = input("\n  > ").strip()
        if sel in ("q", "quit", "salir"):
            return
        elif sel == "c":
            menu_config()
        elif sel == "i":
            print(GUIA)
        elif sel == "t":
            menu_sync()
        elif sel == "l":
            subprocess.run([sys.executable, str(SCRIPTS_DIR / "limpiar_salidas.py")],
                           cwd=PROJECT_ROOT)
        elif sel == "a":
            correr_todo(cfg)
        elif sel.startswith("r"):
            partes = sel.split()
            if len(partes) == 2:
                correr_paso_por_id(partes[1], cfg)
            else:
                print(c("  Uso: r <id>   (ej: 'r 4' o 'r 2b')", "amarillo"))
        else:
            print(c("  Opcion no reconocida.", "amarillo"))


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        try:
            menu()
        except (KeyboardInterrupt, EOFError):
            print("\n  Saliendo.")
    elif args[0] == "status":
        imprimir_estado(load_cfg())
    elif args[0] == "inputs":
        print(GUIA)
    elif args[0] == "run" and len(args) == 2:
        sys.exit(0 if correr_paso_por_id(args[1], load_cfg()) else 1)
    else:
        print("Uso: python scripts/panel.py [status | inputs | run <id>]")
        print("     (sin argumentos = menu interactivo)")
        sys.exit(1)
