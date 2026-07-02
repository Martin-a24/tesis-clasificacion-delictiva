#!/usr/bin/env python3
"""
limpiar_salidas.py
Tesis: Clasificacion de Zonas de Riesgo Delictivo
Autor: Martin Sayago - PUCP

Borra las SALIDAS generadas por el pipeline para poder re-ejecutar los scripts
desde cero. Conserva SIEMPRE los insumos en data/raw/ (delitos, imagenes
PeruSAT-1, limites de Lima, worldpop): esos no se regeneran aqui.

Que borra (vacia el contenido, mantiene la carpeta):
    data/processed/delitos_limpios   (salida de 01)
    data/processed/mosaico           (salida de 02b)
    data/processed/tiles             (salida de 03)
    data/labels                      (salida de 04)
    data/splits                      (salida de 05)
    models                           (salida de 06)
    results                          (salida de 06/07/08)

Que NO borra por defecto (pesado y lento de regenerar):
    data/processed/imagenes_pansharpened  (salida de 02, ~2.7 GB por escena)
    -> usar --incluir-pansharpened para borrarlo tambien.

Uso:
    python scripts/limpiar_salidas.py                 # pide confirmacion
    python scripts/limpiar_salidas.py --dry-run       # solo muestra, no borra
    python scripts/limpiar_salidas.py --si            # sin confirmacion
    python scripts/limpiar_salidas.py --incluir-pansharpened
    python scripts/limpiar_salidas.py --desde-modelo  # solo 06 en adelante (re-entrenar)
"""

import sys
import yaml
import shutil
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


def ruta(*claves, default=None):
    """Resuelve una ruta del config a path absoluto bajo PROJECT_ROOT."""
    d = CFG
    for k in claves:
        d = d.get(k, {})
    rel = d if isinstance(d, str) else default
    return (PROJECT_ROOT / rel).resolve() if rel else None


# Salidas ligeras (se borran por defecto)
SALIDAS = [
    ruta("paths", "processed", "delitos"),
    ruta("paths", "processed", "mosaico", default="data/processed/mosaico"),
    ruta("paths", "processed", "tiles"),
    ruta("paths", "labels"),
    ruta("paths", "splits"),
    ruta("paths", "models"),
    ruta("paths", "results"),
]
SALIDAS = [p for p in SALIDAS if p is not None]

# Salida pesada (opt-in)
PANSHARPENED = ruta("paths", "processed", "pansharpened")

# Carpeta de insumos que jamas debe tocarse
RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()


def tamano_dir(path):
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def fmt(nbytes):
    for unidad in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unidad}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def es_seguro(path):
    """Verifica que el path esta dentro del proyecto y NO dentro de data/raw."""
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError:
        return False, "fuera del proyecto"
    if path == RAW_DIR or RAW_DIR in path.parents or path in RAW_DIR.parents:
        return False, "afecta a data/raw (insumos)"
    if path == PROJECT_ROOT:
        return False, "es la raiz del proyecto"
    return True, ""


def vaciar(path, dry_run):
    """Borra el contenido de un directorio pero conserva el directorio."""
    n = 0
    for item in path.iterdir():
        if dry_run:
            n += 1
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Limpia las salidas del pipeline.")
    parser.add_argument("--si", action="store_true", help="No pedir confirmacion.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar que se borraria, sin borrar.")
    parser.add_argument("--incluir-pansharpened", action="store_true",
                        help="Tambien borrar data/processed/imagenes_pansharpened (pesado).")
    parser.add_argument("--desde-modelo", action="store_true",
                        help="Borrar SOLO las salidas de 06 en adelante (models, results y "
                             "tiles_predicciones.geojson). Conserva tiles, etiquetas y splits "
                             "para re-entrenar sin rehacer 01-05.")
    args = parser.parse_args()

    LABELS_DIR = ruta("paths", "labels")
    PRED_GEOJSON = (LABELS_DIR / "tiles_predicciones.geojson") if LABELS_DIR else None

    archivos_extra = []
    if args.desde_modelo:
        # Solo salidas de entrenamiento/evaluacion/interpretabilidad.
        objetivos = [p for p in [ruta("paths", "models"), ruta("paths", "results")]
                     if p is not None]
        if PRED_GEOJSON is not None:
            archivos_extra.append(PRED_GEOJSON)
    else:
        objetivos = list(SALIDAS)
        if args.incluir_pansharpened and PANSHARPENED is not None:
            objetivos.append(PANSHARPENED)

    print("=" * 60)
    print("  LIMPIEZA DE SALIDAS DEL PIPELINE")
    print("=" * 60)
    print("  Insumos preservados (NO se tocan): data/raw/ (delitos, imagenes,")
    print("  limites de Lima, worldpop).")
    if args.desde_modelo:
        print("  MODO: solo 06 en adelante -> conserva tiles, etiquetas y splits;")
        print("  borra models/, results/ y tiles_predicciones.geojson.")
    elif not args.incluir_pansharpened:
        print("  pansharpened NO se borra (usa --incluir-pansharpened para incluirlo).")
    print()

    total_bytes = 0
    a_borrar = []
    a_borrar_files = []
    for path in objetivos:
        ok, motivo = es_seguro(path)
        if not ok:
            print(f"  OMITIDO (seguridad): {path}  [{motivo}]")
            continue
        existe = path.exists()
        size = tamano_dir(path)
        total_bytes += size
        n_items = len(list(path.iterdir())) if existe else 0
        estado = f"{n_items} items, {fmt(size)}" if existe else "no existe"
        print(f"  {'[dry-run] ' if args.dry_run else ''}{path.relative_to(PROJECT_ROOT)}  ({estado})")
        if existe and n_items > 0:
            a_borrar.append(path)

    for f in archivos_extra:
        ok, motivo = es_seguro(f)
        if not ok:
            print(f"  OMITIDO (seguridad): {f}  [{motivo}]")
            continue
        if f.exists():
            size = f.stat().st_size
            total_bytes += size
            print(f"  {'[dry-run] ' if args.dry_run else ''}{f.relative_to(PROJECT_ROOT)}  (archivo, {fmt(size)})")
            a_borrar_files.append(f)
        else:
            print(f"  {f.relative_to(PROJECT_ROOT)}  (no existe)")

    print(f"\n  Espacio a liberar: {fmt(total_bytes)}")

    if args.dry_run:
        print("\n  (dry-run) No se borro nada.")
        return

    if not a_borrar and not a_borrar_files:
        print("\n  Nada que borrar. Todo limpio.")
        return

    if not args.si:
        resp = input("\n  Confirmas borrar el CONTENIDO de estas carpetas? [escribe 'si']: ").strip().lower()
        if resp not in ("si", "sí", "s", "yes", "y"):
            print("  Cancelado.")
            return

    print()
    for path in a_borrar:
        n = vaciar(path, dry_run=False)
        path.mkdir(parents=True, exist_ok=True)  # conservar carpeta vacia
        print(f"  Vaciado: {path.relative_to(PROJECT_ROOT)}  ({n} items)")
    for f in a_borrar_files:
        f.unlink()
        print(f"  Borrado: {f.relative_to(PROJECT_ROOT)}")

    print(f"\n  Listo. Espacio liberado: {fmt(total_bytes)}")
    if args.desde_modelo:
        print("  Puedes re-entrenar desde scripts/06_entrenar_modelo.py (tiles/labels/splits intactos).")
    else:
        print("  Puedes re-ejecutar el pipeline desde scripts/01_limpiar_datos_delictivos.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
