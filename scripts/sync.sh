#!/usr/bin/env bash
#
# sync.sh - Sincroniza datos pesados entre la PC local y el servidor (phantom)
# usando rsync sobre SSH con ProxyJump (salta el servidor puente automaticamente).
#
# Requisitos (una sola vez):
#   1. Tener en ~/.ssh/config los alias 'bridge' y 'phantom' con ProxyJump.
#   2. Llaves copiadas: ssh-copy-id bridge && ssh-copy-id phantom
#      (asi rsync no pide contrasena).
#
# La ruta del repo en el servidor se asume ~/tesis-clasificacion-delictiva.
# Cambiala con la variable de entorno RDIR si es distinta. Ejemplos:
#   RDIR=proyectos/tesis ./scripts/sync.sh push-imagenes
#   REMOTE=phantom        ./scripts/sync.sh pull-resultados
#
# Uso:
#   ./scripts/sync.sh push-imagenes     # PC  -> servidor: data/raw/imagenes_perusat
#   ./scripts/sync.sh pull-resultados   # servidor -> PC: results/
#   ./scripts/sync.sh pull-modelos      # servidor -> PC: models/
#   ./scripts/sync.sh pull-entregables  # servidor -> ~/tesis-entregables: labels,
#                                       #   results y geojson de delitos (para QGIS y tesis)
#   ./scripts/sync.sh push <ruta>       # PC  -> servidor (ruta relativa al repo)
#   ./scripts/sync.sh pull <ruta>       # servidor -> PC (ruta relativa al repo)
#
# Destino fuera del repo: define DEST para que los 'pull' caigan en otra carpeta.
#   DEST=~/tesis-entregables ./scripts/sync.sh pull data/labels/
#   ./scripts/sync.sh pull-entregables ~/Documentos/tesis   # destino como argumento
#
# Agrega --dry-run al final para previsualizar sin transferir, p. ej.:
#   ./scripts/sync.sh push-imagenes --dry-run

set -euo pipefail

REMOTE="${REMOTE:-phantom}"
RDIR="${RDIR:-tesis-clasificacion-delictiva}"
LDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC=(rsync -avhP)

cmd="${1:-}"; shift || true

# Cualquier flag extra (ej. --dry-run) se reenvia a rsync
EXTRA=("$@")

push() {  # push <ruta_relativa>
    local rel="$1"
    "${RSYNC[@]}" "${EXTRA[@]}" "$LDIR/$rel" "$REMOTE:$RDIR/$rel"
}
pull() {  # pull <ruta_relativa>  (destino local = DEST si esta definido, si no el repo)
    local rel="$1"
    local base="${DEST:-$LDIR}"
    mkdir -p "$base/$(dirname "$rel")"
    "${RSYNC[@]}" "${EXTRA[@]}" "$REMOTE:$RDIR/$rel" "$base/$rel"
}

case "$cmd" in
    push-imagenes)   push "data/raw/imagenes_perusat/" ;;
    pull-resultados) pull "results/" ;;
    pull-modelos)    pull "models/" ;;
    pull-entregables)
        # Paquete liviano para QGIS y la tesis (sin tiles ni pansharpened).
        # Destino: primer argumento, o $DEST, o ~/tesis-entregables.
        if [[ "${EXTRA[0]:-}" != "" && "${EXTRA[0]:0:1}" != "-" ]]; then
            DEST="${EXTRA[0]}"; EXTRA=("${EXTRA[@]:1}")
        fi
        DEST="${DEST:-$HOME/tesis-entregables}"
        echo ">> Bajando entregables a: $DEST"
        pull "data/labels/"                      || echo "  (labels: nada que bajar)"
        pull "results/"                          || echo "  (results: nada que bajar)"
        pull "data/processed/delitos_limpios/"   || echo "  (delitos: nada que bajar)"
        ;;
    push)            rel="${EXTRA[0]:?Falta la ruta a subir}"; EXTRA=("${EXTRA[@]:1}"); push "$rel" ;;
    pull)            rel="${EXTRA[0]:?Falta la ruta a bajar}"; EXTRA=("${EXTRA[@]:1}"); pull "$rel" ;;
    *)
        grep '^#' "${BASH_SOURCE[0]}" | grep -v '^#!' | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
