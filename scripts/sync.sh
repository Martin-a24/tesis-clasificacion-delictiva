#!/usr/bin/env bash
#
# sync.sh - (OPCIONAL) Transfiere datos pesados entre tu PC y un servidor remoto.
#
# ¿Para quien es esto?
#   Solo lo necesitas si editas el codigo en una maquina (tu PC) y ejecutas el
#   pipeline en OTRA (un servidor mas potente). Si trabajas todo en una sola
#   maquina, NO necesitas este script: ignora este archivo por completo.
#
# ¿Que hace?
#   Copia carpetas/archivos con 'rsync' sobre SSH. rsync es ideal para datos
#   pesados porque es REANUDABLE (si se corta, continua), copia solo lo que
#   cambio y muestra progreso. No guarda contrasenas ni IPs aqui: la conexion
#   se define en tu ~/.ssh/config mediante un alias (por defecto: "phantom").
#
# Configuracion (UNA sola vez, en tu PC):
#   1. Define el/los host en ~/.ssh/config. Ejemplo con servidor puente:
#        Host bridge
#            HostName <ip_del_puente>
#            User <usuario_puente>
#        Host phantom
#            HostName <ip_del_servidor>
#            User <tu_usuario>
#            ProxyJump bridge        # omite esta linea si NO hay servidor puente
#   2. Copia tu llave para no escribir contrasenas:
#        ssh-copy-id bridge && ssh-copy-id phantom
#   3. Prueba:  ssh phantom   (debe entrar sin pedir nada)
#
# Variables de entorno para adaptarlo (sin tocar el codigo):
#   REMOTE   alias SSH del servidor          (def: phantom)
#   RDIR     ruta del repo en el servidor     (def: tesis-clasificacion-delictiva)
#   DEST     carpeta local destino de 'pull'  (def: la del repo)
#
# Uso:
#   ./scripts/sync.sh push-imagenes     # PC  -> servidor: data/raw/imagenes_perusat
#   ./scripts/sync.sh pull-resultados   # servidor -> PC: results/
#   ./scripts/sync.sh pull-modelos      # servidor -> PC: models/
#   ./scripts/sync.sh pull-entregables  # servidor -> ~/tesis-entregables (QGIS y tesis)
#   ./scripts/sync.sh push <ruta>       # PC  -> servidor (ruta relativa al repo)
#   ./scripts/sync.sh pull <ruta>       # servidor -> PC (ruta relativa al repo)
#
# Ejemplos:
#   RDIR=proyectos/tesis ./scripts/sync.sh push-imagenes
#   ./scripts/sync.sh pull-entregables ~/Documentos/tesis
#   ./scripts/sync.sh push-imagenes --dry-run   # previsualiza sin transferir
#
set -euo pipefail

REMOTE="${REMOTE:-phantom}"
RDIR="${RDIR:-tesis-clasificacion-delictiva}"
LDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC=(rsync -avhP)

cmd="${1:-}"; shift || true
EXTRA=("$@")    # flags extra (ej. --dry-run) se reenvian a rsync

# Verifica que el alias SSH exista en ~/.ssh/config; si no, guia al usuario.
verificar_config() {
    local host
    host="$(ssh -G "$REMOTE" 2>/dev/null | awk '/^hostname /{print $2}')"
    if [ -z "$host" ] || [ "$host" = "$REMOTE" ]; then
        echo "ERROR: el alias SSH '$REMOTE' no esta configurado en ~/.ssh/config." >&2
        echo "Este script es opcional y solo aplica si ejecutas en un servidor remoto." >&2
        echo "Configura ~/.ssh/config (ver cabecera de este archivo) o define REMOTE=<tu_alias>." >&2
        exit 1
    fi
}

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
    push-imagenes)   verificar_config; push "data/raw/imagenes_perusat/" ;;
    pull-resultados) verificar_config; pull "results/" ;;
    pull-modelos)    verificar_config; pull "models/" ;;
    pull-entregables)
        verificar_config
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
    push)            verificar_config; rel="${EXTRA[0]:?Falta la ruta a subir}"; EXTRA=("${EXTRA[@]:1}"); push "$rel" ;;
    pull)            verificar_config; rel="${EXTRA[0]:?Falta la ruta a bajar}"; EXTRA=("${EXTRA[@]:1}"); pull "$rel" ;;
    *)
        grep '^#' "${BASH_SOURCE[0]}" | grep -v '^#!' | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
