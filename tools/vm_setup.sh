#!/usr/bin/env bash
# =============================================================================
# tools/vm_setup.sh — Setup día-1 de la VM Azure de PIAR
#
#   VM:  lp-gpu-h100-x2-spot  (RG-IAF-YTEC-poc-int, eastus)
#        Standard_NC80adis_H100_v5 — 2×H100 NVL (188GB VRAM), Spot/Deallocate.
#
# Filosofía: en la VM NO se desarrolla — se clona, se ejecuta, se bajan
# artefactos. Este script deja la VM lista para correr el harness de la
# Figura 1 (tools/figura1_scoring.py) y las replicaciones B1/B2 (#16) con
# UN comando. Es idempotente: cada paso chequea si ya está hecho y skipea,
# así que re-correrlo tras un fallo parcial es seguro y barato.
#
# Qué instala / baja (≈60-80 GB en $HOME; runtime estimado 1-2 h):
#   - Miniforge (conda, canal conda-forge) si no hay conda.
#   - env `piar`    (Python 3.12): torch 2.6 cu124 + flash-attn 2.7.4.post1 +
#                   verl (pip -e code/) + vllm 0.8.5  [orden de code/README.md].
#   - env `webshop` (Python 3.10): requirements del env package (pyserini 0.17,
#                   spacy, gym 0.24, ...) + openjdk 11 + spacy models + EL MISMO
#                   stack de training encima (torch/flash-attn/verl/vllm).
#                   ⚠ code/README.md:55-62 instala verl DENTRO del env 3.10:
#                   el training de WebShop corre desde este env (envs.py importa
#                   web_agent_site in-process vía torch.multiprocessing).
#   - Datasets WebShop por mirror HF (el Google Drive oficial está quota-exceeded,
#     ver #16/#17): 5 JSONs desde YWZBrandon/webshop-data + índice Lucene
#     PRE-CONSTRUIDO desde ai-hyz/MemoryArena-product-db (ahorra horas de
#     indexing con Java).
#   - Modelos: Qwen2.5-7B-Instruct + Qwen2.5-1.5B-Instruct (+ tokenizer del
#     0.5B para el self-test del harness), con symlinks en $PIAR_ROOT/Qwen/
#     para que examples/*/run_webshop.sh (model_path=../Qwen/...) funcione tal cual.
#   - uv (para los scripts standalone de tools/).
#
# Uso (EN LA VM):
#   bash tools/vm_setup.sh [--branch <rama>] [--root <dir>] \
#                          [--skip-models] [--skip-datasets]
#
# Env vars (todas opcionales):
#   PIAR_ROOT     destino del clone           (default: $HOME/piar-rl)
#   PIAR_BRANCH   rama a clonar/checkout      (default: pivot-2026-06; usar main post-merge)
#   HF_HOME       cache de HuggingFace        (default: $HOME/.cache/huggingface)
#   PIAR_MIN_DISK_GB  mínimo de disco libre   (default: 100)
#   PIAR_ALLOW_GPU_MISMATCH=1  no abortar si no hay exactamente 2 GPUs
#   MAX_JOBS      paralelismo del build de flash-attn si cae al fallback source
#
# -----------------------------------------------------------------------------
# COMANDOS DESDE LA MÁQUINA LOCAL (NO se corren en la VM):
#
#   # 1) Prender la VM (spot — puede tardar / fallar si no hay capacidad):
#   az vm start -g RG-IAF-YTEC-poc-int -n lp-gpu-h100-x2-spot
#
#   # 2) Auto-shutdown (crearlo UNA vez; hora en UTC — 0300 UTC = 00:00 ART):
#   az vm auto-shutdown -g RG-IAF-YTEC-poc-int -n lp-gpu-h100-x2-spot --time 0300
#
#   # 3) IP y ssh:
#   IP=$(az vm show -d -g RG-IAF-YTEC-poc-int -n lp-gpu-h100-x2-spot \
#        --query publicIps -o tsv)
#   ssh azureuser@$IP    # VERIFICAR-EN-VM: usuario admin real de la VM
#
#   # One-liner día-1 completo (start → ssh → setup):
#   az vm start -g RG-IAF-YTEC-poc-int -n lp-gpu-h100-x2-spot && \
#     IP=$(az vm show -d -g RG-IAF-YTEC-poc-int -n lp-gpu-h100-x2-spot --query publicIps -o tsv) && \
#     ssh azureuser@$IP 'curl -fsSL https://raw.githubusercontent.com/lucaspecina/piar-rl/pivot-2026-06/tools/vm_setup.sh | bash'
#
#   # Al terminar la sesión (spot = pagar solo cuando corre):
#   az vm deallocate -g RG-IAF-YTEC-poc-int -n lp-gpu-h100-x2-spot
#
#   Nota spot: eviction = Deallocate. Todo lo que vive en $HOME (OS disk)
#   persiste; /mnt (resource disk efímero) NO. Por eso el default es $HOME.
# -----------------------------------------------------------------------------
# Marcas "VERIFICAR-EN-VM": puntos asumidos que no se pudieron probar desde
# la máquina de desarrollo (Windows, sin GPU). Buscar con:  grep VERIFICAR-EN-VM
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PIAR_ROOT="${PIAR_ROOT:-$HOME/piar-rl}"
PIAR_BRANCH="${PIAR_BRANCH:-pivot-2026-06}"
PIAR_REPO_URL="${PIAR_REPO_URL:-https://github.com/lucaspecina/piar-rl.git}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PIAR_MIN_DISK_GB="${PIAR_MIN_DISK_GB:-100}"
SKIP_MODELS=0
SKIP_DATASETS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --branch)        PIAR_BRANCH="$2"; shift 2 ;;
        --root)          PIAR_ROOT="$2"; shift 2 ;;
        --skip-models)   SKIP_MODELS=1; shift ;;
        --skip-datasets) SKIP_DATASETS=1; shift ;;
        -h|--help)       sed -n '2,75p' "$0"; exit 0 ;;
        *) echo "Flag desconocido: $1 (ver --help)"; exit 2 ;;
    esac
done

# Paths derivados (verificados contra el código del fork — no cambiar sin
# revisar web_agent_site/utils.py y engine.py):
CODE_DIR="$PIAR_ROOT/code"
WEBSHOP_DIR="$CODE_DIR/agent_system/environments/env_package/webshop/webshop"
DATA_DIR="$WEBSHOP_DIR/data"                       # utils.py:10-17  (../data/)
INDEX_DIR="$WEBSHOP_DIR/search_engine/indexes"     # engine.py:206   (num_products=None → 'indexes')
CONDA_BASE=""                                      # se resuelve en step_conda

# Mirrors HF (verificados 2026-06-12 vía API de HF; ver issue #16):
HF_DATA_PRIMARY="https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main"
HF_DATA_BACKUP="https://huggingface.co/datasets/ai-hyz/MemoryArena-product-db/resolve/main"
HF_INDEX_REPO="ai-hyz/MemoryArena-product-db"      # trae search_engine/indexes-full/ pre-construido

# Tamaños exactos en bytes (HF API, 2026-06-12) — actúan de checksum débil:
SZ_ITEMS_HUMAN=5137548
SZ_ITEMS_SHUFFLE=5479720229
SZ_ITEMS_INS=186295270
SZ_ITEMS_SHUFFLE_1K=4467013
SZ_ITEMS_INS_1K=147099
SHA_ITEMS_HUMAN="cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3"
SZ_INDEX_FDT=675297212                             # indexes-full/_0.fdt (archivo mayor del índice)

FLASH_ATTN_VER="2.7.4.post1"
FLASH_ATTN_RELEASE="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}"

# -----------------------------------------------------------------------------
# Logging + resumen
# -----------------------------------------------------------------------------
if [ -t 1 ]; then
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
    C_GREEN=""; C_RED=""; C_YELLOW=""; C_BOLD=""; C_OFF=""
fi

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
warn() { echo "${C_YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $*${C_OFF}" >&2; }
die()  { echo "${C_RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*${C_OFF}" >&2; exit 1; }

STEP_NAMES=()
STEP_STATUS=()   # OK | FAIL | SKIP-FLAG
STEP_SECS=()
SUMMARY_PRINTED=0

print_summary() {
    [ "$SUMMARY_PRINTED" = 1 ] && return 0
    SUMMARY_PRINTED=1
    echo
    echo "${C_BOLD}================== RESUMEN DEL SETUP ==================${C_OFF}"
    local i fails=0
    for i in "${!STEP_NAMES[@]}"; do
        case "${STEP_STATUS[$i]}" in
            OK)        printf '  %s%-12s%s %-28s (%ss)\n' "$C_GREEN" "VERDE" "$C_OFF" "${STEP_NAMES[$i]}" "${STEP_SECS[$i]}" ;;
            SKIP-FLAG) printf '  %s%-12s%s %-28s (flag --skip)\n' "$C_YELLOW" "SKIP" "$C_OFF" "${STEP_NAMES[$i]}" ;;
            *)         printf '  %s%-12s%s %-28s (%ss)\n' "$C_RED" "ROJO" "$C_OFF" "${STEP_NAMES[$i]}" "${STEP_SECS[$i]}"; fails=$((fails+1)) ;;
        esac
    done
    echo "${C_BOLD}=======================================================${C_OFF}"
    if [ "$fails" -gt 0 ]; then
        echo "${C_RED}${fails} paso(s) en ROJO — el script es idempotente: arreglar y re-correr.${C_OFF}"
    fi
    echo
}
trap print_summary EXIT

# run_step <nombre> <critical|soft> <funcion>
#   critical → un fallo aborta (tras imprimir el resumen).
#   soft     → un fallo se registra ROJO y se sigue (re-correr después).
run_step() {
    local name="$1" mode="$2" fn="$3" rc t0 t1
    echo
    log "${C_BOLD}── PASO: ${name} ──${C_OFF}"
    t0=$(date +%s)
    set +e
    ( set -euo pipefail; "$fn" )
    rc=$?
    set -e
    t1=$(date +%s)
    STEP_NAMES+=("$name"); STEP_SECS+=($((t1 - t0)))
    if [ "$rc" -eq 0 ]; then
        STEP_STATUS+=("OK"); log "PASO ${name}: ${C_GREEN}OK${C_OFF} ($((t1 - t0))s)"
    else
        STEP_STATUS+=("FAIL"); log "PASO ${name}: ${C_RED}FALLÓ${C_OFF} (rc=$rc)"
        [ "$mode" = critical ] && exit 1
    fi
    return 0
}

mark_skipped() { STEP_NAMES+=("$1"); STEP_STATUS+=("SKIP-FLAG"); STEP_SECS+=(0); }

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
ensure_line() {  # ensure_line <file> <línea exacta>
    grep -qxF "$2" "$1" 2>/dev/null || echo "$2" >> "$1"
}

# Activación de conda tolerante a set -u (los scripts de activate referencian
# vars no seteadas en algunas versiones).
conda_on() { set +u; # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$1"; set -u; }

file_size() { stat -c%s "$1" 2>/dev/null || echo 0; }

# fetch_with_resume <url> <dest> <size_bytes> [sha256]
# curl con -C - (resume), reintentos, verificación de tamaño exacto y sha opcional.
CURL_RETRY_ALL=""
if curl --help all 2>/dev/null | grep -q -- --retry-all-errors; then
    CURL_RETRY_ALL="--retry-all-errors"   # curl >= 7.71 (Ubuntu 22.04 OK)
fi
fetch_with_resume() {
    local url="$1" dest="$2" size="$3" sha="${4:-}"
    if [ "$(file_size "$dest")" = "$size" ]; then
        log "  ya está: $(basename "$dest") (${size} bytes) — skip descarga"
    else
        log "  bajando $(basename "$dest") (${size} bytes) desde ${url%%/resolve*}..."
        local attempt
        for attempt in 1 2 3; do
            # shellcheck disable=SC2086
            if curl -fL --retry 5 --retry-delay 15 $CURL_RETRY_ALL \
                    --connect-timeout 30 -C - -o "$dest" "$url"; then
                break
            fi
            warn "  intento $attempt de descarga falló — reintento en 20s"
            sleep 20
            # si el server no soporta ranges, el resume puede dejar basura:
            [ "$attempt" = 2 ] && { warn "  descartando parcial y bajando de cero"; rm -f "$dest"; }
        done
    fi
    [ "$(file_size "$dest")" = "$size" ] || \
        die "tamaño de $(basename "$dest") = $(file_size "$dest"), esperado ${size}. ¿Cambió el mirror? Ver #16."
    if [ -n "$sha" ]; then
        echo "${sha}  ${dest}" | sha256sum -c - >/dev/null || die "sha256 de $(basename "$dest") NO coincide"
        log "  sha256 OK: $(basename "$dest")"
    fi
}

# CLI de HuggingFace (instalado vía uv tool; binario `hf`, fallback `huggingface-cli`)
hf_cli() {
    if command -v hf >/dev/null 2>&1; then hf "$@"
    elif command -v huggingface-cli >/dev/null 2>&1; then huggingface-cli "$@"
    else die "no hay CLI de HuggingFace en PATH (paso uv+hf-cli falló?)"; fi
}

# Instala flash-attn en el env conda ACTIVO. Fast-path: wheel oficial del
# release de GitHub (existen cp310/cp312 + cu12 + torch2.6, verificado
# 2026-06-12). Fallback: build desde source (necesita nvcc, 30-60 min).
install_flash_attn() {
    if python -c "import flash_attn; assert flash_attn.__version__ == '${FLASH_ATTN_VER}'" 2>/dev/null; then
        log "  flash-attn ${FLASH_ATTN_VER} ya instalado — skip"
        return 0
    fi
    local py_tag abi url
    py_tag=$(python -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
    # VERIFICAR-EN-VM: ABI del torch instalado (los wheels cu124 de
    # download.pytorch.org históricamente reportan cxx11abi FALSE en 2.6).
    abi=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
    url="${FLASH_ATTN_RELEASE}/flash_attn-${FLASH_ATTN_VER}+cu12torch2.6cxx11abi${abi}-${py_tag}-${py_tag}-linux_x86_64.whl"
    log "  intentando wheel pre-compilado: $url"
    if pip install "$url"; then
        log "  flash-attn instalado desde wheel"
    else
        warn "  wheel falló — fallback: compilar desde source (necesita nvcc del host)"
        # VERIFICAR-EN-VM: que la imagen de la VM tenga CUDA toolkit (nvcc).
        command -v nvcc >/dev/null 2>&1 || \
            die "no hay nvcc para compilar flash-attn. Instalar CUDA toolkit o conseguir el wheel a mano."
        pip install ninja packaging psutil wheel
        MAX_JOBS="${MAX_JOBS:-16}" pip install "flash-attn==${FLASH_ATTN_VER}" --no-build-isolation
    fi
    python -c "import flash_attn; print('  flash-attn', flash_attn.__version__, 'OK')"
}

# Stack de training (orden EXACTO de code/README.md:25-35 / 55-62):
# torch cu124 → flash-attn --no-build-isolation → pip -e code/ → vllm 0.8.5.
# Extra: -r code/requirements.txt (liger-kernel/uvicorn/fastapi que el
# setup.py no declara) y swanlab (los run_webshop.sh lo usan como logger).
install_training_stack() {
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    install_flash_attn
    pip install -e "$CODE_DIR"
    pip install -r "$CODE_DIR/requirements.txt"   # flash-attn ya satisfecho → no recompila
    pip install vllm==0.8.5
    pip install swanlab                            # trainer.logger=['console','swanlab'] en examples/
}

training_stack_ok() {  # $1 = nombre de env conda
    "$CONDA_BASE/envs/$1/bin/python" - <<'PY' >/dev/null 2>&1
import torch, flash_attn, vllm, verl
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert flash_attn.__version__ == "2.7.4.post1", flash_attn.__version__
PY
}

# =============================================================================
# PASOS
# =============================================================================

step_sanity() {
    [ "$(uname -s)" = Linux ] || die "este script es para la VM Ubuntu, no para $(uname -s)"
    grep -qi ubuntu /etc/os-release 2>/dev/null || warn "no parece Ubuntu — seguimos igual"
    command -v git  >/dev/null || die "falta git  (sudo apt-get install -y git)"
    command -v curl >/dev/null || die "falta curl (sudo apt-get install -y curl)"

    # GPUs: 2×H100 visibles
    command -v nvidia-smi >/dev/null || die "no hay nvidia-smi — ¿driver NVIDIA instalado? VM equivocada?"
    local n_gpu gpu_names
    gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    n_gpu=$(echo "$gpu_names" | grep -c . || true)
    log "GPUs detectadas (${n_gpu}):"; echo "$gpu_names" | sed 's/^/    /'
    if [ "$n_gpu" -ne 2 ]; then
        [ "${PIAR_ALLOW_GPU_MISMATCH:-0}" = 1 ] || \
            die "se esperaban 2 GPUs y hay ${n_gpu} (override: PIAR_ALLOW_GPU_MISMATCH=1)"
        warn "GPU count=${n_gpu} ≠ 2 — continuando por PIAR_ALLOW_GPU_MISMATCH=1"
    fi
    echo "$gpu_names" | grep -qi h100 || warn "las GPUs no reportan H100 — ¿VM correcta?"

    # VERIFICAR-EN-VM: versión CUDA del driver del host. Los wheels usados son
    # cu124 ⇒ el driver debe soportar CUDA >= 12.4 (driver >= 550).
    local cuda_drv
    cuda_drv=$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9.]+' || echo "?")
    log "CUDA del driver: ${cuda_drv} (se necesita >= 12.4 para los wheels cu124)"
    if [ "$cuda_drv" != "?" ] && [ "$(printf '%s\n' "12.4" "$cuda_drv" | sort -V | head -1)" != "12.4" ]; then
        warn "driver reporta CUDA ${cuda_drv} < 12.4 — torch cu124 puede no funcionar"
    fi

    # Disco: el destino es $HOME (persiste el deallocate del spot). /mnt NO persiste.
    mkdir -p "$(dirname "$PIAR_ROOT")"
    local avail_gb
    avail_gb=$(df -BG --output=avail "$(dirname "$PIAR_ROOT")" | tail -1 | tr -dc '0-9')
    log "Disco libre en $(dirname "$PIAR_ROOT"): ${avail_gb} GB (mínimo ${PIAR_MIN_DISK_GB}, recomendado 150)"
    [ "$avail_gb" -ge "$PIAR_MIN_DISK_GB" ] || \
        die "solo ${avail_gb} GB libres (< ${PIAR_MIN_DISK_GB}). Liberar espacio o agrandar el OS disk."
    [ "$avail_gb" -ge 150 ] || warn "menos de 150 GB libres — el setup completo usa ~60-80 GB"
    case "$PIAR_ROOT" in /mnt/*) warn "PIAR_ROOT está en /mnt (disco EFÍMERO): se pierde al deallocate del spot" ;; esac
}

step_repo() {
    if [ -d "$PIAR_ROOT/.git" ]; then
        log "repo ya clonado en $PIAR_ROOT — fetch + checkout $PIAR_BRANCH"
        git -C "$PIAR_ROOT" fetch origin
        if [ -n "$(git -C "$PIAR_ROOT" status --porcelain)" ]; then
            warn "working tree con cambios locales — NO hago pull (la VM no desarrolla; revisar a mano)"
        else
            git -C "$PIAR_ROOT" checkout "$PIAR_BRANCH"
            git -C "$PIAR_ROOT" pull --ff-only origin "$PIAR_BRANCH"
        fi
    else
        log "clonando $PIAR_REPO_URL (branch $PIAR_BRANCH) → $PIAR_ROOT"
        git clone --branch "$PIAR_BRANCH" "$PIAR_REPO_URL" "$PIAR_ROOT"
    fi
    log "HEAD: $(git -C "$PIAR_ROOT" log --oneline -1)"

    # Artefactos de runtime que viven dentro del working tree pero no deben
    # ensuciar `git status` (exclude local de la VM, NO toca el repo):
    local excl="$PIAR_ROOT/.git/info/exclude"
    ensure_line "$excl" "/Qwen/"             # symlinks a los modelos (run scripts usan ../Qwen/...)
    ensure_line "$excl" "/ISTAR/"            # parquet de examples.data_preprocess.prepare
    ensure_line "$excl" "/code/checkpoints/"
    ensure_line "$excl" "/code/outputs/"
    ensure_line "$excl" "/code/swanlog/"
    ensure_line "$excl" "/code/wandb/"
    # (data/ e indexes del webshop ya están cubiertos por el .gitignore vendoreado)
}

step_uv_hfcli() {
    if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
        log "instalando uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null || die "uv no quedó en PATH ($HOME/.local/bin)"
    log "uv: $(uv --version)"

    if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
        log "instalando CLI de HuggingFace (uv tool)"
        uv tool install "huggingface_hub[cli]"
    fi
    hf_cli version >/dev/null 2>&1 || hf_cli --help >/dev/null
    log "hf CLI OK"

    # Env persistente para sesiones ssh futuras
    {
        echo "# generado por tools/vm_setup.sh — $(date '+%Y-%m-%d')"
        echo "export PIAR_ROOT=\"$PIAR_ROOT\""
        echo "export HF_HOME=\"$HF_HOME\""
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    } > "$HOME/.piar_env"
    ensure_line "$HOME/.bashrc" "source \$HOME/.piar_env"
}

step_conda() {
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [ -x "$HOME/miniforge3/bin/conda" ]; then
        CONDA_BASE="$HOME/miniforge3"
    elif [ -x "$HOME/miniconda3/bin/conda" ]; then
        CONDA_BASE="$HOME/miniconda3"
    else
        # Miniforge y no Miniconda: default conda-forge ⇒ sin fricción de ToS
        # de los canales de Anaconda en una VM corporativa.
        log "instalando Miniforge en $HOME/miniforge3"
        curl -fL --retry 3 -o /tmp/miniforge.sh \
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
        rm -f /tmp/miniforge.sh
        CONDA_BASE="$HOME/miniforge3"
    fi
    log "conda base: $CONDA_BASE"
    "$CONDA_BASE/bin/conda" --version
    echo "source \"$CONDA_BASE/etc/profile.d/conda.sh\"" >> "$HOME/.piar_env"
}

# env `piar` (Python 3.12) — stack de training puro (code/README.md:25-35).
# Para Sokoban, tools con GPU y cualquier cosa que no necesite el env package
# de WebShop. (El training de WebShop usa el env `webshop`, ver abajo.)
step_env_piar() {
    conda_on base
    if ! conda env list | awk '{print $1}' | grep -qx piar; then
        conda create -n piar python=3.12 -y
    fi
    if training_stack_ok piar; then
        log "env piar ya completo (torch/flash-attn/vllm/verl) — skip"
        return 0
    fi
    conda_on piar
    install_training_stack
}

# env `webshop` (Python 3.10) — env package de WebShop + stack de training.
# Equivale a `setup.sh -d all` SIN los gdown muertos (datasets van por mirror
# en step_datasets) y SIN el indexing (índice pre-construido), + los pasos de
# code/README.md:55-62 (verl dentro del env 3.10).
step_env_webshop() {
    conda_on base
    if ! conda env list | awk '{print $1}' | grep -qx webshop; then
        conda create -n webshop python=3.10 -y
    fi

    # nmslib ANTES de pip: pyserini 0.17.0 requiere nmslib>=2.1.1, que no tiene
    # wheel cp310 en PyPI y suele fallar compilando con gcc moderno. conda-forge
    # lo trae pre-compilado. openjdk=11 es el de setup.sh:32 (pyserini/Lucene).
    # faiss-cpu: setup.sh:28 (deps de pyserini para dense retrieval).
    # (mkl de setup.sh:27 se omite: faiss-cpu de conda-forge usa openblas.)
    conda install -y -n webshop -c conda-forge --override-channels \
        openjdk=11 nmslib faiss-cpu

    conda_on webshop
    if python -c "import pyserini, spacy" 2>/dev/null && training_stack_ok webshop; then
        if python -c "import spacy; spacy.load('en_core_web_lg'); spacy.load('en_core_web_sm')" 2>/dev/null; then
            log "env webshop ya completo — skip"
            return 0
        fi
    fi

    pip install -r "$WEBSHOP_DIR/requirements.txt"   # pyserini 0.17, gym 0.24, spacy 3.7.2, torch 2.6 (PyPI=cu124)...
    python -m spacy download en_core_web_lg          # setup.sh:53-54
    python -m spacy download en_core_web_sm

    install_training_stack                            # README:55-62: verl/vllm DENTRO del env 3.10
}

step_datasets() {
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    # --- 5 JSONs (paths que el env lee: web_agent_site/utils.py:10-17 +
    #     env_manager.py:751-756 según use_small) -------------------------------
    # items_human_ins.json se carga SIEMPRE (engine.py:250, incondicional).
    fetch_with_resume "$HF_DATA_PRIMARY/items_human_ins.json"    items_human_ins.json    "$SZ_ITEMS_HUMAN" "$SHA_ITEMS_HUMAN"
    fetch_with_resume "$HF_DATA_PRIMARY/items_ins_v2.json"       items_ins_v2.json       "$SZ_ITEMS_INS"
    fetch_with_resume "$HF_DATA_PRIMARY/items_shuffle.json"      items_shuffle.json      "$SZ_ITEMS_SHUFFLE"
    # Los _1000 son los defaults del trainer (ppo_trainer.yaml:323 use_small=True):
    fetch_with_resume "$HF_DATA_PRIMARY/items_shuffle_1000.json" items_shuffle_1000.json "$SZ_ITEMS_SHUFFLE_1K"
    fetch_with_resume "$HF_DATA_PRIMARY/items_ins_v2_1000.json"  items_ins_v2_1000.json  "$SZ_ITEMS_INS_1K"
    # Backup manual si YWZBrandon se cae: mismos nombres en $HF_DATA_BACKUP
    # (solo los 3 archivos full; los _1000 habría que regenerarlos como full[:1000],
    #  cf. engine.py:257-258 "we assume products already shuffled").

    # --- Índice Lucene pre-construido (ahorra el run_indexing.sh con Java) ----
    # Runtime: engine.py:206 abre BASE_DIR/../search_engine/indexes con
    # num_products=None (lo que pasan envs.py:115 y env_manager.py:759 SIEMPRE).
    if [ -f "$INDEX_DIR/segments_1" ] && [ "$(file_size "$INDEX_DIR/_0.fdt")" = "$SZ_INDEX_FDT" ]; then
        log "índice Lucene ya está en $INDEX_DIR — skip"
    else
        log "bajando índice pre-construido desde $HF_INDEX_REPO (search_engine/indexes-full/, ~1.5 GB)"
        local tmp="$WEBSHOP_DIR/search_engine/.hf_index_tmp"
        mkdir -p "$tmp"
        # VERIFICAR-EN-VM: nombre del subdir en el dataset HF = indexes-full
        # (verificado 2026-06-12 vía https://huggingface.co/api/datasets/ai-hyz/MemoryArena-product-db/tree/main/search_engine)
        hf_cli download "$HF_INDEX_REPO" --repo-type dataset \
            --include "search_engine/indexes-full/*" --local-dir "$tmp"
        mkdir -p "$INDEX_DIR"
        # mv archivo a archivo (mismo filesystem) y limpiar el tmp:
        mv -f "$tmp/search_engine/indexes-full/"* "$INDEX_DIR/"
        rm -rf "$tmp"
        [ -f "$INDEX_DIR/segments_1" ] || die "el índice no tiene segments_1 — descarga incompleta"
        [ "$(file_size "$INDEX_DIR/_0.fdt")" = "$SZ_INDEX_FDT" ] || \
            die "_0.fdt = $(file_size "$INDEX_DIR/_0.fdt") bytes, esperado $SZ_INDEX_FDT"
        log "índice OK: $(ls "$INDEX_DIR" | wc -l) archivos en $INDEX_DIR"
    fi
}

step_models() {
    mkdir -p "$PIAR_ROOT/Qwen"
    local repo snap
    for repo in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
        log "bajando $repo (HF_HOME=$HF_HOME)"
        # VERIFICAR-EN-VM: `hf download` imprime el path del snapshot como
        # última línea de stdout (las barras de progreso van a stderr).
        snap=$(hf_cli download "$repo" | tail -n1)
        if [ ! -d "$snap" ]; then
            warn "no pude capturar el snapshot path de stdout — lo busco en el cache"
            snap=$(ls -d "$HF_HOME/hub/models--${repo//\//--}/snapshots/"*/ 2>/dev/null | head -1)
        fi
        [ -d "$snap" ] || die "no encuentro el snapshot de $repo en $HF_HOME"
        # Symlink para que examples/*/run_webshop.sh (model_path=../Qwen/<name>,
        # relativo a cwd=code/) funcione sin editar nada:
        ln -sfn "${snap%/}" "$PIAR_ROOT/Qwen/$(basename "$repo")"
        log "  $PIAR_ROOT/Qwen/$(basename "$repo") -> ${snap%/}"
    done
    # Tokenizer del 0.5B: lo usa el --self-test del harness (solo tokenizer, sin pesos):
    hf_cli download Qwen/Qwen2.5-0.5B-Instruct \
        --include "tokenizer*" --include "*.json" >/dev/null
    log "tokenizer Qwen2.5-0.5B-Instruct cacheado (para figura1 --self-test)"
}

step_verify_piar() {
    "$CONDA_BASE/envs/piar/bin/python" - <<'PY'
import torch, flash_attn, vllm, verl
n = torch.cuda.device_count()
print(f"torch {torch.__version__} | cuda available={torch.cuda.is_available()} | device_count={n}")
for i in range(n):
    print(f"  GPU{i}: {torch.cuda.get_device_name(i)}")
print(f"flash_attn {flash_attn.__version__} | vllm {vllm.__version__} | verl {verl.__version__}")
assert n == 2, f"se esperaban 2 GPUs visibles desde torch, hay {n}"
assert torch.cuda.is_bf16_supported(), "bf16 no soportado?!"
PY
}

step_verify_webshop() {
    conda_on webshop      # activa JAVA_HOME del openjdk de conda (pyserini/jnius)
    cd "$WEBSHOP_DIR"
    python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())

# (1) smoke import del env (dispara engine.py → pyserini → JVM):
from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
print("import WebAgentTextEnv: OK")

# (2) el índice pre-construido responde una query real:
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher(os.path.join(os.getcwd(), "search_engine", "indexes"))
hits = s.search("red long sleeve shirt", k=5)
assert len(hits) > 0, "el índice Lucene devolvió 0 hits"
print(f"LuceneSearcher sobre search_engine/indexes: OK ({len(hits)} hits)")

# (3) el stack de training convive en este env (README:55-62):
import torch, flash_attn, vllm, verl  # noqa: F401
print(f"training stack en env webshop: torch {torch.__version__}, "
      f"cuda={torch.cuda.device_count()} GPUs, vllm {vllm.__version__}")
assert torch.cuda.device_count() == 2
PY
}

step_verify_figura1() {
    cd "$PIAR_ROOT"
    # Comando EXACTO documentado en tools/README.md (self-test del harness;
    # solo tokenizer, sin pesos). Puebla además el cache de uv para el run real.
    PYTHONUTF8=1 uv run --with torch --with transformers --no-project \
        python tools/figura1_scoring.py --self-test --model Qwen/Qwen2.5-0.5B-Instruct
}

print_next_steps() {
    cat <<EOF

${C_BOLD}=============== NEXT STEPS (la VM quedó lista) ===============${C_OFF}

  Entorno en sesiones nuevas:   source ~/.piar_env
  Envs conda:                   conda activate piar   (py3.12, training puro)
                                conda activate webshop (py3.10, WebShop + training)

  ${C_BOLD}1) Figura 1 (#23 — prereg congelado):${C_OFF}
     Falta generar los ~200 rollouts congelados a experiments/E001/rollouts.jsonl
     (generador pendiente; contrato JSONL en el docstring de tools/figura1_scoring.py).
     Con los rollouts en mano, los dos comandos del prereg (tools/README.md):

     cd \$PIAR_ROOT
     PYTHONUTF8=1 uv run --with torch --with transformers --no-project \\
         python tools/figura1_scoring.py \\
         --rollouts experiments/E001/rollouts.jsonl \\
         --model Qwen/Qwen2.5-7B-Instruct --device auto --history both \\
         --out experiments/E001/figura1_scores.json

     PYTHONUTF8=1 uv run --with torch --with transformers --no-project \\
         python tools/figura1_scoring.py \\
         --rollouts experiments/E001/rollouts.jsonl \\
         --model Qwen/Qwen2.5-7B-Instruct --device auto --history both \\
         --shuffled --baseline-json experiments/E001/figura1_scores.json \\
         --out experiments/E001/figura1_scores_shuffled.json

     (Shakedown previo: --model Qwen/Qwen2.5-1.5B-Instruct — ya descargado.)

  ${C_BOLD}2) Replicación B1 (iStar) / B2 (GiGPO) en WebShop (#16):${C_OFF}
     cd \$PIAR_ROOT/code && conda activate webshop
     bash examples/istar_trainer/run_webshop.sh     # B1
     bash examples/gigpo_trainer/run_webshop.sh     # B2
     ANTES del primer run, ajustar el script (asume 8 GPUs y mirror HF chino):
       - trainer.n_gpus_per_node=8 → 2   (y revisar ppo_micro_batch_size_per_gpu)
       - comentar 'export HF_ENDPOINT=https://hf-mirror.com'
       - logger: 'swanlab login' (ya instalado) o cambiar a trainer.logger=['console']
       - spot ⇒ checkpointing agresivo: bajar trainer.save_freq
     El model_path=../Qwen/Qwen2.5-7B-Instruct ya resuelve al snapshot local.

  ${C_BOLD}3) Al terminar la sesión${C_OFF} (desde tu máquina LOCAL, no acá):
     az vm deallocate -g RG-IAF-YTEC-poc-int -n lp-gpu-h100-x2-spot

==============================================================
EOF
}

# =============================================================================
# MAIN
# =============================================================================
log "${C_BOLD}PIAR vm_setup — root=$PIAR_ROOT branch=$PIAR_BRANCH HF_HOME=$HF_HOME${C_OFF}"

run_step "sanity"           critical step_sanity
run_step "repo"             critical step_repo
run_step "uv+hf-cli"        critical step_uv_hfcli
export PATH="$HOME/.local/bin:$PATH"
run_step "miniforge"        critical step_conda
[ -n "$CONDA_BASE" ] || CONDA_BASE=$( { command -v conda >/dev/null && conda info --base; } || echo "$HOME/miniforge3" )

run_step "env-piar"         soft step_env_piar
run_step "env-webshop"      soft step_env_webshop

if [ "$SKIP_DATASETS" = 1 ]; then mark_skipped "datasets+index"; else
    run_step "datasets+index" soft step_datasets
fi
if [ "$SKIP_MODELS" = 1 ]; then mark_skipped "modelos"; else
    run_step "modelos"        soft step_models
fi

run_step "verify-piar"      soft step_verify_piar
run_step "verify-webshop"   soft step_verify_webshop
run_step "verify-figura1"   soft step_verify_figura1

print_summary
FAILS=0
for s in "${STEP_STATUS[@]}"; do [ "$s" = FAIL ] && FAILS=$((FAILS+1)); done
if [ "$FAILS" -eq 0 ]; then
    print_next_steps
else
    exit 1
fi
