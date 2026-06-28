#!/usr/bin/env bash
# =============================================================================
# Diagnostic Percept — single self-contained server runner.
#
# scp THIS ONE FILE to the server and run it. It:
#   1. installs everything (system deps + a kept Python venv + CUDA torch),
#   2. clones the repo,
#   3. runs the WHOLE pipeline on one GPU  (discovery H1-H5 -> H6 benchmark +
#      consensus -> H7 scale -> H6 pass-2 -> H8 -> sycophancy),
#   4. runs the level-by-level analysis -> a single report,
#   5. moves the results to a safe folder, then DELETES the downloaded model,
#      caches and tmp (the venv/packages are kept), and
#   6. prints the path to the results.
#
# Designed for a single GPU (e.g. one H200 MIG slice, ~71 GB). It auto-detects
# the GPU and uses NF4 4-bit Qwen3 so it fits comfortably and never OOMs.
#
# Credentials: NONE are required (Qwen3 is open weights, the repo is public).
# You will be asked for an OPTIONAL HuggingFace token only to lift download
# rate limits — just press Enter to skip.
#
# IMPORTANT: this is a multi-hour run. Run it under tmux or nohup so an SSH
# disconnect doesn't kill it:
#       tmux new -s dp 'bash run_on_server.sh'      # then Ctrl-b d to detach
#   or  nohup bash run_on_server.sh > dp.log 2>&1 & ; tail -f dp.log
#
# Tunables (prefix the command, e.g.  N_BENCH=300 bash run_on_server.sh):
#   MODEL      default Qwen/Qwen3-32B   (use Qwen/Qwen3-14B for ~2x faster)
#   N_BENCH    default 1273 (full MedQA test); lower for a quick first pass
#   USE_4BIT   default 1 (NF4). Set 0 for bf16 if you have >70 GB free.
#   KEEP_MODEL default 0 (delete weights at the end). Set 1 to keep for re-runs.
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/ArioMoniri/diagnosticpercept.git"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
N_BENCH="${N_BENCH:-1273}"
USE_4BIT="${USE_4BIT:-1}"
KEEP_MODEL="${KEEP_MODEL:-0}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"   # wheel build; cu124 runs fine on a 12.8 driver

WORK="${WORK:-$HOME/dp_work}"                       # deleted at the end (model+repo+tmp)
VENV="${VENV:-$HOME/dp_venv}"                       # kept (packages)
FINAL="${FINAL:-$HOME/diagnosticpercept_results}"   # kept (the results + report)
mkdir -p "$FINAL"
trap cleanup EXIT      # delete model on success OR failure (guarded inside cleanup)

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# Cleanup runs on EVERY exit (success or failure) so the downloaded model never
# lingers — UNLESS KEEP_MODEL=1, or results weren't produced, or WORK looks
# dangerous. Registered after the vars are set (below).
cleanup() {
  local rc=$?
  [ "${KEEP_MODEL:-0}" = "1" ] && { echo "KEEP_MODEL=1 — leaving $WORK."; return 0; }
  case "${WORK:-}" in
    ""|"/"|"$HOME"|"$HOME/"|"/root"|"/home") echo "refusing to rm WORK='${WORK:-<empty>}'"; return 0 ;;
  esac
  # Only delete once results actually landed in FINAL — otherwise keep WORK so
  # a re-run can resume instead of recomputing from nothing.
  if [ -n "$(ls -A "$FINAL" 2>/dev/null)" ]; then
    echo "Cleanup: removing model + caches + tmp + checkout ($WORK) ..."
    rm -rf "$WORK"
    echo "Removed $WORK. (venv kept at $VENV; results kept at $FINAL)"
  else
    echo "Cleanup: $FINAL is empty — keeping $WORK for a resume."
  fi
}

say "Diagnostic Percept server run — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "MODEL=$MODEL  N_BENCH=$N_BENCH  USE_4BIT=$USE_4BIT  KEEP_MODEL=$KEEP_MODEL"
echo "WORK=$WORK  VENV=$VENV  FINAL=$FINAL"

# --- GPU check ---------------------------------------------------------------
say "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || nvidia-smi -L || true
else
  echo "!! nvidia-smi not found — is this a GPU box? Continuing, but the model load will fail without CUDA."
fi

# --- optional HF token -------------------------------------------------------
say "Credentials (optional)"
if [ -z "${HF_TOKEN:-}" ] && [ -t 0 ]; then
  echo "Qwen3 is open — no token needed. A HuggingFace token only lifts download"
  echo "rate limits. Paste one (hf_...) or just press Enter to skip:"
  read -r -p "HF_TOKEN: " HF_TOKEN || true
fi
export HF_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] && echo "Using provided HF token." || echo "No HF token (fine)."

# --- system deps -------------------------------------------------------------
say "System packages"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y -qq || true
  apt-get install -y -qq git python3 python3-venv python3-pip curl ca-certificates >/dev/null 2>&1 \
    || echo "!! apt install had warnings — continuing (tools may already be present)."
fi
command -v git >/dev/null 2>&1 || die "git is required but not installed."
command -v python3 >/dev/null 2>&1 || die "python3 is required but not installed."

# --- python venv (kept) ------------------------------------------------------
say "Python environment ($VENV)"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools

# torch first, from the CUDA index; then the rest. Qwen3 is supported in
# stable transformers (>=4.51) so no git-main needed — much more reliable.
say "Installing PyTorch ($TORCH_CUDA) + libs (first run only, a few minutes)"
if ! python -c "import torch, bitsandbytes, transformers, datasets, scipy, sklearn, matplotlib" >/dev/null 2>&1; then
  # CUDA torch from the official index. NO CPU fallback — a CPU build would run
  # the whole pipeline at unusable speed; better to fail loudly here.
  python -m pip install -q --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" torch \
    || die "torch ($TORCH_CUDA) install failed. Check network / try TORCH_CUDA=cu121 or cu126."
  python -m pip install -q \
    "transformers>=4.53" "tokenizers>=0.20" "accelerate>=0.34" "bitsandbytes>=0.43" \
    "datasets>=2.20" "scipy>=1.11" "scikit-learn>=1.3" "matplotlib>=3.7" "tqdm>=4.66" \
    "safetensors>=0.4" "huggingface_hub>=0.24"
fi
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda)"
# Hard gate: a CPU-only torch here means the install silently degraded — abort
# rather than crawl for hours or die deep inside model load.
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "torch reports CUDA unavailable. A CPU build was installed or the GPU isn't visible. Aborting before the long run."

# --- clone (or reuse for resume) --------------------------------------------
say "Repository"
mkdir -p "$WORK"
REPO="$WORK/diagnosticpercept"
if [ -d "$REPO/.git" ]; then
  echo "Reusing existing checkout (resume) — git pull"
  git -C "$REPO" pull --ff-only || true
else
  git clone --depth 1 "$REPO_URL" "$REPO"
fi
GIT_SHA="$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo @ $GIT_SHA"

# --- environment for the run -------------------------------------------------
# Caches live under WORK so the cleanup step removes the downloaded model.
export HF_HOME="$WORK/hf"
export HF_HUB_CACHE="$WORK/hf"
export TRANSFORMERS_CACHE="$WORK/hf"
export HF_DATASETS_CACHE="$WORK/hf/datasets"
export TORCH_HOME="$WORK/torch"
export TMPDIR="$WORK/tmp"
export TRITON_CACHE_DIR="$WORK/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
mkdir -p "$HF_HOME" "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# Model selection: force the chosen Qwen3 + NF4 so it fits one GPU.
export MODEL_OVERRIDE="$MODEL"
export USE_4BIT="$USE_4BIT"
export N_BENCH="$N_BENCH"
export RESULTS_DIR="$REPO/results"
export DP_GIT_SHA="$GIT_SHA"

# --- run the pipeline --------------------------------------------------------
say "Running the full pipeline (this is the long part)"
cd "$REPO"
# Bootstrap logs use a distinct prefix so they can never be clobbered by files
# the pipeline itself writes under results/.
set +e
python scripts/run_all.py 2>&1 | tee "$FINAL/_bootstrap_run.log"
RUN_RC=${PIPESTATUS[0]}
set -e
echo "run_all.py process exit code: $RUN_RC  (note: stage failures are logged, not fatal)"

# --- analyze -----------------------------------------------------------------
say "Level-by-level analysis"
python scripts/analyze_all.py "$RESULTS_DIR" 2>&1 | tee "$FINAL/_bootstrap_analysis.log" || true

# --- collect results (kept) — MUST succeed before the trap deletes WORK ------
say "Collecting results -> $FINAL"
[ -d "$RESULTS_DIR" ] || die "no results dir at $RESULTS_DIR — keeping WORK, not deleting anything."
cp -a "$RESULTS_DIR/." "$FINAL/" || die "results copy to $FINAL failed — keeping WORK so nothing is lost."
echo "Results copied to $FINAL"
deactivate 2>/dev/null || true

# --- done (cleanup runs automatically via the EXIT trap) ---------------------
say "FINISHED"
echo "Pipeline process exit: $RUN_RC  (check the report for any per-stage failures)"
echo
echo "RESULTS  : $FINAL"
echo "REPORT   : $FINAL/analysis/report.md"
echo "JSON     : $FINAL/analysis/report.json"
echo "RUN LOG  : $FINAL/_bootstrap_run.log"
echo
echo "Quick look:  sed -n '1,90p' $FINAL/analysis/report.md"
