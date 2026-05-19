#!/bin/bash
set -euo pipefail

# ── Load .env if present ─────────────────────────────────────────────────────
if [[ -f .env ]]; then
  set -a
  source <(grep -v '^\s*#' .env | grep -v '^\s*$')
  set +a
fi

# ── Configuration (all overridable via .env or environment) ──────────────────
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"
LLAMA_HF_REPO="${LLAMA_HF_REPO:-lmstudio-community/gemma-4-E4B-it-GGUF:Q4_K_M}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-16384}"
LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:-auto}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
# Optional multimodal projector — enables image/audio input.
# NOTE: When using `-hf/--hf-repo`, llama-server downloads a compatible mmproj automatically
# (unless `--no-mmproj` is passed). Therefore, in the common case you should leave this empty.
# If you want to force a local mmproj file, set this to an absolute path and we'll pass `--mmproj`.
LLAMA_MMPROJ="${LLAMA_MMPROJ:-}"
# Parallel inference slots — each slot gets ctx_size / parallel tokens of context.
# Mac/local default is single-user lowest latency; increase only for concurrent users.
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"

OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
ARCH_NAME="$(uname -m 2>/dev/null || echo unknown)"
IS_APPLE_SILICON=0
if [[ "$OS_NAME" == "Darwin" && "$ARCH_NAME" == "arm64" ]]; then
  IS_APPLE_SILICON=1
fi

# ── Validate binary path ────────────────────────────────────────────────────
if [[ -z "$LLAMA_SERVER_BIN" ]]; then
  echo "ERROR: LLAMA_SERVER_BIN is not set."
  echo ""
  echo "  Set the absolute path to your llama-server binary in .env:"
  echo "    LLAMA_SERVER_BIN=/Users/you/llama.cpp/build/bin/llama-server"
  echo ""
  echo "  Or export it before running this script:"
  echo "    export LLAMA_SERVER_BIN=/path/to/llama-server"
  exit 1
fi

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "ERROR: Binary not found or not executable: $LLAMA_SERVER_BIN"
  echo "  Did you build llama.cpp?  cmake --build build --config Release"
  exit 1
fi

# ── Memory budget planner ─────────────────────────────────────────────────────
# Apple Silicon uses unified memory; there is no separate NVIDIA-style VRAM pool.
# Gemma-4-E4B Q4_K_M approximate memory breakdown:
#   Model weights : ~3.0 GB
#   mmproj        : ~0.4 GB
#   KV cache      : ~(ctx_size * 340 KB) -> heavily depends on ctx
#   Runtime/OS    : ~0.5 GB overhead
#
# Formula (conservative): memory_needed ~= 3.4 + 0.5 + (CTX_SIZE / 1024 * 0.33) GB
#
# Examples:
#   ctx=32768  np=1 -> ~14.5 GB total
#   ctx=16384  np=1 -> ~9.0  GB total
#   ctx=8192   np=1 -> ~6.3  GB total
#   ctx=4096   np=1 -> ~5.0  GB total
#
# NOTE: --parallel N splits the context window across N slots.
#       For one local developer/user, LLAMA_PARALLEL=1 preserves the full context.

_estimate_memory() {
  local ctx=$1
  # Rough estimate in GB: model_base + ctx_contribution
  local model_base="3.9"
  local ctx_gb
  ctx_gb=$(awk "BEGIN {printf \"%.1f\", $ctx / 1024 * 0.33}")
  local total
  total=$(awk "BEGIN {printf \"%.1f\", $model_base + $ctx_gb}")
  echo "$total"
}

ESTIMATED_MEMORY=$(_estimate_memory "$LLAMA_CTX_SIZE")

# Detect available memory. On Apple Silicon this is unified memory; elsewhere,
# nvidia-smi is only used when available.
DETECTED_MEMORY_GB=""
MEMORY_LABEL="system memory"
if [[ "$IS_APPLE_SILICON" == "1" ]]; then
  DETECTED_MEMORY_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f", $1 / 1024 / 1024 / 1024}')
  MEMORY_LABEL="unified memory"
elif command -v nvidia-smi &>/dev/null; then
  DETECTED_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [[ -n "$DETECTED_VRAM" ]]; then
    DETECTED_MEMORY_GB=$(awk "BEGIN {printf \"%.1f\", $DETECTED_VRAM / 1024}")
    MEMORY_LABEL="GPU VRAM"
  fi
fi

# ── Launch ───────────────────────────────────────────────────────────────────
echo "llama-server configuration:"
echo "  binary     : $LLAMA_SERVER_BIN"
echo "  model      : $LLAMA_HF_REPO"
echo "  mmproj     : ${LLAMA_MMPROJ:-auto (hf-repo)}"
echo "  ctx_size   : $LLAMA_CTX_SIZE"
echo "  gpu_layers : $LLAMA_GPU_LAYERS"
echo "  port       : $LLAMA_PORT"
echo "  parallel   : $LLAMA_PARALLEL"
echo "  per-user   : $(( LLAMA_CTX_SIZE / LLAMA_PARALLEL )) tokens"
echo "  endpoint   : http://localhost:${LLAMA_PORT}/v1"
if [[ "$IS_APPLE_SILICON" == "1" ]]; then
  echo "  platform   : Apple Silicon / Metal (${ARCH_NAME})"
else
  echo "  platform   : ${OS_NAME}/${ARCH_NAME}"
fi
echo ""
echo "  memory estimate : ~${ESTIMATED_MEMORY} GB needed"
if [[ -n "$DETECTED_MEMORY_GB" ]]; then
  echo "  ${MEMORY_LABEL} : ${DETECTED_MEMORY_GB} GB detected"
  # Warn if estimate is tight for the detected memory pool.
  LIMIT=$(awk "BEGIN {printf \"%.1f\", $DETECTED_MEMORY_GB * 0.75}")
  OVER=$(awk "BEGIN {print ($ESTIMATED_MEMORY > $LIMIT) ? 1 : 0}")
  if [[ "$OVER" == "1" ]]; then
    echo ""
    echo "  WARNING: Estimated model memory (~${ESTIMATED_MEMORY} GB) is tight for ${MEMORY_LABEL} (${DETECTED_MEMORY_GB} GB)."
    echo "     Recommendations:"
    echo "       - Reduce context: LLAMA_CTX_SIZE=8192"
    echo "       - Keep single-user parallelism: LLAMA_PARALLEL=1"
    echo "       - Close memory-heavy apps before launching."
    echo "       - Example safe config for ${DETECTED_MEMORY_GB} GB ${MEMORY_LABEL}:"
    if awk "BEGIN {exit ($DETECTED_MEMORY_GB >= 16) ? 0 : 1}"; then
      echo "         LLAMA_CTX_SIZE=16384  LLAMA_PARALLEL=1"
    else
      echo "         LLAMA_CTX_SIZE=8192   LLAMA_PARALLEL=1"
    fi
    echo ""
    echo "  Continuing anyway in 5 seconds... (Ctrl+C to abort)"
    sleep 5
  fi
else
  echo "  memory detected : unavailable"
fi
echo ""

MMPROJ_ARGS=()
if [[ -n "$LLAMA_MMPROJ" ]]; then
  if [[ -f "$LLAMA_MMPROJ" ]]; then
    MMPROJ_ARGS=("--mmproj" "$LLAMA_MMPROJ")
  else
    echo "WARNING: LLAMA_MMPROJ is set but file not found: $LLAMA_MMPROJ"
    echo "         Ignoring LLAMA_MMPROJ and relying on hf-repo auto mmproj download."
  fi
fi

exec "$LLAMA_SERVER_BIN" \
  -hf "$LLAMA_HF_REPO" \
  ${MMPROJ_ARGS[@]+"${MMPROJ_ARGS[@]}"} \
  -c "$LLAMA_CTX_SIZE" \
  -ngl "$LLAMA_GPU_LAYERS" \
  --parallel "$LLAMA_PARALLEL" \
  --cont-batching \
  --jinja \
  --host 0.0.0.0 \
  --port "$LLAMA_PORT"
