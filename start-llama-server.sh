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
# With ctx=32768 and parallel=4, each user gets ~8192 tokens. Increase if you have VRAM.
LLAMA_PARALLEL="${LLAMA_PARALLEL:-4}"

# ── Validate binary path ────────────────────────────────────────────────────
if [[ -z "$LLAMA_SERVER_BIN" ]]; then
  echo "ERROR: LLAMA_SERVER_BIN is not set."
  echo ""
  echo "  Set the absolute path to your llama-server binary in .env:"
  echo "    LLAMA_SERVER_BIN=/home/you/llama.cpp/build/bin/llama-server"
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

# ── VRAM budget planner ───────────────────────────────────────────────────────
# Gemma-4-E4B Q4_K_M approximate VRAM breakdown:
#   Model weights : ~3.0 GB
#   mmproj        : ~0.4 GB
#   KV cache      : ~(ctx_size * 340 KB) → heavily depends on ctx and parallel
#   OS/driver     : ~0.5 GB overhead
#
# Formula (conservative): VRAM_needed ≈ 3.4 + 0.5 + (CTX_SIZE / 1024 * 0.33) GB
#
# Examples:
#   ctx=32768  np=4 → ~14.5 GB total (needs 16 GB GPU)
#   ctx=16384  np=4 → ~9.0  GB total (fits 12 GB GPU comfortably)
#   ctx=8192   np=4 → ~6.3  GB total (fits 8 GB GPU)
#   ctx=8192   np=2 → ~6.3  GB total (same — KV cache is total, not per-slot)
#   ctx=4096   np=4 → ~5.0  GB total (fits most GPUs)
#
# NOTE: --parallel N splits the context window across N slots.
#       Each user gets ctx_size/N tokens. So ctx=16384 np=4 → 4096 per user.
#       For RAG workloads (short queries + retrieved context), 4096 per user is fine.

_estimate_vram() {
  local ctx=$1
  # Rough estimate in GB: model_base + ctx_contribution
  local model_base="3.9"
  local ctx_gb
  ctx_gb=$(awk "BEGIN {printf \"%.1f\", $ctx / 1024 * 0.33}")
  local total
  total=$(awk "BEGIN {printf \"%.1f\", $model_base + $ctx_gb}")
  echo "$total"
}

ESTIMATED_VRAM=$(_estimate_vram "$LLAMA_CTX_SIZE")

# Detect available GPU VRAM (nvidia-smi)
DETECTED_VRAM=""
if command -v nvidia-smi &>/dev/null; then
  DETECTED_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
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
echo ""
echo "  VRAM estimate : ~${ESTIMATED_VRAM} GB needed"
if [[ -n "$DETECTED_VRAM" ]]; then
  DETECTED_GB=$(awk "BEGIN {printf \"%.1f\", $DETECTED_VRAM / 1024}")
  echo "  GPU VRAM      : ${DETECTED_GB} GB detected"
  # Warn if estimate exceeds detected VRAM
  OVER=$(awk "BEGIN {print ($ESTIMATED_VRAM > $DETECTED_GB) ? 1 : 0}")
  if [[ "$OVER" == "1" ]]; then
    echo ""
    echo "  ⚠  WARNING: Estimated VRAM (~${ESTIMATED_VRAM} GB) exceeds GPU capacity (${DETECTED_GB} GB)!"
    echo "     Recommendations:"
    echo "       - Reduce context: LLAMA_CTX_SIZE=16384  (or 8192 for tight GPUs)"
    echo "       - Reduce parallel: LLAMA_PARALLEL=2"
    echo "       - Example safe config for ${DETECTED_GB} GB GPU:"
    if awk "BEGIN {exit ($DETECTED_GB >= 12) ? 0 : 1}"; then
      echo "         LLAMA_CTX_SIZE=16384  LLAMA_PARALLEL=4  (per-user: 4096 tokens)"
    elif awk "BEGIN {exit ($DETECTED_GB >= 8) ? 0 : 1}"; then
      echo "         LLAMA_CTX_SIZE=8192   LLAMA_PARALLEL=2  (per-user: 4096 tokens)"
    else
      echo "         LLAMA_CTX_SIZE=4096   LLAMA_PARALLEL=2  (per-user: 2048 tokens)"
    fi
    echo ""
    echo "  Continuing anyway in 5 seconds... (Ctrl+C to abort)"
    sleep 5
  fi
else
  echo "  GPU VRAM      : (nvidia-smi not found — cannot detect)"
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
