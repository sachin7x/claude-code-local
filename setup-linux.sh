#!/bin/bash
# Claude Code Local — Linux setup (NVIDIA CUDA, AMD ROCm, or CPU)
#
# The macOS half of this project is MLX. Everything else — the Anthropic
# server, the tool-call recovery, the launchers — is plain Python, so this
# script sets the same thing up on Linux with the torch backend.
#
# Usage:
#   bash setup-linux.sh                     # pick a model for your GPU
#   LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct bash setup-linux.sh
#   LLM_SKIP_DOWNLOAD=1 bash setup-linux.sh # install + launcher only

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Claude Code Local — Linux Setup              ║"
echo "║     Same server, torch instead of MLX.           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

if [ "$(uname -s)" != "Linux" ]; then
  echo "ERROR: this is the Linux setup. On an Apple Silicon Mac run: bash setup.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${LLM_VENV:-$HOME/.local/llm-server}"
SERVER_DIR="$HOME/.local/native-llm-server"
BIN_DIR="$HOME/.local/bin"

# ── System detection ──────────────────────────────────────────
MEM_GB=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
VRAM_GB=0
GPU_NAME=""
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  # Sum every card: device_map="auto" will shard across all of them.
  VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
            | awk '{s+=$1} END {print int(s/1024)}')
fi

echo "Detected: ${GPU_NAME:-no NVIDIA GPU}"
echo "VRAM:     ${VRAM_GB} GB"
echo "RAM:      ${MEM_GB} GB"
echo ""

# ── Python venv ───────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
  echo "ERROR: $PYTHON_BIN not found. Install Python 3.10+ first."
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "Creating virtualenv at $VENV..."
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip

if [ "$VRAM_GB" -gt 0 ]; then
  echo "Installing torch (CUDA build) + transformers..."
  "$VENV/bin/pip" install --quiet torch
  "$VENV/bin/pip" install --quiet transformers accelerate bitsandbytes
else
  echo "No CUDA GPU — installing the CPU build of torch (generation will be slow)..."
  "$VENV/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/pip" install --quiet transformers accelerate
fi

# ── Pick a model ──────────────────────────────────────────────
# 4-bit weights are ~0.6 GB per billion params; leave room for the KV cache of
# a Claude Code prompt, which is where the "and then it OOMed" reports come from.
if [ -n "$LLM_MODEL" ]; then
  MODEL_ID="$LLM_MODEL"
  MODEL_LABEL="$LLM_MODEL (from \$LLM_MODEL)"
  QUANT=""
elif [ "$VRAM_GB" -ge 48 ]; then
  MODEL_ID="Qwen/Qwen2.5-Coder-32B-Instruct"
  MODEL_LABEL="Qwen 2.5 Coder 32B (bf16)"
  QUANT=""
elif [ "$VRAM_GB" -ge 24 ]; then
  MODEL_ID="Qwen/Qwen2.5-Coder-32B-Instruct"
  MODEL_LABEL="Qwen 2.5 Coder 32B (4-bit)"
  QUANT="4bit"
elif [ "$VRAM_GB" -ge 16 ]; then
  MODEL_ID="Qwen/Qwen2.5-Coder-14B-Instruct"
  MODEL_LABEL="Qwen 2.5 Coder 14B (4-bit)"
  QUANT="4bit"
elif [ "$VRAM_GB" -ge 8 ]; then
  MODEL_ID="Qwen/Qwen2.5-Coder-7B-Instruct"
  MODEL_LABEL="Qwen 2.5 Coder 7B (4-bit)"
  QUANT="4bit"
elif [ "$VRAM_GB" -gt 0 ]; then
  MODEL_ID="Qwen/Qwen2.5-Coder-3B-Instruct"
  MODEL_LABEL="Qwen 2.5 Coder 3B (4-bit)"
  QUANT="4bit"
else
  MODEL_ID="Qwen/Qwen2.5-Coder-3B-Instruct"
  MODEL_LABEL="Qwen 2.5 Coder 3B (CPU — expect single-digit tok/s)"
  QUANT=""
fi

echo ""
echo "Selected: $MODEL_LABEL"
echo "Model ID: $MODEL_ID"
echo ""

# ── Download weights ──────────────────────────────────────────
if [ "${LLM_SKIP_DOWNLOAD:-0}" != "1" ]; then
  echo "Downloading $MODEL_ID (one time)..."
  "$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_ID", allow_patterns=[
    "*.json", "*.safetensors", "*.model", "*.txt", "*.py"])
print("Done.")
PY
fi

# ── Install the server ────────────────────────────────────────
# A symlink, not a copy: edit proxy/server.py in the repo and the change takes
# effect on the next restart. server.py resolves its own realpath, so it finds
# proxy/backends/ next to the real file rather than next to the symlink.
mkdir -p "$SERVER_DIR" "$BIN_DIR"
ln -sf "$SCRIPT_DIR/proxy/server.py" "$SERVER_DIR/server.py"
echo "Server installed (symlink) → $SERVER_DIR/server.py -> $SCRIPT_DIR/proxy/server.py"

# ── Launcher ──────────────────────────────────────────────────
CLAUDE_BIN=$(command -v claude || echo "$HOME/.local/bin/claude")
if [ ! -x "$CLAUDE_BIN" ]; then
  echo ""
  echo "WARNING: Claude Code not found. Install it with:"
  echo "  npm install -g @anthropic-ai/claude-code"
  echo ""
fi

LAUNCHER="$BIN_DIR/claude-local"
cat > "$LAUNCHER" <<LAUNCH
#!/bin/bash
# Claude Code — local model ($MODEL_LABEL)
CLAUDE_BIN="$CLAUDE_BIN"
PYTHON="$VENV/bin/python"
SERVER="$SERVER_DIR/server.py"
PORT="\${LLM_PORT:-4000}"

if ! curl -s "http://localhost:\$PORT/health" 2>/dev/null | grep -q '"ok"'; then
  LLM_BACKEND=torch \\
  LLM_MODEL="$MODEL_ID" \\
  LLM_PORT="\$PORT" \\
  ${QUANT:+LLM_LOAD_IN_4BIT=1 \\
  }"\$PYTHON" "\$SERVER" >/tmp/llm-server.log 2>&1 &
  echo "  Loading $MODEL_LABEL..."
  until curl -s "http://localhost:\$PORT/health" 2>/dev/null | grep -q '"ok"'; do
    sleep 2
  done
fi

echo ""
echo "  → Claude Code with LOCAL AI"
echo "  → $MODEL_LABEL"
echo "  → 100% on-device, no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL="http://localhost:\$PORT" \\
ANTHROPIC_API_KEY=sk-local \\
exec "\$CLAUDE_BIN" --model claude-sonnet-4-6
LAUNCH
chmod +x "$LAUNCHER"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Setup complete!                              ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Model:    $MODEL_ID"
echo "║  Server:   $SERVER_DIR/server.py"
echo "║  Launcher: $LAUNCHER"
echo "║"
echo "║  Run 'claude-local' to start coding with local"
echo "║  AI (add ~/.local/bin to PATH if it isn't there)."
echo "╚══════════════════════════════════════════════════╝"
echo ""
