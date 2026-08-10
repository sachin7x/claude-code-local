# 🐧 Linux — NVIDIA CUDA, AMD ROCm, or CPU

MLX only exists on Apple Silicon. The rest of this project doesn't care:
`proxy/server.py` speaks the Anthropic Messages API, recovers garbled tool
calls and shapes the prompt in plain Python, and hands "load a model, tokenize,
stream tokens" to a backend.

| Backend | Runs on | Engine |
|---|---|---|
| `mlx` | Apple Silicon | [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| `torch` | NVIDIA CUDA · AMD ROCm · CPU | [transformers](https://github.com/huggingface/transformers) |

`LLM_BACKEND=auto` (the default) picks `mlx` when it imports and `torch`
otherwise, so nothing changes for an existing Mac install.

---

## 🚀 Setup

```bash
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
bash setup-linux.sh
```

It detects your GPU with `nvidia-smi`, installs torch + transformers into
`~/.local/llm-server`, downloads a model sized for your VRAM, symlinks the
server into `~/.local/native-llm-server/`, and writes a `claude-local`
launcher to `~/.local/bin`.

```bash
claude-local          # boots the server if it isn't up, then runs Claude Code
```

Override the model, or install without downloading anything:

```bash
LLM_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct bash setup-linux.sh
LLM_SKIP_DOWNLOAD=1 bash setup-linux.sh
```

### What it picks for you

| Total VRAM | Model | Precision |
|---|---|---|
| 48 GB+ | Qwen 2.5 Coder 32B | bf16 |
| 24-48 GB | Qwen 2.5 Coder 32B | 4-bit (bitsandbytes) |
| 16-24 GB | Qwen 2.5 Coder 14B | 4-bit |
| 8-16 GB | Qwen 2.5 Coder 7B | 4-bit |
| < 8 GB | Qwen 2.5 Coder 3B | 4-bit |
| no GPU | Qwen 2.5 Coder 3B | fp32 on CPU — works, single-digit tok/s |

Multi-GPU: VRAM is summed and `device_map="auto"` shards the model across
every card.

---

## 🔧 Running the server by hand

```bash
LLM_BACKEND=torch \
LLM_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct \
LLM_LOAD_IN_4BIT=1 \
~/.local/llm-server/bin/python proxy/server.py
```

Then point Claude Code at it:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_API_KEY=sk-local claude
```

`curl localhost:4000/health` reports the backend and device it actually came
up on:

```json
{"status":"ok","model":"Qwen/Qwen2.5-Coder-14B-Instruct","backend":"torch",
 "device":"NVIDIA GeForce RTX 4090 (24GB)","cuda_allocated_gb":9.4,
 "cuda_reserved_gb":10.1,"cuda_peak_gb":11.8}
```

---

## ⚙️ Environment variables

| Variable | Default | What it does |
|---|---|---|
| `LLM_BACKEND` | `auto` | `mlx`, `torch`, or `auto` |
| `LLM_MODEL` | per backend | HuggingFace id or local path (`MLX_MODEL` still works) |
| `LLM_PORT` | `4000` | Port to serve on (`MLX_PORT` still works) |
| `LLM_MAX_TOKENS` | `8192` | Default cap when the request doesn't set one |
| `LLM_DEVICE` | `auto` | `cuda`, `cpu`, `mps`, or `auto` |
| `LLM_DTYPE` | `auto` | `bfloat16`, `float16`, `float32`, or `auto` |
| `LLM_LOAD_IN_4BIT` | off | 4-bit NF4 via bitsandbytes (CUDA) |
| `LLM_LOAD_IN_8BIT` | off | 8-bit via bitsandbytes (CUDA) |
| `LLM_PROMPT_CACHE` | `1` | KV reuse across requests; `0` to disable |

Every `MLX_*` variable still applies to the MLX backend, unchanged.

---

## 🧠 Prompt caching

Claude Code re-sends the whole transcript every turn, so the expensive part is
prefilling a prompt the server has already seen. Both backends reuse it: the
KV cache is cropped to the prefix the new prompt shares with the last one and
only the delta is prefilled.

```
[10:14:02]   Prompt: 8431 tokens
[10:14:02]   Cache hit: 8102 reused, 329 new tokens to prefill (saved 8102 tokens)
```

Set `LLM_PROMPT_CACHE=0` if you suspect the cache and want a clean prefill
every turn.

---

## 🩺 Troubleshooting

**`LLM_BACKEND=torch but transformers/torch are not importable`** — install
them into the venv you're running the server with:
`~/.local/llm-server/bin/pip install torch transformers accelerate`.

**CUDA out of memory** — drop to a smaller model or add `LLM_LOAD_IN_4BIT=1`.
The KV cache for a 20k-token Claude Code prompt is not small; leave a few GB
of headroom beyond the weights.

**bitsandbytes errors on ROCm** — 4-bit is CUDA-only. Use bf16 with a smaller
model instead.

**The model answers, but never calls a tool** — that's a model problem, not a
backend one. See [TOOL-CALL-RELIABILITY.md](TOOL-CALL-RELIABILITY.md); use an
instruct model trained for tool use (the Qwen 2.5 Coder line is a good start).

---

[← back to the README](../README.md)
