"""Inference backends for the native Anthropic server.

The server itself is backend-agnostic: it owns the Anthropic protocol, the
tool-call parsing/recovery and the prompt shaping, and delegates "load a model,
tokenize, stream tokens" to one of these.

    mlx    — Apple Silicon, mlx-lm (the original, still the default on macOS)
    torch  — everything else: NVIDIA/CUDA, AMD/ROCm, or plain CPU, via
             transformers

Pick one with LLM_BACKEND=mlx|torch|auto. `auto` (the default) uses MLX when
mlx-lm imports, and torch otherwise — so an Apple Silicon install behaves
exactly as it always has, and a Linux box works without any flags.
"""

import os
import sys

BACKEND_ENV = "LLM_BACKEND"


def resolve_backend_name(name=None):
    """Turn LLM_BACKEND (or an explicit name) into a concrete backend name."""
    name = (name or os.environ.get(BACKEND_ENV) or "auto").strip().lower()
    if name != "auto":
        return name
    if sys.platform == "darwin":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "torch"


def get_backend(name=None, log=None):
    """Instantiate the requested backend. Raises SystemExit with a usable
    message when the chosen backend's runtime isn't installed, because a
    traceback about `mlx.core` on a Linux box tells the user nothing."""
    resolved = resolve_backend_name(name)
    if resolved == "mlx":
        try:
            from .mlx_backend import MLXBackend
        except ImportError as e:
            raise SystemExit(
                f"LLM_BACKEND=mlx but mlx-lm is not importable ({e}).\n"
                "MLX only runs on Apple Silicon. On Linux/Windows use "
                "LLM_BACKEND=torch (pip install torch transformers)."
            )
        return MLXBackend(log=log)
    if resolved == "torch":
        try:
            from .torch_backend import TorchBackend
        except ImportError as e:
            raise SystemExit(
                f"LLM_BACKEND=torch but transformers/torch are not importable ({e}).\n"
                "Install them with: pip install torch transformers accelerate"
            )
        return TorchBackend(log=log)
    raise SystemExit(f"Unknown {BACKEND_ENV}={resolved!r} (expected mlx, torch or auto)")
