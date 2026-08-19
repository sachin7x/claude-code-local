"""MLX backend — Apple Silicon, unified memory, Metal.

This is the original inference path, lifted out of server.py unchanged: same
env vars, same prompt-cache trimming, same memory accounting.
"""

import os
import subprocess
import time

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache, RotatingKVCache

from .base import Backend, Chunk

GB = 1024 ** 3


def env_int(name, default):
    """int() an env var, treating unset *and empty* as "use the default"."""
    raw = os.environ.get(name, "")
    return int(raw) if raw.strip() else default


def psutil_rss_gb():
    """RSS without a psutil dependency — the mlx venv doesn't ship one."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 / 1024 if out else 0.0


class MLXBackend(Backend):
    name = "mlx"
    device_label = "Apple Silicon GPU (Metal)"

    def __init__(self, log=None):
        super().__init__(log=log)
        self.model_path = None
        # MLX_PREFILL_SIZE used to default to 8192, which quietly TRIPLES peak
        # memory on a long prompt and is the reason a big Claude Code request
        # could take the machine into swap (or straight into the OOM killer on
        # a 32GB Mac) while the model itself fits fine.
        #
        # mlx_lm's prefill loop runs the WHOLE model on each chunk, lm_head
        # included, so a single 8192-token chunk allocates an (8192 x vocab)
        # bf16 logits tensor it immediately throws away — 4.3GB at Gemma's
        # 262144 vocab. On top of that the full attention layers score the
        # entire chunk against every key so far. None of it is reused, all of
        # it is live at once.
        #
        # Measured on gemma-4-31b-it-abliterated-4bit-mlx (16.7GB of weights
        # resident), same prompt each time, via mx.get_peak_memory():
        #
        #     prompt    prefill 8192      prefill 1024      prefill 512
        #     21.4k     34.2GB / 49.3s    20.9GB / 37.7s    -      / 39.9s
        #     38.5k     -      / -        23.5GB / 83.1s    21.9GB / 71.0s
        #
        # 17.5GB of transients on a 21k prompt, against 4.2GB for the same work
        # at 512. Small chunks are not slower either — the GPU saturates far
        # below 8192 tokens — so nothing was ever traded away for that memory.
        self.prefill_size = env_int("MLX_PREFILL_SIZE", 512)
        # Gemma 4's RotatingKVCache doesn't support quantization.
        self.kv_bits = env_int("MLX_KV_BITS", 0)
        self.kv_quant_start = env_int("MLX_KV_QUANT_START", 256)
        # Ceiling on MLX's buffer recycle pool, which counts toward this
        # process's memory. Measured at 0.0GB on the model above either way, so
        # treat it as a backstop for larger ones (Llama 70B 8-bit) rather than
        # a fix on its own. MLX_CACHE_LIMIT_GB=0 disables the cap.
        self.cache_limit_gb = float(os.environ.get("MLX_CACHE_LIMIT_GB") or 3)
        self._prompt_cache = None
        self._cached_token_prefix = None

    # ─── Load ────────────────────────────────────────────────────────────────

    def load(self, model_path):
        self.model_path = model_path
        if self.cache_limit_gb > 0:
            mx.set_cache_limit(int(self.cache_limit_gb * GB))
            self.log(f"MLX buffer cache capped at {self.cache_limit_gb:g}GB")
        self.log(f"Loading model: {model_path}")
        t0 = time.time()
        self.model, self.tokenizer = load(model_path)
        mx.eval(self.model.parameters())
        self.log(f"Model loaded in {time.time() - t0:.1f}s")

        # Gemma uses sliding-window attention → RotatingKVCache, which mlx-lm
        # can't quantize yet ("RotatingKVCache Quantization NYI"). Auto-disable
        # so inference doesn't 500 on the first call. (Credit: asdmoment, PR #7.)
        if self.kv_bits and "gemma" in model_path.lower():
            self.log("Gemma detected: disabling KV cache quantization (RotatingKVCache NYI)")
            self.kv_bits = 0
        self.log(f"KV cache quantization: {self.kv_bits}-bit" if self.kv_bits
                 else "KV cache: full precision")
        return self.tokenizer

    # ─── Memory ──────────────────────────────────────────────────────────────

    def mem_snapshot(self):
        """Watch mx_peak, not rss. `ps rss` cannot see Metal buffers — it
        reported a flat 16.7GB straight through a prefill MLX measured at 34GB.
        """
        try:
            rss = psutil_rss_gb()
        except Exception:
            rss = 0.0
        return {
            "rss_gb": round(rss, 1),
            "mx_active_gb": round(mx.get_active_memory() / GB, 1),
            "mx_cache_gb": round(mx.get_cache_memory() / GB, 1),
            "mx_peak_gb": round(mx.get_peak_memory() / GB, 1),
        }

    def release_transients(self, tag=""):
        """Hand MLX's recycle pool back after every generation, and log the peak.

        The log line is half the point. When a memory death happens there is no
        traceback and usually no crash report, because the process is killed
        from outside — so a per-request peak, written down while the server was
        still alive, is the only thing that distinguishes "that prompt was too
        big" from any other silent exit.
        """
        before = mx.get_cache_memory() / GB
        peak = mx.get_peak_memory() / GB
        mx.clear_cache()
        self.log(f"  Memory{tag}: peak {peak:.1f}GB this request, "
                 f"released {before:.1f}GB of buffer cache, rss now {psutil_rss_gb():.1f}GB")
        mx.reset_peak_memory()

    # ─── Prompt cache ────────────────────────────────────────────────────────

    def prepare_prompt(self, token_ids):
        # Only a standard KVCache can be trimmed and extended. Gemma 4's
        # RotatingKVCache is a circular buffer that breaks on trim+extend.
        cache_is_safe = (self._prompt_cache is not None
                         and not isinstance(self._prompt_cache[0], RotatingKVCache))

        cache_hit_len = 0
        if cache_is_safe and self._cached_token_prefix is not None:
            for i in range(min(len(token_ids), len(self._cached_token_prefix))):
                if token_ids[i] == self._cached_token_prefix[i]:
                    cache_hit_len = i + 1
                else:
                    break

        # Always leave at least 1 token to prefill — stream_generate raises
        # ValueError on an empty prompt (new prompt == cached prefix).
        if cache_hit_len >= len(token_ids):
            cache_hit_len = len(token_ids) - 1

        # .offset is the live token count; .step is a fixed 256-token allocation
        # increment. Reading .step made trim_amount negative for any prefix
        # longer than 256 tokens (issue #46). Caches without .offset can't be
        # trimmed safely at all.
        cache_offset = getattr(self._prompt_cache[0], "offset", None) if self._prompt_cache else None
        if cache_hit_len > 0 and cache_offset is None:
            self.log("  Cache has no offset (untrimmable type) — full prefill")
            cache_hit_len = 0

        if cache_hit_len > 0:
            trim_amount = cache_offset - cache_hit_len
            if trim_amount > 0:
                for c in self._prompt_cache:
                    c.trim(trim_amount)
            prompt_for_gen = token_ids[cache_hit_len:]
            self.log(f"  Cache hit: {cache_hit_len} reused, {len(prompt_for_gen)} new tokens "
                     f"to prefill (saved {cache_hit_len} tokens)")
        else:
            if self._prompt_cache is not None and isinstance(self._prompt_cache[0], RotatingKVCache):
                self.log("  RotatingKVCache: fresh cache each request (no trim support)")
            else:
                self.log(f"  Cache miss: full prefill of {len(token_ids)} tokens")
            self._prompt_cache = None
            prompt_for_gen = token_ids

        if self._prompt_cache is None:
            self._prompt_cache = make_prompt_cache(self.model)
            self.log(f"  Created new prompt cache ({len(self._prompt_cache)} layers)")

        # MLX updates the cache in place during generation.
        self._cached_token_prefix = token_ids
        return prompt_for_gen, cache_hit_len

    def reset_cache(self):
        self._prompt_cache = None
        self._cached_token_prefix = None

    # ─── Generation ──────────────────────────────────────────────────────────

    def stream(self, prompt_tokens, max_tokens, temperature):
        gen_kwargs = {
            "prefill_step_size": self.prefill_size,
            "prompt_cache": self._prompt_cache,
            "sampler": make_sampler(temp=temperature if temperature > 0 else 0.0),
        }
        if self.kv_bits:
            gen_kwargs["kv_bits"] = self.kv_bits
            gen_kwargs["kv_group_size"] = 64
            gen_kwargs["quantized_kv_start"] = self.kv_quant_start

        for response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt_tokens,
            max_tokens=max_tokens,
            **gen_kwargs,
        ):
            yield Chunk(response.text, response.generation_tokens, response.finish_reason)
