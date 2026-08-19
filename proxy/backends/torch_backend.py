"""Torch/transformers backend — NVIDIA CUDA, AMD ROCm, or plain CPU.

The Apple-Silicon-only half of this project is MLX, not the server: the
Anthropic protocol, the tool-call recovery and the prompt shaping in
server.py are pure Python. This backend gives all of that a home on a Linux
box with a GPU (or no GPU at all, slowly).

Env vars:
    LLM_MODEL / MLX_MODEL   HuggingFace id or local path
    LLM_DEVICE              cuda | cpu | mps | auto (default auto)
    LLM_DTYPE               bfloat16 | float16 | float32 | auto (default auto)
    LLM_LOAD_IN_4BIT=1      load 4-bit via bitsandbytes (needs CUDA)
    LLM_LOAD_IN_8BIT=1      load 8-bit via bitsandbytes (needs CUDA)
    LLM_PROMPT_CACHE=0      disable KV reuse across requests
"""

import os
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.cache_utils import DynamicCache
from transformers.generation.stopping_criteria import StoppingCriteria

from .base import Backend, Chunk

GB = 1024 ** 3

# Turn markers that are part of the chat template rather than the answer.
# MLX's detokenizer never surfaces these; transformers' streamer does when
# skip_special_tokens is off, and we keep it off because tool-call markup
# (<tool_call>, <|tool_call>) is an *added* token for Qwen/Hermes and would be
# deleted along with the real special tokens.
TURN_MARKERS = (
    "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>",
    "<end_of_turn>", "<turn|>", "<|turn>", "</s>",
)


def _resolve_device(requested):
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(requested, device):
    if requested and requested != "auto":
        return getattr(torch, requested)
    if device == "cpu":
        return torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class _TokenCounter(StoppingCriteria):
    """Counts generated tokens, and stops generation on request.

    transformers has no streaming token count, and `len(text)` is not it — one
    token is 1-4 characters of anything. The abort flag is what lets a caller
    that walks away mid-stream (an SSE client hanging up) end the generation
    instead of leaving a thread writing into the KV cache the next request is
    about to reuse.
    """

    def __init__(self):
        self.n = 0
        self.abort = threading.Event()

    def __call__(self, input_ids, scores, **kwargs):
        self.n += 1
        return self.abort.is_set()


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, log=None):
        super().__init__(log=log)
        self.device = _resolve_device(os.environ.get("LLM_DEVICE", "auto"))
        self.dtype = None
        self.device_label = self.device
        self.cache_enabled = os.environ.get("LLM_PROMPT_CACHE", "1") not in ("0", "false")
        self._cache = None
        self._cached_token_prefix = None
        self._cache_ready = False
        self._gen_error = None

    # ─── Load ────────────────────────────────────────────────────────────────

    def load(self, model_path):
        self.dtype = _resolve_dtype(os.environ.get("LLM_DTYPE", "auto"), self.device)
        kwargs = {"dtype": self.dtype}

        quant = None
        if os.environ.get("LLM_LOAD_IN_4BIT") == "1":
            quant = "4bit"
        elif os.environ.get("LLM_LOAD_IN_8BIT") == "1":
            quant = "8bit"
        if quant:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant == "4bit",
                load_in_8bit=quant == "8bit",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
            )
            # bitsandbytes places the weights itself; moving them afterwards
            # with .to() is an error.
            kwargs["device_map"] = "auto"

        self.log(f"Loading model: {model_path}")
        self.log(f"  Device: {self.device} · dtype: {self.dtype}"
                 + (f" · {quant} (bitsandbytes)" if quant else ""))
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if "device_map" not in kwargs:
            self.model.to(self.device)
        self.model.eval()
        self.log(f"Model loaded in {time.time() - t0:.1f}s")

        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            self.device_label = f"{props.name} ({props.total_memory / GB:.0f}GB)"
            self.log(f"  GPU: {self.device_label}")
        self.log("Prompt cache: enabled (KV reuse)" if self.cache_enabled
                 else "Prompt cache: disabled")
        return self.tokenizer

    # ─── Memory ──────────────────────────────────────────────────────────────

    def mem_snapshot(self):
        if self.device != "cuda":
            return {"device": self.device}
        return {
            "device": self.device,
            "cuda_allocated_gb": round(torch.cuda.memory_allocated() / GB, 1),
            "cuda_reserved_gb": round(torch.cuda.memory_reserved() / GB, 1),
            "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / GB, 1),
        }

    def release_transients(self, tag=""):
        if self.device != "cuda":
            return
        peak = torch.cuda.max_memory_allocated() / GB
        reserved = torch.cuda.memory_reserved() / GB
        torch.cuda.empty_cache()
        self.log(f"  Memory{tag}: peak {peak:.1f}GB this request, "
                 f"released {reserved - torch.cuda.memory_reserved() / GB:.1f}GB of reserved cache")
        torch.cuda.reset_peak_memory_stats()

    # ─── Prompt cache ────────────────────────────────────────────────────────

    def prepare_prompt(self, token_ids):
        """transformers wants the full prompt even on a cache hit — it slices
        off whatever the cache already covers itself — so unlike MLX we always
        return every token and only crop the cache."""
        self._cache_ready = False
        if not self.cache_enabled:
            self._cache = None
            return token_ids, 0

        cache_hit_len = 0
        if self._cache is not None and self._cached_token_prefix is not None:
            for i in range(min(len(token_ids), len(self._cached_token_prefix))):
                if token_ids[i] == self._cached_token_prefix[i]:
                    cache_hit_len = i + 1
                else:
                    break
            # Leave at least one token for the forward pass to run on.
            cache_hit_len = min(cache_hit_len, len(token_ids) - 1)

        if cache_hit_len > 0:
            try:
                self._cache.crop(cache_hit_len)
            except Exception as e:
                self.log(f"  Cache crop failed ({e}) — full prefill")
                cache_hit_len = 0
                self._cache = None

        if cache_hit_len > 0:
            self.log(f"  Cache hit: {cache_hit_len} reused, {len(token_ids) - cache_hit_len} "
                     f"new tokens to prefill (saved {cache_hit_len} tokens)")
        else:
            self.log(f"  Cache miss: full prefill of {len(token_ids)} tokens")
            self._cache = DynamicCache()

        self._cached_token_prefix = list(token_ids)
        self._cache_ready = True
        return token_ids, cache_hit_len

    def reset_cache(self):
        self._cache = None
        self._cached_token_prefix = None
        self._cache_ready = False

    # ─── Generation ──────────────────────────────────────────────────────────

    def _run_generate(self, kwargs):
        try:
            with torch.inference_mode():
                self.model.generate(**kwargs)
        except Exception as e:  # surfaced on the consuming thread below
            self._gen_error = e
        finally:
            kwargs["streamer"].end()

    def stream(self, prompt_tokens, max_tokens, temperature):
        self._gen_error = None
        input_ids = torch.tensor([list(prompt_tokens)], device=self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=False)
        counter = _TokenCounter()

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": max_tokens,
            "streamer": streamer,
            "stopping_criteria": [counter],
            "pad_token_id": pad_id,
            "use_cache": True,
        }
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs["do_sample"] = False
        # Only reuse the cache for the generation prepare_prompt just set up.
        # The tool-call retry path re-prompts with a different transcript, and
        # feeding that a cache built from the first prompt corrupts the state.
        if self._cache_ready and self._cache is not None:
            kwargs["past_key_values"] = self._cache
        else:
            self._cache = None
            self._cached_token_prefix = None
        self._cache_ready = False

        thread = threading.Thread(target=self._run_generate, args=(kwargs,), daemon=True)
        thread.start()

        try:
            for text in streamer:
                for marker in TURN_MARKERS:
                    text = text.replace(marker, "")
                if text:
                    yield Chunk(text, counter.n)
        finally:
            # Whoever was consuming this may have walked away (a hung-up SSE
            # client closes the socket mid-generation). generate() would keep
            # running on its thread and keep appending to the KV cache, and the
            # next request would then crop a cache that is still growing —
            # which surfaces as a bewildering off-by-one tensor size error deep
            # inside attention. Stop it, drain it, and drop the cache.
            if thread.is_alive():
                counter.abort.set()
                for _ in streamer:
                    pass
                thread.join()
                self.reset_cache()

        if self._gen_error is not None:
            err, self._gen_error = self._gen_error, None
            self.reset_cache()
            raise err

        yield Chunk("", counter.n, "length" if counter.n >= max_tokens else "stop")
