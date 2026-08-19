"""What a backend has to provide for proxy/server.py to run on it."""


class Chunk:
    """One step of a generation stream.

    Mirrors the fields of mlx_lm's GenerationResponse that the server reads,
    so both backends can be consumed by the same loop.
    """

    __slots__ = ("text", "generation_tokens", "finish_reason")

    def __init__(self, text, generation_tokens, finish_reason=None):
        self.text = text
        self.generation_tokens = generation_tokens
        self.finish_reason = finish_reason


class Backend:
    name = "base"
    # Human-readable description of the compute the model will run on
    # ("Apple Silicon GPU (Metal)", "NVIDIA CUDA", "CPU"), for logs and /health.
    device_label = "unknown"

    def __init__(self, log=None):
        self.log = log or (lambda msg: None)
        self.model = None
        self.tokenizer = None

    def load(self, model_path):
        """Load weights + tokenizer. Returns the tokenizer.

        The tokenizer must expose the HuggingFace surface the server uses:
        apply_chat_template, encode, chat_template, eos_token_id.
        """
        raise NotImplementedError

    def mem_snapshot(self):
        """Backend-specific memory numbers for /health, as a flat dict of GB."""
        return {}

    def release_transients(self, tag=""):
        """Hand large per-request allocations back after a generation."""

    # ─── Prompt cache ────────────────────────────────────────────────────────
    # A backend that can reuse KV state across requests trims its cache to the
    # shared prefix and returns only the tokens still to prefill. One that
    # can't returns the whole prompt and a hit length of 0; the server logs
    # and behaves the same either way.

    def prepare_prompt(self, token_ids):
        """Returns (tokens_to_prefill, cache_hit_len)."""
        return token_ids, 0

    def reset_cache(self):
        """Drop any KV state, so the next request prefills from scratch."""

    def stream(self, prompt_tokens, max_tokens, temperature):
        """Yield Chunk objects until the model stops or hits max_tokens."""
        raise NotImplementedError
