"""Execute compiled AR class methods without rewriting eager FP32 operations."""
import torch


class ExactScriptAR:
    dispatch_profile = "compiled_methods_eager_fp32_order"

    def __init__(self, transformer):
        self._compiled = torch.jit.script(transformer)
        self._prompt = self._compiled.process_prompt
        self._decode = self._compiled.decode_next_token
        if hasattr(self._prompt, "__code__") or hasattr(self._decode, "__code__"):
            raise RuntimeError("AR methods must execute the compiled class instance")

    @property
    def blocks(self):
        return self._compiled.blocks

    def process_prompt(self, *args, **kwargs):
        # Default JIT graph rewrites change prompt GEMM rounding and semantic
        # sampling. Keep compilation, and preserve the original operation order.
        with torch.jit.optimized_execution(False):
            return self._prompt(*args, **kwargs)

    def decode_next_token(self, *args, **kwargs):
        with torch.jit.optimized_execution(False):
            return self._decode(*args, **kwargs)
