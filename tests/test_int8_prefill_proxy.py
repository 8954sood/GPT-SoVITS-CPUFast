"""Approximate FFNs must stay scoped to prompt and restore unrelated engine state."""

from types import SimpleNamespace

import pytest
import torch

import cpufast_prefill_int8_probe as probe


def test_prompt_scope_preserves_attention_cache_contract_and_decode(monkeypatch):
    class Linear:
        def __init__(self, weight, bias, *, relu=False):
            self.weight, self.bias, self.relu = weight, bias, relu

        def __call__(self, x):
            y = torch.addmm(self.bias, x, self.weight)
            return y.relu() if self.relu else y

    monkeypatch.setattr(probe, "Int8PrefillLinear", Linear)

    class MLP:
        def __init__(self):
            self.w1_t, self.w2_t = torch.randn(4, 8), torch.randn(8, 4)
            self.b1, self.b2 = torch.randn(8), torch.randn(4)

        def forward(self, x):
            return torch.addmm(
                self.b2,
                torch.addmm(self.b1, x.reshape(-1, 4), self.w1_t).relu(),
                self.w2_t,
            ).view_as(x)

    class Transformer:
        def __init__(self):
            self.blocks = [SimpleNamespace(block=SimpleNamespace(mlp=MLP()))]
            self.cache = object()
            self.fail = False

        def process_prompt(self, x):
            result = self.blocks[0].block.mlp.forward(x)
            if self.fail:
                raise RuntimeError("prompt error")
            return result, self.cache

        def decode_next_token(self, x):
            return self.blocks[0].block.mlp.forward(x)

    original = Transformer()
    mlp = original.blocks[0].block.mlp
    saved = mlp.forward
    proxy = probe.build_prefill_ffn_proxy(original)
    x = torch.randn(1, 3, 4)
    with torch.no_grad():
        expected, cache = original.process_prompt(x)
        actual, observed = proxy.process_prompt(x)
        assert torch.equal(actual, expected) and observed is cache
        assert mlp.forward == saved and not proxy.busy
        assert proxy.audit[0]["calls"] == 1
        assert torch.equal(
            proxy.decode_next_token(x[:, :1]), original.decode_next_token(x[:, :1])
        )
        assert proxy.audit[0]["calls"] == 1
        original.fail = True
        with pytest.raises(RuntimeError, match="prompt error"):
            proxy.process_prompt(x)
        assert mlp.forward == saved and not proxy.busy
