"""Check that real AR class instances compile without changing prompt/decode math."""
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("padding_kind", ["none", "boolean", "float"])
def test_compiled_real_block_matches_python_prompt_and_appended_decode(monkeypatch, padding_kind):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "GPT_SoVITS"))
    from AR.models.t2s_model import T2SBlock, T2SMLP

    generator = torch.Generator().manual_seed(22)
    def rand(*shape):
        return torch.randn(*shape, generator=generator)
    mlp = T2SMLP(rand(16,8),rand(16),rand(8,16),rand(8))
    plain = T2SBlock(2,8,mlp,rand(24,8),rand(24),rand(8,8),rand(8),
                    torch.ones(8),torch.zeros(8),1e-5,torch.ones(8),torch.zeros(8),1e-5)
    compiled = torch.jit.script(plain)
    assert hasattr(plain.process_prompt, "__code__")
    assert not hasattr(compiled.process_prompt, "__code__")
    # Compilation shares coefficients, rather than copying/casting/repacking them.
    for name in ("qkv_w_t", "out_w_t", "norm_w1", "norm_b1", "norm_w2", "norm_b2"):
        assert getattr(plain,name).data_ptr() == getattr(compiled,name).data_ptr()
    x = rand(1,7,8)
    mask = torch.zeros(1,2,7,7,dtype=torch.bool)
    padding = None
    if padding_kind != "none":
        padding = torch.zeros(1,7,1,dtype=torch.bool)
        padding[:,-2:] = True
        mask[:,:,:,-2:] = True
        if padding_kind == "float":
            padding = (~padding).float()
    with torch.no_grad():
        a, ak, av, length = plain.process_prompt(x,mask,4,padding,True)
        b, bk, bv, other_length = compiled.process_prompt(x,mask,4,padding,True)
        assert length == other_length == 7
        assert torch.equal(a.view(torch.int32),b.view(torch.int32))
        for _ in range(3):
            assert torch.equal(ak[:,:length].view(torch.int32),bk[:,:length].view(torch.int32))
            assert torch.equal(av[:,:length].view(torch.int32),bv[:,:length].view(torch.int32))
            token = rand(1,1,8)
            a, ak, av = plain.decode_next_token(token,ak,av,length,None,True)
            b, bk, bv = compiled.decode_next_token(token,bk,bv,length,None,True)
            assert torch.equal(a.view(torch.int32),b.view(torch.int32))
            length += 1
