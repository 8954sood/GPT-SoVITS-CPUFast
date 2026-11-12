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
    with torch.no_grad(), torch.jit.optimized_execution(False):
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


def test_compiled_transformer_preserves_real_geometry_and_restores_jit_policy(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "GPT_SoVITS"))
    from AR.models.t2s_model import T2SBlock, T2SMLP, T2STransformer
    from cpufast_script_ar import ExactScriptAR

    generator = torch.Generator().manual_seed(22)
    def rand(*shape):
        return torch.randn(*shape,generator=generator) * .03
    blocks = []
    for _ in range(2):
        mlp = T2SMLP(rand(2048,512),rand(2048),rand(512,2048),rand(512))
        blocks.append(T2SBlock(16,512,mlp,rand(1536,512),rand(1536),rand(512,512),rand(512),
                      torch.ones(512),torch.zeros(512),1e-5,torch.ones(512),torch.zeros(512),1e-5))
    plain = T2STransformer(2,blocks)
    scripted = ExactScriptAR(plain)
    assert len(scripted.blocks) == 2
    for a,b in zip(plain.blocks,scripted.blocks):
        assert a.qkv_w_t.data_ptr() == b.qkv_w_t.data_ptr()
    x = rand(1,209,512)
    mask = torch.zeros(1,16,209,209,dtype=torch.bool)
    with torch.no_grad(), torch.jit.optimized_execution(True):
        a,ak,av,length = plain.process_prompt(x,mask,4,None,True)
        b,bk,bv,other_length = scripted.process_prompt(x,mask,4,None,True)
        assert torch._C._get_graph_executor_optimize()
        assert length == other_length == 209
        assert torch.equal(a.view(torch.int32),b.view(torch.int32))
        for _ in range(3):
            for old,new in zip(ak+av,bk+bv):
                assert torch.equal(old[:,:length].view(torch.int32),new[:,:length].view(torch.int32))
            token = rand(1,1,512)
            a,ak,av,length_a = plain.decode_next_token(token,ak,av,length,None,True)
            b,bk,bv,length_b = scripted.decode_next_token(token,bk,bv,length,None,True)
            assert torch.equal(a.view(torch.int32),b.view(torch.int32))
            assert length_a == length_b == length+1
            assert torch._C._get_graph_executor_optimize()
            length = length_a
