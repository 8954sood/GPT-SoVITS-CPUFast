import pytest
import torch

from cpufast_kv_cache import head_major_prefix_cache


@pytest.mark.parametrize("batch,length,heads,dim,steps", [(1,209,16,32,1500),(1,226,16,32,1500),(2,7,4,8,3)])
def test_prompt_prefix_and_future_attention_match_full_cache_transpose(batch,length,heads,dim,steps):
    generator = torch.Generator().manual_seed(42)
    prefix = torch.randn(batch,length,heads*dim,generator=generator)
    before = prefix.clone()
    # Legacy path copies the whole padded capacity, although only prefix is valid.
    old = torch.full((batch,length+steps,heads*dim),float('nan'))
    old[:,:length] = prefix
    old = old.reshape(batch,length+steps,heads,dim).transpose(1,2).contiguous()
    new = head_major_prefix_cache(prefix,heads,steps)
    assert new.shape == old.shape and new.is_contiguous()
    assert torch.equal(prefix,before)
    assert torch.equal(new[:,:,:length].view(torch.int32),old[:,:,:length].view(torch.int32))
    query = torch.randn(batch,heads,1,dim,generator=generator)
    for offset in range(min(steps,3)):
        token = torch.randn(batch,heads,1,dim,generator=generator)
        old[:,:,length+offset:length+offset+1] = token
        new[:,:,length+offset:length+offset+1] = token
        active = length+offset+1
        expected = torch.nn.functional.scaled_dot_product_attention(query,old[:,:,:active],old[:,:,:active])
        actual = torch.nn.functional.scaled_dot_product_attention(query,new[:,:,:active],new[:,:,:active])
        assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))


def test_copy_preserves_ieee_bits_and_does_not_alias_prompt():
    bits = torch.tensor([0,-2147483648,0x7f800000,0x7fc01234],dtype=torch.int32)
    prefix = bits.view(torch.float32).reshape(1,2,2)
    new = head_major_prefix_cache(prefix,2,1)
    restored = new[:,:,:2].transpose(1,2).contiguous().reshape_as(prefix)
    assert torch.equal(restored.view(torch.int32),prefix.view(torch.int32))
    new.fill_(0)
    assert torch.equal(prefix.view(torch.int32).reshape(-1),bits)
