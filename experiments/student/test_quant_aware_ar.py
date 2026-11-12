import copy

import torch
import torch.nn.functional as F

from experiments.student.quant_aware_ar import QuantAwareARStudent, decode_linear
from experiments.student.test_distillation import small_teacher


def test_prompt_remains_fp32_and_decode_matches_defined_integer_grid():
    torch.manual_seed(4)
    layer = torch.nn.Linear(16, 8)
    value = torch.randn(2, 6, 16, requires_grad=True)
    value.data[:, -1] = 0
    before = copy.deepcopy(layer.state_dict())
    result = decode_linear(layer, value, 3)
    torch.testing.assert_close(
        result[:, :3], F.linear(value[:, :3], layer.weight.half().float(), layer.bias)
    )
    weight = layer.weight.detach().half().float()
    sw = (weight.abs().amax(-1) / 127).clamp_min(1e-12)
    for batch in range(2):
        for token in range(3, 6):
            row = value[batch, token].detach()
            sx = row.abs().max() / 127 if row.abs().max() > 0 else row.new_tensor(1.0)
            xq = (row * sx.reciprocal()).round().clamp(-127, 127).to(torch.int64)
            wq = (weight / sw[:, None]).round().clamp(-127, 127).to(torch.int64)
            expected = (wq @ xq).float() * sx * sw + layer.bias.detach()
            torch.testing.assert_close(
                result[batch, token], expected, atol=1e-6, rtol=1e-6
            )
    result.square().sum().backward()
    assert torch.isfinite(value.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())
    assert all(torch.equal(v, before[k]) for k, v in layer.state_dict().items())


def test_qat_features_finite_gradients_export_identity_and_teacher_unchanged():
    source = small_teacher()
    original = copy.deepcopy(source["weight"])
    model = QuantAwareARStudent(source, [0, 2])
    value = torch.randn(1, 6, 4)
    mask = torch.ones(6, 6, dtype=torch.bool).triu(1)
    logits, hidden = model.forward_with_features(value, mask, 2, [0, 2])
    (
        logits.square().mean() + sum(v.square().mean() for v in hidden.values())
    ).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    saved = model.checkpoint(source)
    restored = QuantAwareARStudent(saved, [0, 1])
    assert torch.equal(logits, restored(value, mask, 2))
    assert all(torch.equal(v, original[k]) for k, v in source["weight"].items())
