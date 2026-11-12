import torch

from experiments.student.decoder_training_scope import configure_trainable


def test_last_stage_training_leaves_prefix_exact_after_optimizer_update():
    model = torch.nn.Module()
    model.stem = torch.nn.Linear(4, 4)
    model.resblocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    model.stage_offsets = [0, 1]
    model.conv_post = torch.nn.Linear(4, 1)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    names = configure_trainable(model, "last_stage")
    assert names and not model.stem.weight.requires_grad and not model.resblocks[0].weight.requires_grad
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
    value = model.stem(torch.ones(1, 4))
    for block in model.resblocks:
        value = block(value)
    model.conv_post(value).square().sum().backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name not in names:
            assert torch.equal(parameter, before[name]) and parameter.grad is None
    assert any(not torch.equal(dict(model.named_parameters())[name], before[name]) for name in names)
    configure_trainable(model, "all")
    assert all(p.requires_grad for p in model.parameters())
