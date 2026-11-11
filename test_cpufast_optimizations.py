import pytest
import torch

from cpufast_optimizations import (
    CPUFastRuntime,
    PackedConv1d,
    PhaseUpsample1d,
    install_convolutions,
    parameter_digest,
)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dilation,groups", [(1, 1), (3, 2)])
def test_fp32_layout_preserves_parameters_and_convolution(transposed, dilation, groups):
    torch.manual_seed(7)
    constructor = torch.nn.ConvTranspose1d if transposed else torch.nn.Conv1d
    layer = constructor(
        8, 4, 5, stride=2, padding=2, dilation=dilation, groups=groups
    ).eval()
    x = torch.randn(2, 8, 29)
    original = parameter_digest(layer)
    packed = PackedConv1d(layer).eval()
    assert packed.weight is layer.weight and packed.bias is layer.bias
    with torch.no_grad():
        expected, actual = layer(x), packed(x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert parameter_digest(packed) == original == parameter_digest(layer)
    assert list(layer.state_dict()) == list(packed.state_dict())
    # A subsequent legitimate parameter update must invalidate the inference cache.
    old_pack = packed._packed
    with torch.no_grad():
        layer.weight.add_(0.25)
        torch.testing.assert_close(packed(x), layer(x), rtol=2e-5, atol=2e-6)
    assert packed._packed is not old_pack


def test_training_falls_back_and_retains_gradients():
    original = torch.nn.Conv1d(3, 5, 3, padding=1)
    layer = PackedConv1d(original)
    x = torch.randn(1, 3, 11, requires_grad=True)
    layer(x).square().sum().backward()
    assert x.grad is not None and original.weight.grad is not None
    assert layer._packed is None


def test_recursive_install_keeps_state_dict_and_is_idempotent():
    model = torch.nn.Sequential(
        torch.nn.Conv1d(3, 8, 1), torch.nn.Sequential(torch.nn.Conv1d(8, 2, 3))
    ).eval()
    before = parameter_digest(model)
    assert install_convolutions(model) == 2
    assert install_convolutions(model) == 0
    assert parameter_digest(model) == before


def test_stage_threads_restore_after_generator_close():
    from types import SimpleNamespace

    class Decoder:
        def infer_panel_batch_infer(self):
            assert torch.get_num_threads() == 1
            return next(self.infer_panel_naive())

        def infer_panel_naive_batched(self):
            return self.infer_panel_batch_infer()

        def infer_panel_naive(self):
            assert torch.get_num_threads() == 1
            yield 7

    decoder = Decoder()

    class Pipeline:
        t2s_model = SimpleNamespace(model=decoder)

        def run(self, inputs):
            assert torch.get_num_threads() == 2
            value = decoder.infer_panel_batch_infer()
            assert torch.get_num_threads() == 2
            yield value
            raise RuntimeError("test cancellation")

    runtime = CPUFastRuntime(Pipeline(), ar_threads=1, acoustic_threads=2)
    before = torch.get_num_threads()
    generator = runtime.run({})
    assert next(generator) == 7
    generator.close()
    assert torch.get_num_threads() == before
    with pytest.raises(RuntimeError, match="test cancellation"):
        list(runtime.run({}))
    assert torch.get_num_threads() == before
    with pytest.raises(ValueError, match="already"):
        CPUFastRuntime(runtime.pipeline)


@pytest.mark.parametrize(
    "stride,kernel,padding", [(10, 20, 5), (8, 16, 4), (2, 8, 3), (2, 2, 0)]
)
def test_phase_upsampling_matches_v2proplus_geometries(stride, kernel, padding):
    torch.manual_seed(11)
    original = torch.nn.ConvTranspose1d(
        8, 4, kernel, stride=stride, padding=padding
    ).eval()
    before = parameter_digest(original)
    phase = PhaseUpsample1d(original).eval()
    with torch.no_grad():
        for length in (1, 7, 31):
            x = torch.randn(1, 8, length)
            torch.testing.assert_close(phase(x), original(x), rtol=2e-5, atol=2e-6)
    assert phase.weight is original.weight
    assert parameter_digest(phase) == parameter_digest(original) == before
