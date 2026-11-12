import torch
import cpufast_tap_epilogue_runtime as runtime
from cpufast_fft import FFTConv1d
from cpufast_tap import TapConv1d


def test_fused_tap_retains_gemm_parameters_and_unsupported_paths(monkeypatch):
    root = torch.nn.Module()
    root.tap = TapConv1d(FFTConv1d(torch.nn.Conv1d(3, 4, 7, padding=3).eval())).eval()
    x = torch.randn(1, 3, 27)
    residual = torch.randn(1, 4, 27)
    with torch.no_grad():
        expected = root.tap(x, activate=True, residual=residual)
    params = [(p, p._version) for p in root.parameters()]
    calls = []
    def finish(y, bias, activate, residual):
        calls.append(True)
        y = y + bias[:, None]
        y = torch.nn.functional.leaky_relu(y, .1) if activate else y
        return y.unsqueeze(0) + residual
    monkeypatch.setattr(runtime, 'finish', finish)
    rows = runtime.enable_tap_epilogues(root)
    with torch.no_grad():
        assert torch.equal(root.tap(x, activate=True, residual=residual), expected)
    assert calls == [True] and rows[0]['fused_calls'] == 1
    assert all(a is b and version == b._version for (a, version), b in zip(params, root.parameters()))
    # Autograd and noncontiguous temporal residual retain the original implementation.
    root.tap(x, activate=True, residual=residual)
    with torch.no_grad():
        root.tap(x, activate=True, residual=torch.randn(1, 4, 54)[:, :, ::2])
    assert calls == [True] and rows[0]['fallback_calls'] == 2
