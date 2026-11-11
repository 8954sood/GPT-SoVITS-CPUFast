import torch, json
from pathlib import Path
import cpufast_cpp_ops
from cpufast_fft import FFTConv1d
from cpufast_winograd import WinogradConv1d

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(3)
rows = []
with torch.no_grad():
    for batch, length, channels, kernel, dilation, bias in [
        (1, 73, 24, 7, 1, True),
        (2, 147, 24, 11, 3, False),
        (1, 541, 48, 11, 5, True),
        (2, 92, 24, 3, 5, True),
    ]:
        conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=(kernel - 1) * dilation // 2,
            dilation=dilation,
            bias=bias,
        ).eval()
        x = torch.randn(batch, channels, length)
        res = torch.randn(batch, channels, length + 4)[..., 2:-2]
        for backend in ["fused", "packed", "packed_layout"]:
            m = FFTConv1d(
                conv, backend="packed" if backend == "packed_layout" else backend
            )
            m.output_layout = backend == "packed_layout"
            m.fused_finish = False
            y = m(x)
            torch.testing.assert_close(y, conv(x), atol=1e-4, rtol=1e-3)
            for active, rr in [(False, None), (True, None), (False, res), (True, res)]:
                expected = torch.nn.functional.leaky_relu(y, 0.1) if active else y
                if rr is not None:
                    expected = expected + rr
                m.fused_finish = True
                out = m(x, activate=active, residual=rr)
                m.fused_finish = False
                assert torch.equal(out, expected), (
                    batch,
                    length,
                    backend,
                    (out - expected).abs().max(),
                )
                rows.append(
                    {
                        "kind": "fft",
                        "batch": batch,
                        "length": length,
                        "kernel": kernel,
                        "dilation": dilation,
                        "backend": backend,
                        "activation": active,
                        "residual": rr is not None,
                        "exact": True,
                    }
                )
    for dilation in [1, 3, 5]:
        for length in [17, 96, 173]:
            conv = torch.nn.Conv1d(
                12, 12, 3, padding=dilation, dilation=dilation
            ).eval()
            x = torch.randn(2, 12, length)
            rr = torch.randn(2, 12, length + 4)[..., 2:-2]
            m = WinogradConv1d(conv, native=True)
            y = m(x)
            for active, res in [(True, None), (False, rr), (True, rr)]:
                expected = torch.nn.functional.leaky_relu(y, 0.1) if active else y
                if res is not None:
                    expected = expected + res
                out = m(x, activate=active, residual=res)
                assert torch.equal(out, expected), (
                    dilation,
                    length,
                    (out - expected).abs().max(),
                )
                rows.append(
                    {
                        "kind": "winograd",
                        "dilation": dilation,
                        "length": length,
                        "activation": active,
                        "residual": res is not None,
                        "exact": True,
                    }
                )
Path("/repo/.experiments/fusion_tests.json").write_text(json.dumps(rows, indent=2))
print("Passed", len(rows), "exact epilogue cases")
