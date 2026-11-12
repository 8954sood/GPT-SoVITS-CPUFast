"""C48 -> 36 -> 48 pair execution, unchanged outer channels and residual."""

import platform
import unittest

import torch
import torch.nn.functional as F

from cpufast_fft import FFTConv1d
from cpufast_tap import TapConv1d
from cpufast_winograd import WinogradConv1d


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class RectangularBottleneckRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_cpp_ops  # noqa: F401

    def test_actual_tap_pair_preserves_outer_residual_and_immutable_parameters(self):
        torch.manual_seed(36048)
        with torch.no_grad():
            # First real candidate uses K11 d3/d5; K7 and d1 cover the supported
            # adjacent pair geometries without changing outer-stage channels.
            for kernel, dilation in ((11, 3), (11, 5), (7, 1)):
                c1 = torch.nn.Conv1d(
                    48,
                    36,
                    kernel,
                    padding=dilation * (kernel - 1) // 2,
                    dilation=dilation,
                ).eval()
                c2 = torch.nn.Conv1d(36, 48, kernel, padding=(kernel - 1) // 2).eval()
                f1 = FFTConv1d(c1, backend="packed")
                f2 = FFTConv1d(c2, backend="packed")
                first, second = TapConv1d(f1), TapConv1d(f2)
                before = [(p, p.clone()) for c in (c1, c2) for p in c.parameters()]
                value = torch.randn(1, 48, 75520)
                expected = c2(F.leaky_relu(c1(F.leaky_relu(value, 0.1)), 0.1)) + value
                for threads in (1, 2):
                    torch.set_num_threads(threads)
                    actual = second(
                        first(F.leaky_relu(value, 0.1), activate=True), residual=value
                    )
                    self.assertEqual(actual.shape, (1, 48, 75520))
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                key = first._cache_key
                self.assertTrue(all(torch.equal(p, b) for p, b in before))
                c1.weight[0, 0, 0].add_(0.01)
                updated = first(F.leaky_relu(value, 0.1), activate=True)
                self.assertNotEqual(first._cache_key, key)
                torch.testing.assert_close(
                    updated,
                    F.leaky_relu(c1(F.leaky_relu(value, 0.1)), 0.1),
                    rtol=2e-4,
                    atol=2e-4,
                )
                c1.weight[0, 0, 0].sub_(0.01)
                # Identity is the contract; arithmetic roundtrip is not asserted
                # after the deliberate cache-mutation check.
                current = [p for c in (first, second) for p in c.parameters()]
                self.assertTrue(all(a is b for a, (b, _) in zip(current, before)))
                for parameter, original in before[1:]:
                    self.assertTrue(torch.equal(parameter, original))

    def test_rectangular_fft_and_winograd_cache_and_fused_finishing(self):
        torch.manual_seed(48036)
        with torch.no_grad():
            for inputs, outputs in ((48, 36), (36, 48)):
                for kernel, dilation in (
                    (11, 3),
                    (11, 5),
                    (7, 1),
                    (3, 1),
                    (3, 3),
                    (3, 5),
                ):
                    conv = torch.nn.Conv1d(
                        inputs,
                        outputs,
                        kernel,
                        padding=dilation * (kernel - 1) // 2,
                        dilation=dilation,
                    ).eval()
                    layer = (
                        WinogradConv1d(conv, native=True)
                        if kernel == 3
                        else FFTConv1d(conv, backend="packed")
                    )
                    layer.parallel_transforms = True
                    if isinstance(layer, FFTConv1d):
                        layer.output_layout = layer.stream_products = (
                            layer.fused_finish
                        ) = True
                    before = [p.clone() for p in conv.parameters()]
                    value = torch.randn(2, inputs, 137)
                    residual = torch.randn(2, outputs * 2, 137)[:, ::2, :]
                    expected = F.leaky_relu(conv(value), 0.1) + residual
                    for threads in (1, 2):
                        torch.set_num_threads(threads)
                        output = layer(value, activate=True, residual=residual)
                        torch.testing.assert_close(
                            output, expected, rtol=2e-4, atol=2e-4
                        )
                    self.assertIs(layer.weight, conv.weight)
                    self.assertIs(layer.bias, conv.bias)
                    self.assertTrue(
                        all(
                            torch.equal(p, b) for p, b in zip(conv.parameters(), before)
                        )
                    )


if __name__ == "__main__":
    unittest.main()
