"""Opt-in output-cache padding preserves original non-12-aligned coefficients."""

import platform
import unittest

import torch

from cpufast_fft import FFTConv1d


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class FFTChannelPaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_cpp_ops  # noqa: F401

    def test_actual_width640_channels_match_explicit_zero_padded_reference_exactly(
        self,
    ):
        torch.manual_seed(818)
        with torch.no_grad():
            for channels in (320, 160, 80, 40, 20):
                outputs = (channels + 11) // 12 * 12
                conv = torch.nn.Conv1d(
                    channels, channels, 7, padding=9, dilation=3
                ).eval()
                reference = torch.nn.Conv1d(
                    channels, outputs, 7, padding=9, dilation=3
                ).eval()
                reference.weight.zero_()
                reference.bias.zero_()
                reference.weight[:channels].copy_(conv.weight)
                reference.bias[:channels].copy_(conv.bias)
                before = conv.weight.clone(), conv.bias.clone()
                padded = FFTConv1d(conv, backend="packed", pad_channels=True)
                explicit = FFTConv1d(reference, backend="packed")
                self.assertIs(padded.weight, conv.weight)
                self.assertIs(padded.bias, conv.bias)
                for batch, layout, stream in (
                    (1, False, False),
                    (1, True, False),
                    (2, True, True),
                ):
                    padded.output_layout = explicit.output_layout = layout
                    padded.stream_products = explicit.stream_products = stream
                    padded.fused_finish = explicit.fused_finish = layout
                    padded.parallel_transforms = explicit.parallel_transforms = stream
                    x = torch.randn(batch, channels, 41)
                    residual = torch.randn_like(x) if stream else None
                    reference_residual = (
                        torch.nn.functional.pad(residual, (0, 0, 0, outputs - channels))
                        if residual is not None
                        else None
                    )
                    for threads in (1, 2):
                        torch.set_num_threads(threads)
                        expected = explicit(
                            x, activate=layout, residual=reference_residual
                        )[:, :channels]
                        actual = padded(x, activate=layout, residual=residual)
                        self.assertEqual(actual.shape, (batch, channels, 41))
                        self.assertTrue(
                            torch.equal(actual, expected),
                            (channels, batch, layout, stream, threads),
                        )
                self.assertEqual(padded._packed.shape[1] * 12, outputs)
                self.assertTrue(torch.equal(conv.weight, before[0]))
                self.assertTrue(torch.equal(conv.bias, before[1]))
                padded.pad_channels = False
                with self.assertRaisesRegex(ValueError, "divisible by 12"):
                    padded(x)

    def test_aligned_default_is_unchanged_and_invalid_backend_rejected(self):
        conv = torch.nn.Conv1d(24, 24, 11, padding=5).eval()
        with self.assertRaisesRegex(ValueError, "packed FP32"):
            FFTConv1d(conv, pad_channels=True)
        plain = FFTConv1d(conv, backend="packed")
        padded = FFTConv1d(conv, backend="packed", pad_channels=True)
        plain.output_layout = padded.output_layout = True
        plain.stream_products = padded.stream_products = True
        plain.fused_finish = padded.fused_finish = True
        plain.parallel_transforms = padded.parallel_transforms = True
        value = torch.randn(2, 24, 137)
        residual = torch.randn_like(value)
        with torch.no_grad():
            self.assertTrue(
                torch.equal(
                    plain(value, activate=True, residual=residual),
                    padded(value, activate=True, residual=residual),
                )
            )


if __name__ == "__main__":
    unittest.main()
