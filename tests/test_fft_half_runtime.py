"""The approximate candidate must affect only declared large FFT products."""

import platform
import unittest

import torch

from cpufast_fft import FFTConv1d
from cpufast_fft_half_runtime import enable_fft_half_products


def fft(channels=96, backend="packed"):
    module = FFTConv1d(
        torch.nn.Conv1d(channels, channels, 7, padding=3).eval(), backend=backend
    )
    module.output_layout = True
    module.stream_products = True
    module.fused_finish = True
    module.parallel_transforms = True
    return module


class HalfProductScopeTests(unittest.TestCase):
    def test_scope_and_original_parameters_are_preserved(self):
        decoder = torch.nn.ModuleList([fft(), torch.nn.Conv1d(24, 24, 3)])
        before = [(p, p.clone()) for p in decoder.parameters()]
        audit = enable_fft_half_products(decoder)
        self.assertEqual(len(audit), 1)
        self.assertTrue(decoder[0].fast_half)
        self.assertFalse(decoder[0].stream_products)
        self.assertTrue(decoder[0].parallel_transforms)
        self.assertTrue(decoder[0].fused_finish)
        self.assertFalse(hasattr(decoder[1], "fast_half"))
        self.assertEqual(audit[0]["module"], "0")
        for current, (parameter, value) in zip(decoder.parameters(), before):
            self.assertIs(current, parameter)
            self.assertTrue(torch.equal(current, value))

    def test_rejects_bad_scope_before_changing_any_target(self):
        for invalid in (fft(48), fft(96, "torch")):
            first = fft()
            with self.assertRaises(ValueError):
                enable_fft_half_products(torch.nn.ModuleList([first, invalid]))
            self.assertFalse(first.fast_half)
            self.assertTrue(first.stream_products)
        with self.assertRaises(ValueError):
            enable_fft_half_products(torch.nn.Conv1d(3, 3, 1))
        already = fft()
        already.fast_half = True
        with self.assertRaises(ValueError):
            enable_fft_half_products(already)

    @unittest.skipUnless(
        platform.system() == "Linux" and platform.machine() == "aarch64",
        "Linux ARM64 only",
    )
    def test_native_half_product_path_is_finite_fp32_and_immutable(self):
        import cpufast_cpp_ops  # noqa: F401
        import cpufast_fast_ops  # noqa: F401

        torch.manual_seed(765)
        module = fft()
        before = module.weight.clone(), module.bias.clone()
        enable_fft_half_products(module)
        with torch.no_grad():
            value = torch.randn(2, 96, 137)
            residual = torch.randn_like(value)
            for threads in (1, 2):
                torch.set_num_threads(threads)
                output = module(value, activate=True, residual=residual)
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(output.shape, value.shape)
                self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(module._packed.dtype, torch.float32)
            self.assertEqual(module._packed_half.dtype, torch.float16)
            self.assertTrue(torch.equal(module.weight, before[0]))
            self.assertTrue(torch.equal(module.bias, before[1]))


if __name__ == "__main__":
    unittest.main()
