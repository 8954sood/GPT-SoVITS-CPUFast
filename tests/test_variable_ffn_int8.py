"""Variable FFN1536 native decode packing and quantized-reference correctness."""

import platform
import unittest

import torch


@unittest.skipUnless(
    platform.system() == "Linux" and platform.machine() == "aarch64", "Linux ARM64 only"
)
class VariableFFNInt8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cpufast_fast_ops  # noqa: F401

    def test_ffn_512_to_1536_and_back_match_quantized_reference(self):
        from cpufast_fast_ar import pack, packq

        torch.manual_seed(1536512)
        with torch.no_grad():
            for inputs, outputs in ((512, 1536), (1536, 512)):
                weight_t = (
                    (torch.randn(inputs, outputs) * 0.03).half().float().contiguous()
                )
                bias = torch.randn(outputs) * 0.01
                original = weight_t.clone(), bias.clone()
                packed = pack(weight_t)
                restored = packed.transpose(1, 2).reshape(outputs, inputs).T.float()
                self.assertTrue(torch.equal(restored, weight_t))
                quantized, scales = packq(packed)
                self.assertEqual(quantized.shape, (outputs, inputs))
                for zero in (False, True):
                    value = (
                        torch.zeros(1, 1, inputs) if zero else torch.randn(1, 1, inputs)
                    )
                    scale = (
                        value.abs().max() / 127 if not zero else value.new_tensor(1.0)
                    )
                    qx = (value * (1 / scale)).round().clamp(-127, 127).int()
                    expected = (qx @ quantized.int().T).float() * scale * scales + bias
                    for threads in (1, 2):
                        torch.set_num_threads(threads)
                        actual = torch.ops.cpufast_cpp.int8_linear(
                            value, quantized, scales, bias
                        )
                        self.assertEqual(actual.shape, (1, 1, outputs))
                        torch.testing.assert_close(
                            actual, expected, rtol=1e-5, atol=1e-5
                        )
                self.assertTrue(torch.equal(weight_t, original[0]))
                self.assertTrue(torch.equal(bias, original[1]))
                with self.assertRaises(RuntimeError):
                    torch.ops.cpufast_cpp.int8_linear(
                        torch.ones(4, inputs), quantized, scales, bias
                    )


if __name__ == "__main__":
    unittest.main()
