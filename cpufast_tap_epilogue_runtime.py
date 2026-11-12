"""Opt-in same-order FP32 Tap epilogue fusion, using the existing native DSO."""
import types
import torch
from cpufast_tap import TapConv1d


def finish(y, bias, activate, residual):
    return torch.ops.cpufast_cpp.fft_finish(y[None, :, None, :], bias, 1, y.shape[-1], activate, residual, True)


def enable_tap_epilogues(decoder):
    audit = []
    for name, layer in decoder.named_modules():
        if type(layer) is not TapConv1d:
            continue
        original = layer.forward
        row = {'module': name, 'weight_shape': list(layer.weight.shape), 'fused_calls': 0, 'fallback_calls': 0}
        audit.append(row)
        def forward(self, x, *, activate=False, residual=None, _original=original, _row=row):
            bias = self.bias
            invalid_bias = bias is not None and (bias.device.type != 'cpu' or bias.dtype != torch.float32 or not bias.is_contiguous())
            invalid_residual = residual is not None and (residual.device.type != 'cpu' or residual.dtype != torch.float32
                or residual.ndim != 3 or residual.stride(2) != 1)
            if (self.training or torch.is_grad_enabled() or x.ndim != 3 or x.shape[0] != 1
                or x.device.type != 'cpu' or x.dtype != torch.float32 or self.weight.dtype != torch.float32
                or torch.is_autocast_enabled('cpu') or any(p.is_inference() for p in self.parameters())
                or invalid_bias or invalid_residual):
                _row['fallback_calls'] += 1
                return _original(x, activate=activate, residual=residual)
            key = (id(self.weight), self.weight._version, self.weight.data_ptr())
            if key != self._cache_key:
                self._packed = self.weight.detach().permute(2, 0, 1).contiguous()
                self._cache_key = key
            length = x.shape[-1] + 2*self.padding - self.dilation*(self.weight.shape[-1]-1)
            if residual is not None and list(residual.shape) != [1, self.weight.shape[0], length]:
                _row['fallback_calls'] += 1
                return _original(x, activate=activate, residual=residual)
            padded = torch.nn.functional.pad(x, (self.padding, self.padding)).squeeze(0)
            y = torch.empty((self.weight.shape[0], length), dtype=x.dtype, device=x.device)
            torch.mm(self._packed[0], padded[:, :length], out=y)
            for j in range(1, self.weight.shape[-1]):
                start = j*self.dilation
                torch.addmm(y, self._packed[j], padded[:, start:start+length], out=y)
            _row['fused_calls'] += 1
            return finish(y, bias, activate, residual)
        layer.forward = types.MethodType(forward, layer)
    return audit
