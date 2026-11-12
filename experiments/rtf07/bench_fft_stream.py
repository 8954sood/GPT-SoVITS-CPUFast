"""Compare full-buffer and streamed FP32 FFT products on representative shapes."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import cpufast_cpp_ops  # noqa: F401

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--samples', type=int, default=5)
p.add_argument('--warmups', type=int, default=3)
a = p.parse_args()
if a.output.exists():
    p.error('Fresh output required')
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(92)
report = {'note': 'Synthetic FP32 spectral matrices; not speech RTF', 'torch': torch.__version__, 'platform': platform.platform(), 'threads': 2, 'cases': []}
for channels, blocks, fft in [(384, 26, 128), (192, 93, 256), (96, 185, 256)]:
    freq = fft // 2 + 1
    tiles = (blocks + 7) // 8
    x = torch.randn(3 * freq, tiles, channels, 8)
    w = torch.randn(3 * freq, channels // 12, channels, 12)
    def original():
        return torch.ops.cpufast_cpp.fft_unpack_layout(torch.ops.cpufast_cpp.fft_packed_gemm(x, w), 1, blocks)
    def stream():
        return torch.ops.cpufast_cpp.fft_packed_stream(x, w, 1, blocks)
    expected = original()
    assert torch.equal(expected, stream())
    for name, fn in [('original', original), ('stream', stream)]:
        for _ in range(a.warmups):
            fn()
        values = []
        cpu = []
        for _ in range(a.samples):
            start, cpu_start = time.perf_counter(), time.process_time()
            out = fn()
            values.append(time.perf_counter() - start)
            cpu.append(time.process_time() - cpu_start)
            assert torch.equal(out, expected)
        row = {'channels': channels, 'blocks': blocks, 'fft': fft, 'method': name, 'seconds': values, 'cpu_seconds': cpu, 'median_seconds': statistics.median(values), 'exact': True}
        report['cases'].append(row)
        a.output.write_text(json.dumps(report, indent=2) + '\n')
        print(row, flush=True)
