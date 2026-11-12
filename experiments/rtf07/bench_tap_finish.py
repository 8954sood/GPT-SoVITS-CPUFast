"""Actual-shape Tap epilogue probe using the existing baseline FFT-finish operator.

No checkpoint or default runtime changes. All wall/CPU samples include complete
bias/activation/residual epilogues, but exclude preparation of the GEMM result.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import cpufast_cpp_ops  # noqa: F401


def baseline(y, bias, activate, residual):
    if bias is not None:
        y.add_(bias[:, None])
    result = y.unsqueeze(0)
    if activate:
        result = F.leaky_relu(result, 0.1)
    return result if residual is None else result + residual


def candidate(y, bias, activate, residual):
    return torch.ops.cpufast_cpp.fft_finish(y[None, :, None, :], bias, 1, y.shape[-1], activate, residual, True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Fresh output required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    report = {'scope': 'Epilogue-only primitive bit checks and actual-shape synthetic micro; not TTS RTF',
              'machine': platform.machine(), 'torch': str(torch.__version__), 'threads': 2,
              'operator': 'Existing canonical cpufast_cpp.fft_finish, effective1/blocks1',
              'bit_conditions': [], 'cases': []}
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    with torch.no_grad():
        edges = torch.tensor([0., -0., 1., -1., 1e-38, -1e-38, 1e-44, -1e-44,
                              float('inf'), -float('inf'), float('nan'), torch.finfo(torch.float32).max,
                              -torch.finfo(torch.float32).max], dtype=torch.float32)
        for length in (1, 3, 4, 7, 16, 257):
            source = edges.repeat((3*length+len(edges)-1)//len(edges))[:3*length].reshape(3, length)
            for bias in (None, torch.tensor([0., -0., .125])):
                for activate in (False, True):
                    for residual in (None, torch.randn(1, 3, length), torch.randn(1, 3, length*2)[:, :, ::2]):
                        # fft_finish requires contiguous temporal residual stride. Check
                        # supported stride1, including noncontiguous channel/batch strides.
                        if residual is not None and residual.stride(2) != 1:
                            residual = torch.randn(1, 6, length)[:, ::2, :]
                        a = baseline(source.clone(), bias, activate, residual)
                        b = candidate(source.clone(), bias, activate, residual)
                        exact = torch.equal(a.view(torch.int32), b.view(torch.int32))
                        report['bit_conditions'].append({'length': length, 'bias': bias is not None,
                            'activate': activate, 'residual': residual is not None,
                            'residual_strides': list(residual.stride()) if residual is not None else None,
                            'all_bits_equal_including_nan_payload_and_signed_zero': exact})
                        if not exact:
                            save()
                            raise RuntimeError('Tap epilogue IEEE bit pattern gate failed')
        for channels, length in ((24, 170240), (36, 85120), (48, 85120)):
            source = torch.randn(channels, length)
            bias = torch.randn(channels)
            for activate in (True, False):
                residual = None if activate else torch.randn(1, channels, length)
                a = baseline(source.clone(), bias, activate, residual)
                b = candidate(source.clone(), bias, activate, residual)
                exact = torch.equal(a.view(torch.int32), b.view(torch.int32))
                if not exact:
                    raise RuntimeError('Actual-shape bit pattern gate failed')
                samples = {'baseline': [], 'candidate': []}
                cpu = {'baseline': [], 'candidate': []}
                for index in range(8):
                    for name in (('baseline','candidate') if index%2 == 0 else ('candidate','baseline')):
                        y = source.clone()
                        start_cpu, start = time.process_time(), time.perf_counter()
                        result = (baseline if name == 'baseline' else candidate)(y, bias, activate, residual)
                        elapsed, cpu_elapsed = time.perf_counter()-start, time.process_time()-start_cpu
                        if index >= 3:
                            samples[name].append(elapsed)
                            cpu[name].append(cpu_elapsed)
                        del result, y
                report['cases'].append({'channels': channels, 'length': length, 'activation': activate,
                    'residual': residual is not None, 'all_bits_equal': exact, 'warmups_each': 3,
                    'wall_seconds': samples, 'cpu_seconds': cpu,
                    'median_ratio_candidate_over_baseline': statistics.median(samples['candidate'])/statistics.median(samples['baseline'])})
                save()
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
