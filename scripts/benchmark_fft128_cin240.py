"""Opt-in six-module FFT-size substitution with unchanged approved model parameters."""
import hashlib
import json
import os
import sys
from pathlib import Path

from cpufast_fft128_cin240_runtime import enable_fft128_cin240


def main():
    import cpufast_engine
    import benchmark_cpufast
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    root = Path(__file__).resolve().parents[1]
    paths = [Path(os.environ.get(k, root / v)) for k, v in (
        ('CPUFAST_AR_LIBRARY', 'libcpufast_ar_torch.so'),
        ('CPUFAST_FAST_LIBRARY', 'libcpufast_fast.so'))]
    paths += [p.with_suffix('.so.json') for p in paths]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    original = cpufast_engine.optimize
    audit = []
    calls = {}
    handles = []
    def optimize(pipeline, **options):
        runtime = original(pipeline, **options)
        audit.extend(enable_fft128_cin240(pipeline.vits_model.dec))
        names = {row['module'] for row in audit}
        for name, layer in pipeline.vits_model.dec.named_modules():
            if name in names:
                calls[name] = 0
                def count(module, inputs, result, name=name):
                    if module.n_fft != 128:
                        raise RuntimeError('FFT policy changed during inference')
                    calls[name] += 1
                handles.append(layer.register_forward_hook(count))
        return runtime
    cpufast_engine.optimize = optimize
    try:
        benchmark_cpufast.main()
    finally:
        cpufast_engine.optimize = original
        for handle in handles:
            handle.remove()
    after = hashes()
    if len(audit) != 6 or not calls or not all(calls.values()) or before != after:
        raise RuntimeError('FFT128 module invocation or baseline DSO integrity failure')
    report = json.loads(output.read_text())
    report['fft128_cin240'] = {
        'scope': 'Only six original V400 stage0 Cin240 conv2 FFT lengths256to128; normal FP32; unchanged model tensors and AR runtime',
        'modules': audit, 'calls': calls,
        'helper_sha256': hashlib.sha256((root/'cpufast_fft128_cin240_runtime.py').read_bytes()).hexdigest(),
        'baseline_library_files_before': before, 'baseline_library_files_after': after,
        'baseline_library_files_unchanged': before == after,
        'numerical_acceptance_basis': 'Prior same-model KO/JA/EN max1int16LSB, SNR>92dB, identical semantic; owner additionally approved KO output with reply ko_D ㄱㅊ네; no JA/EN or C24 model approval',
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
