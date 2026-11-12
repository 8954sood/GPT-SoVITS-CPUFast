"""Opt-in Tap epilogue fusion with unchanged approved model parameters and native DSOs."""
import hashlib
import json
import os
import sys
from pathlib import Path

from cpufast_tap_epilogue_runtime import enable_tap_epilogues


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
    def optimize(pipeline, **options):
        runtime = original(pipeline, **options)
        audit.extend(enable_tap_epilogues(pipeline.vits_model.dec))
        return runtime
    cpufast_engine.optimize = optimize
    try:
        benchmark_cpufast.main()
    finally:
        cpufast_engine.optimize = original
    after = hashes()
    if len(audit) != 24 or not all(r['fused_calls'] for r in audit) or any(r['fallback_calls'] for r in audit) or before != after:
        raise RuntimeError('Tap fused invocation or baseline DSO integrity failure')
    report = json.loads(output.read_text())
    if not all(report.get(k) for k in ('checkpoint_files_unchanged', 'parameters_unchanged', 'ar_parameters_unchanged', 'student_files_unchanged')):
        raise RuntimeError('Tap candidate changed model files or parameters')
    if not all(row.get('runtime_local_pcm_match') and row.get('runtime_local_generated_tokens_match') for row in report['warmups'] + report['runs']):
        raise RuntimeError('Tap candidate did not reproduce its own approved PCM/semantic')
    if report['fp16_acoustic_modules'] or report['experimental_options']['fast_experimental']:
        raise RuntimeError('Tap candidate contaminated FP32 acoustics')
    report['tap_finish'] = {
        'scope': 'Only Tap bias/activation/residual epilogues fused with existing fft_finish/effective1; same FP32 arithmetic order and GEMMs; canonical FFT256 policy and AR runtime unchanged',
        'modules': audit, 'calls': sum(r['fused_calls'] for r in audit),
        'helper_sha256': hashlib.sha256((root/'cpufast_tap_epilogue_runtime.py').read_bytes()).hexdigest(),
        'baseline_library_files_before': before, 'baseline_library_files_after': after,
        'baseline_library_files_unchanged': before == after,
        'numerical_acceptance_basis': 'Prior actual N1 primitive72conditions bitexact including signedzero/NaN/Inf/subnormals; full model must remain exact to approved AR17+V400 PCM',
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
