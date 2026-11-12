"""Five fixed fresh AB/BA pairs; FFT-size rounding is distinguished from exact PCM."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from cpufast_student_assets import load_manifest, sha256, snapshot

MANIFEST = 'e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb'
PCM = {'baseline': '2d7691a829c0f793796be37c9a1287a88497ea41dcd1328729333c6b0ec0e83e',
       'fft128': 'c0a60fd5c7fd67d8f03170bc224e9e127b6384d142d3100557d90756a454b45f'}


def validate(data, guard, case, label, *, profile=False):
    required = ('checkpoint_files_unchanged', 'parameters_unchanged', 'ar_parameters_unchanged', 'student_files_unchanged')
    if not all(data.get(k) for k in required) or guard['exit_code'] != 0:
        raise RuntimeError('Model/resource integrity failure')
    if any(data['cgroup'].get(k) != v for k, v in {'cpu.max': '200000 100000', 'memory.max': '8000000000', 'memory.swap.max': '0'}.items()):
        raise RuntimeError('Actual N1 resource limit changed')
    if not data['approximate_ar_int8_only'] or data['experimental_options']['fast_experimental'] or data['fp16_acoustic_modules']:
        raise RuntimeError('Approved AR/FP32 acoustic runtime contaminated')
    blocks = sorted({int(r['block']) for r in data['ar_coefficient_audit']['matrices']})
    if blocks != list(range(17)):
        raise RuntimeError('Actual loaded AR depth changed')
    for row in data['runs'] + data['warmups']:
        semantic = [{'tokens': r['index'], 'sha256': r['generated_sha256']} for r in row['semantic']]
        if semantic != case['semantic'] or row['audio_seconds'] != 5.62 or row['pcm_sha256'] != PCM[label]:
            raise RuntimeError('Fixed-model semantic/duration or per-runtime expected PCM changed')
        if label == 'fft128':
            comparison = row['comparison']
            limit = 0 if profile else 1
            if not comparison['same_shape'] or not comparison['same_rate'] or comparison['max_pcm_delta'] > limit:
                raise RuntimeError('FFT rounding exceeded prior one-LSB equivalence bound')
            if not profile and comparison['snr_db'] is not None and comparison['snr_db'] < 92:
                raise RuntimeError('FFT rounding SNR fell below prior numeric bound')
    if label == 'fft128':
        audit = data['fft128_cin240']
        if len(audit['modules']) != 6 or len(audit['calls']) != 6 or set(audit['calls'].values()) != {4} or not audit['baseline_library_files_unchanged']:
            # Separate profiler performs an additional un-timed inference.
            if not (profile and len(audit['modules']) == 6 and len(audit['calls']) == 6 and set(audit['calls'].values()) == {5} and audit['baseline_library_files_unchanged']):
                raise RuntimeError('Six-module execution/DSO audit failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, MANIFEST)
    case = next(r for r in manifest['listening_report']['cases'] if r['language'] == 'ko')
    before = snapshot(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output/'student_summary.json'
    if summary_path.exists():
        parser.error('Fresh output directory required')
    summary = {'scope': 'Actual N1 unchanged approved AR17+originalV400: canonical FFT256 baseline vs six Cin240 FFT128 modules only',
               'manifest_sha256': MANIFEST, 'protocol': 'Exactly5freshpairs AB/BA, each3warm1sample; one separate excluded candidate profile; all raw/outliers retained',
               'numeric_policy': 'Same semantic/duration and fixed separate PCM hashes; candidate versus baseline max1int16LSB and SNR>=92dB. Owner approved KO output: ko_D ㄱㅊ네; approval limited to this Korean same-reference/seed output, not JA/EN or C24.',
               'quality_approval': {'owner_reply': 'ko_D ㄱㅊ네', 'scope': 'FFT128-on-Cin240 AR17+originalV400 Korean same reference/seed only; no JA/EN or C24 model approval'},
               'order': [], 'complete': False}
    records = {'baseline': [], 'fft128': []}
    def save():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    def run(name, label, compare=None, profile=False):
        output = args.output/(name+'.json')
        command = [sys.executable, '/repo/scripts/benchmark_fft128_cin240.py' if label == 'fft128' else '/repo/benchmark_cpufast.py',
                   '--engine', '--head-major', '--cook-phase', '--fft-layout', '--packed-fft', '--fused-finish', '--tap-small',
                   '--ar-threads', '2', '--threads', '2', '--s1', str(args.manifest.parent/'ar.pt'),
                   '--s2', '/repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth', '--hubert', '/project/models/chinese-hubert-base',
                   '--reference', '/reference.wav', '--prompt-text', manifest['prompt_text'], '--prompt-lang', manifest['prompt_language'],
                   '--lang', 'ko', '--text', case['text'], '--warmups', '3', '--samples', '1', '--ar-prefetch', '256',
                   '--student-manifest', str(args.manifest), '--student-manifest-sha256', MANIFEST,
                   '--ar-int8-only', '--fft-stream', '--parallel-transforms', '--inference-mode', '--output', str(output)]
        if compare is not None:
            command += ['--compare-pcm', str(compare)]
        if profile:
            command += ['--profile-output', str(args.output/'ko_student_profile_ops.json'), '--profile-no-shapes']
        guardfile = output.with_name(output.stem+'_guard.json')
        subprocess.run([sys.executable, '/repo/resource_guard.py', '--report', str(guardfile), '--', *command], check=True)
        data = json.loads(output.read_text())
        guard = json.loads(guardfile.read_text())
        validate(data, guard, case, label, profile=profile)
        return data['runs'][0], guard
    for index in range(5):
        for label in (('baseline', 'fft128') if index % 2 == 0 else ('fft128', 'baseline')):
            name = f'ko_paired_{index}_{label}'
            comparison = args.output/'ko_paired_0_baseline.wav' if label == 'fft128' else None
            row, guard = run(name, label, comparison)
            records[label].append(row)
            summary['order'].append({'round': index, 'runtime': label, 'name': name, 'rtf': row['rtf'], 'peak_sampled_tree_rss_bytes': guard['peak_sampled_tree_rss_bytes']})
            save()
    summary['aggregate'] = {}
    for label, rows in records.items():
        rtfs = [r['rtf'] for r in rows]
        summary['aggregate'][label] = {'rtfs': rtfs, 'median_rtf': statistics.median(rtfs), 'min_rtf': min(rtfs), 'max_rtf': max(rtfs),
            'all_measured_rtf_below_one': max(rtfs) < 1,
            'median_wall_seconds': statistics.median(r['seconds'] for r in rows),
            'median_cpu_seconds': statistics.median(r['cpu_seconds'] for r in rows),
            'median_stages': {k: statistics.median(r['stages'][k] for r in rows) for k in rows[0]['stages']}}
    save()
    run('ko_fft128_profile', 'fft128', args.output/'ko_paired_4_fft128.wav', profile=True)
    summary['student_assets_unchanged'] = before == snapshot(args.manifest)
    if not summary['student_assets_unchanged']:
        raise RuntimeError('Approved private assets changed')
    summary['complete'] = True
    save()


if __name__ == '__main__':
    main()
