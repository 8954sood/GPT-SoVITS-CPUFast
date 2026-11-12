"""Actual ARM64 adapter check; does not deploy an HTTP service or change models."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 's2', 'hubert', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Fresh output required')
    import numpy as np
    import torch
    from cpufast_server_engine import Engine, APPROVED_MANIFEST_SHA256
    from cpufast_student_assets import sha256, snapshot
    from streaming_audio import pcm16
    before = snapshot(args.manifest)
    input_before = {'s2': sha256(args.s2), 'reference': sha256(args.reference)}
    engine = Engine(SimpleNamespace(manifest=args.manifest, manifest_sha256=APPROVED_MANIFEST_SHA256,
        s2=args.s2, hubert=args.hubert, ar_threads=2, threads=2))
    manifest = engine.manifest
    case = next(row for row in manifest['listening_report']['cases'] if row['language']=='ko')
    ref = engine.prepare_reference(args.reference, manifest['prompt_text'], manifest['prompt_language'])
    prepared_hashes = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in ref.items()}
    cache = args.output.with_suffix('.prepared.pt')
    torch.save(ref, cache)
    def forbidden(*args, **kwargs):
        raise AssertionError('Prepared request attempted reference loading or preprocessing')
    engine.pipeline.set_ref_audio = forbidden
    engine.pipeline._get_ref_spec = forbidden
    engine.pipeline._set_prompt_semantic = forbidden
    engine.pipeline.sv_model.compute_embedding3 = forbidden
    engine.pipeline.vits_model.build_decode_condition = forbidden
    original_frontend = engine.pipeline.text_preprocessor.segment_and_extract_feature_for_text
    def target_only(text, *args, **kwargs):
        if text == manifest['prompt_text']:
            forbidden()
        return original_frontend(text, *args, **kwargs)
    engine.pipeline.text_preprocessor.segment_and_extract_feature_for_text = target_only
    import av
    av.open = forbidden
    reports = []
    for index in range(5):
        if index == 4:
            ref = torch.load(cache, weights_only=True, map_location='cpu')
        audio, report = engine.synthesize_prepared(case['text'], ref, lang='ko', seed=42)
        assert report['native_pcm_sha256'] == case['pcm_sha256']
        assert report['semantic'] == case['semantic'] and report['approved_request_pcm_match']
        assert audio.dtype == np.int16 and audio.ndim == 1 and len(audio) == 179840
        assert engine.pipeline.prompt_cache['prompt_semantic'] is None
        assert engine.pipeline.prompt_cache['runtime']['decode_ge'] is None
        assert prepared_hashes == {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in ref.items()}
        reports.append(dict(index=index, role='warmup' if index<3 else 'same_process' if index==3 else 'disk_cache_reload',
                            wire_native_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                            wire_pcm16_sha256=hashlib.sha256(pcm16(audio)).hexdigest(), **report))
        print('ADAPTER_VALIDATION', index, report['native_pcm_sha256'], report['approved_request_pcm_match'], flush=True)
    assert len({r['wire_native_sha256'] for r in reports}) == 1
    assert len({r['wire_pcm16_sha256'] for r in reports}) == 1
    native_after = {name: sha256(Path(__file__).resolve().parents[1]/name) for name in engine.native_files_sha256}
    assert native_after == engine.native_files_sha256
    assert before == snapshot(args.manifest)
    assert input_before == {'s2': sha256(args.s2), 'reference': sha256(args.reference)}
    cgroup = {n: (Path('/sys/fs/cgroup')/n).read_text().strip() for n in ('cpu.max','memory.max','memory.swap.max')}
    assert cgroup == {'cpu.max':'200000 100000','memory.max':'8000000000','memory.swap.max':'0'}
    result = dict(complete=True, scope='Actual N1 approved adapter native32k preservation and persistent tensor-reference cache; validation only, not a new paired RTF estimate',
        cgroup=cgroup, engine_ready=engine.ready, manifest_sha256=APPROVED_MANIFEST_SHA256,
        original_inputs_unchanged=True, student_inputs_unchanged=True, native_files_unchanged=True,
        native_files_sha256=native_after, fingerprint=engine.fingerprint, model_input_sha256=engine.input_sha256,
        runtime_source_sha256=engine.source_sha256, tensor_reference_keys=sorted(ref),
        prepared_tensors_unchanged=True, reference_preprocessing_forbidden_after_prepare=True,
        pipeline_prompt_cache_released_after_every_request=True, reports=reports)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
