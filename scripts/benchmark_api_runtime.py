"""Source-pinned sentence API runtime exploration, with immutable output gates."""
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import time
from types import SimpleNamespace

EXPECTED_WIRE = '23fb33fc18e04d63d9eab28910d32c3369a65bcd8c1922df2e4f65fede707706'
EXPECTED_NATIVE = ('48dca061126afdacd1b1f4e96deb62e2ff86cce96e54b0895f3312b99b57a897',
                   '719ce751b7b17cd1522a01235578e88739dfc7e79d09fd9628862f9c0b28389a')


def counters():
    values={}
    for name in ('cpu.stat','cpu.pressure'):
        p=Path('/sys/fs/cgroup')/name
        if p.exists():values[name]=p.read_text().strip()
    r=resource.getrusage(resource.RUSAGE_SELF)
    values.update(voluntary_context_switches=r.ru_nvcsw,involuntary_context_switches=r.ru_nivcsw,maxrss_kib=r.ru_maxrss)
    values['host_cpu_jiffies']=Path('/proc/stat').read_text().splitlines()[0]
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 's2', 'hubert', 'reference', 'output'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--threads', type=int, choices=(1, 2), default=2)
    p.add_argument('--ar-threads', type=int, choices=(1, 2), default=2)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--tap-finish', action='store_true')
    p.add_argument('--script-ar',action='store_true')
    p.add_argument('--script-no-opt',action='store_true')
    p.add_argument('--int8-probe',choices=('none','quant-neon','ffn2-rows8','packed4'),default='none')
    p.add_argument('--int8-probe-library',type=Path)
    p.add_argument('--expected-wait-policy')
    p.add_argument('--expected-spincount')
    p.add_argument('--expected-preload')
    p.add_argument('--gc-policy', choices=('default','original','gen0','freeze'), default='default')
    args = p.parse_args()
    assert not args.output.exists()
    for flag,variable in (('expected_wait_policy','OMP_WAIT_POLICY'),('expected_spincount','GOMP_SPINCOUNT'),('expected_preload','LD_PRELOAD')):
        expected=getattr(args,flag)
        if expected is not None and os.environ.get(variable)!=expected:p.error('Actual '+variable+' differs from required policy')
    import torch
    from cpufast_server_engine import Engine, APPROVED_MANIFEST_SHA256
    from cpufast_student_assets import snapshot, sha256
    from streaming_audio import split_text, pcm16
    assets_before = snapshot(args.manifest)
    inputs_before = {'s2':sha256(args.s2), 'reference':sha256(args.reference)}
    engine = Engine(SimpleNamespace(manifest=args.manifest, manifest_sha256=APPROVED_MANIFEST_SHA256,
        s2=args.s2, hubert=args.hubert, threads=args.threads, ar_threads=args.ar_threads))
    script_ar_audit=None
    if args.script_ar:
        original=engine.pipeline.t2s_model.model.t2s_transformer
        before_versions=engine._current_versions()
        begin=time.perf_counter()
        compiled=torch.jit.script(original)
        script_ar_audit={'constructor_seconds':time.perf_counter()-begin,
            'original_type':str(type(original)),'compiled_type':str(type(compiled)),
            'original_decode_has_python_code':hasattr(original.decode_next_token,'__code__'),
            'compiled_decode_has_python_code':hasattr(compiled.decode_next_token,'__code__'),
            'decode_graph':str(compiled.decode_next_token.graph)}
        engine.pipeline.t2s_model.model.t2s_transformer=compiled
        assert before_versions==engine._current_versions()
        assert not script_ar_audit['compiled_decode_has_python_code']
        script_ar_audit['all_authoritative_parameter_pointers_versions_unchanged']=True
    probe_audit=None
    if args.int8_probe!='none':
        if args.int8_probe_library is None:p.error('Explicit probe library required')
        from experiments.rtf07.bench_fft_output_tiles import load_probe
        probe_manifest=load_probe(args.int8_probe_library)
        replacement='-DCPUFAST_REPLACE_INT8_'+{'quant-neon':'QUANT_NEON','ffn2-rows8':'FFN2_ROWS8','packed4':'PACKED4'}[args.int8_probe]+'=1'
        assert replacement in probe_manifest['flags']
        counter=(lambda:torch.ops.cpufast_cpp.int8_packed4_stats()[0]) if args.int8_probe=='packed4' else getattr(torch.ops.cpufast_cpp,'int8_quant_neon_calls' if args.int8_probe=='quant-neon' else 'int8_ffn2_rows8_calls')
        before_calls=counter()
        probe_audit={'kind':args.int8_probe,'manifest':probe_manifest,'library_sha256':sha256(args.int8_probe_library)}
    manifest = engine.manifest
    case = next(c for c in manifest['listening_report']['cases'] if c['language']=='ko')
    segments = split_text(case['text'])
    assert len(segments)==2
    ref = engine.prepare_reference(args.reference, manifest['prompt_text'], manifest['prompt_language'])
    import gc
    if args.gc_policy=='gen0':
        engine.pipeline.empty_cache=lambda:gc.collect(0)
    elif args.gc_policy in ('original','freeze'):
        engine.pipeline.empty_cache=lambda:gc.collect()
    diagnostic={}
    gc_start={}
    def gc_callback(phase,info):
        generation=info['generation']
        if phase=='start':gc_start[generation]=time.perf_counter()
        elif generation in gc_start:
            key='gc_generation_'+str(generation)+'_seconds'
            diagnostic[key]=diagnostic.get(key,0.)+time.perf_counter()-gc_start.pop(generation)
    gc.callbacks.append(gc_callback)
    def timed(owner,name,label):
        original=getattr(owner,name)
        def run(*a,**kw):
            begin=time.perf_counter()
            try:return original(*a,**kw)
            finally:diagnostic[label]=diagnostic.get(label,0.)+time.perf_counter()-begin
        setattr(owner,name,run)
    transformer=engine.pipeline.t2s_model.model.t2s_transformer
    class TransformerProbe:
        def __getattr__(self,name):return getattr(transformer,name)
        def process_prompt(self,*a,**kw):
            begin=time.perf_counter()
            try:
                result=transformer.process_prompt(*a,**kw)
                diagnostic['ar_prefill_calls']=diagnostic.get('ar_prefill_calls',0)+1
                diagnostic['ar_cache_audit']={'input_shape':list(a[0].shape),'requested_decode_capacity':a[2],
                    'keys_shapes':[list(k.shape) for k in result[1]],'values_shapes':[list(v.shape) for v in result[2]],'valid_prefix_length':result[3]}
                return result
            finally:diagnostic['ar_prefill_seconds']=diagnostic.get('ar_prefill_seconds',0.)+time.perf_counter()-begin
        def decode_next_token(self,*a,**kw):
            begin=time.perf_counter()
            try:
                result=transformer.decode_next_token(*a,**kw)
                diagnostic['ar_decode_calls']=diagnostic.get('ar_decode_calls',0)+1
                return result
            finally:diagnostic['ar_decode_seconds']=diagnostic.get('ar_decode_seconds',0.)+time.perf_counter()-begin
    engine.pipeline.t2s_model.model.t2s_transformer=TransformerProbe()
    timed(engine.pipeline,'empty_cache','empty_cache_seconds')
    timed(engine.pipeline.vits_model,'prepare_decode_latent','acoustic_latent_seconds')
    timed(engine.pipeline.vits_model.dec,'forward','student_decoder_seconds')
    timed(engine.pipeline.text_preprocessor,'preprocess','target_frontend_seconds')
    tap_audit=[]
    if args.tap_finish:
        from cpufast_tap_epilogue_runtime import enable_tap_epilogues
        tap_audit=enable_tap_epilogues(engine.pipeline.vits_model.dec)
    def pair(role, index):
        reports=[];wire=[];before=counters();begin=time.perf_counter();cpu_begin=time.process_time()
        for i,text in enumerate(segments):
            diagnostic.clear()
            with torch.jit.optimized_execution(not args.script_no_opt):
                audio,report=engine.synthesize_prepared(text, ref, lang='ko', seed=42+i)
            report['diagnostic_stage_seconds']=dict(diagnostic)
            wire.append(pcm16(audio));reports.append(report)
        elapsed=time.perf_counter()-begin
        wire=b''.join(wire)
        wire_hash=hashlib.sha256(wire).hexdigest()
        rate=engine.sample_rate
        pcm_match=(len(wire)==int(5.08*rate*2)
            and (rate==32000 or wire_hash==EXPECTED_WIRE)
            and tuple(r['native_pcm_sha256'] for r in reports)==EXPECTED_NATIVE)
        assert all(r['model_parameter_versions_unchanged'] for r in reports)
        observation=dict(role=role,index=index,wall_seconds=elapsed,cpu_seconds=time.process_time()-cpu_begin,sample_rate=rate,audio_seconds=len(wire)/2/rate,rtf=elapsed/(len(wire)/2/rate),
            resource_counters_before=before,resource_counters_after=counters(),expected_native_and_wire_pcm_match=pcm_match,wire_pcm_sha256=wire_hash,segments=reports)
        print('API_RUNTIME',role,index,args.ar_threads,args.threads,elapsed,flush=True)
        return observation
    rows=[pair('warmup',i) for i in range(args.warmups)]
    if args.gc_policy=='freeze':
        gc.collect();gc.freeze()
    rows += [pair('sample',i) for i in range(args.runs)]
    profile=None
    if args.profile:
        wrapped=[]
        def scope(owner,name,label):
            original=getattr(owner,name)
            def run(*a,**kw):
                with torch.profiler.record_function(label):return original(*a,**kw)
            setattr(owner,name,run);wrapped.append((owner,name,original))
        scope(engine.pipeline.vits_model,'decode','API_acoustic_decode')
        scope(engine.pipeline.vits_model.dec,'forward','API_student_decoder')
        scope(engine.pipeline.t2s_model.model,'infer_panel_batch_infer','API_AR')
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:
                profile_observation=pair('excluded_profile',0)
            profile={'observation':profile_observation,'operators':[{'name':e.key,'input_shapes':e.input_shapes,
                'count':e.count,'self_cpu_time_total_us':e.self_cpu_time_total,'cpu_time_total_us':e.cpu_time_total}
                for e in prof.key_averages(group_by_input_shape=True)]}
        finally:
            for owner,name,original in wrapped:setattr(owner,name,original)
    if probe_audit is not None:
        probe_audit['actual_calls']=counter()-before_calls
        if args.int8_probe=='packed4':probe_audit['stats_calls_hits_misses_entries_bytes']=list(torch.ops.cpufast_cpp.int8_packed4_stats())
        assert probe_audit['actual_calls']>0
    native_after={n:sha256(Path(__file__).resolve().parents[1]/n) for n in engine.native_files_sha256}
    assert native_after==engine.native_files_sha256
    assert snapshot(args.manifest)==assets_before
    assert inputs_before=={'s2':sha256(args.s2),'reference':sha256(args.reference)}
    cgroup={n:(Path('/sys/fs/cgroup')/n).read_text().strip() for n in ('cpu.max','memory.max','memory.swap.max')}
    assert cgroup=={'cpu.max':'200000 100000','memory.max':'8000000000','memory.swap.max':'0'}
    result=dict(complete=True,scope='Fixed sentence split/seed42,43/full reference/native32k API adapter exploration; HTTP overhead excluded',
        environment={k:os.environ.get(k) for k in ('OMP_WAIT_POLICY','GOMP_SPINCOUNT','OPENBLAS_NUM_THREADS','OPENBLAS_CORETYPE','LD_PRELOAD')},
        settings={'threads':args.threads,'ar_threads':args.ar_threads,'tap_finish':args.tap_finish,'script_no_opt':args.script_no_opt,'gc_policy':args.gc_policy},
        cgroup=cgroup,all_expected_pcm_match=all(r['expected_native_and_wire_pcm_match'] for r in rows),observations=rows,excluded_profile=profile,tap_finish_audit=tap_audit,
        loaded_openmp_libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libgomp' in line}),
        process_threads=len(list(Path('/proc/self/task').iterdir())),torch_parallel_info=torch.__config__.parallel_info(),
        script_ar=script_ar_audit,int8_probe=probe_audit,model_files_unchanged=True,original_inputs_unchanged=True,native_files_unchanged=True,
        native_files_sha256=native_after,model_input_sha256=engine.input_sha256,
        runtime_source_sha256=engine.source_sha256,manifest_sha256=APPROVED_MANIFEST_SHA256)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
