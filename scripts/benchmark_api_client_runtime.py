"""Temporary actual-client sentence-stream probe; no deployment or audio files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
from measure_streaming_api import measure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','s2','hubert','reference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--runs',type=int,default=5)
    p.add_argument('--tap-finish',action='store_true')
    p.add_argument('--port',type=int,default=18736)
    args=p.parse_args();assert not args.output.exists()
    root=Path(__file__).resolve().parents[1]
    from cpufast_student_assets import snapshot,sha256
    before_assets=snapshot(args.manifest)
    original_inputs={'s2':sha256(args.s2),'reference':sha256(args.reference)}
    native_before={n:sha256(root/n) for n in ('libcpufast_ar_torch.so','libcpufast_ar_torch.so.json','libcpufast_fast.so','libcpufast_fast.so.json')}
    environment={k:os.environ.get(k) for k in ('OMP_WAIT_POLICY','GOMP_SPINCOUNT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_PRELOAD')}
    assert environment=={'OMP_WAIT_POLICY':'ACTIVE','GOMP_SPINCOUNT':'100000','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','LD_PRELOAD':'/build/native/libgsv_mimalloc.so'}
    manifest=json.loads(args.manifest.read_text())
    case=next(c for c in manifest['listening_report']['cases'] if c['language']=='ko')
    command=[sys.executable,str(root/'server.py'),'--host','127.0.0.1','--port',str(args.port),
        '--data-dir',str(args.output.with_suffix('.private-styles')),'--manifest',str(args.manifest),
        '--manifest-sha256','696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865',
        '--s2',str(args.s2),'--hubert',str(args.hubert),'--ar-threads','2','--threads','2']
    if args.tap_finish:
        bootstrap="import sys; import cpufast_server_engine as e; from cpufast_tap_epilogue_runtime import enable_tap_epilogues; old=e.Engine.__init__; exec('def init(self,settings):\\n old(self,settings)\\n self._probe_tap_audit=enable_tap_epilogues(self.pipeline.vits_model.dec)'); e.Engine.__init__=init; import runpy; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        command=[sys.executable,'-c',bootstrap]+command[1:]
    result={'scope':'Actual loopback HTTP client request start through complete PCM; sentence boundaries/seeds42,43/full reference/model unchanged',
        'environment':environment,'model_files_before':before_assets,'original_inputs_before':original_inputs,'native_files_before':native_before,'tap_finish_probe':args.tap_finish,'command':command,'trials':[],'audio_saved':False,'public_host_port_opened':False}
    logpath=args.output.with_suffix('.server.log')
    log=logpath.open('wb');process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',timeout=180) as client:
            for _ in range(720):
                if process.poll() is not None:raise RuntimeError('Temporary server startup failed')
                try:
                    response=client.get('/health')
                    if response.status_code==200 and response.json()['ready']:break
                except httpx.RequestError:pass
                time.sleep(.25)
            else:raise RuntimeError('Startup timeout')
            result['startup_health']=response.json()
            with args.reference.open('rb') as audio:
                response=client.post('/v1/styles',files={'audio':('reference.wav',audio,'audio/wav')},
                    data={'name':'api-runtime-probe','prompt_text':manifest['prompt_text'],'prompt_lang':manifest['prompt_language']})
            assert response.status_code==201,response.text
            style_id=response.json()['style_id']
            request=dict(style_id=style_id,text=case['text'],lang='ko',seed=42,latency_mode='fast')
            expected_hash=None
            for i in range(args.warmups+args.runs):
                h=client.get('/health').json();assert h['ready'] and not h['busy']
                rss_before=next(line.split()[1] for line in Path(f'/proc/{process.pid}/status').read_text().splitlines() if line.startswith('VmRSS:'))
                row=measure(client,request)
                row['server_rss_bytes_before']=int(rss_before)*1024
                row['server_rss_bytes_after']=int(next(line.split()[1] for line in Path(f'/proc/{process.pid}/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
                row.update(index=i,role='warmup' if i<args.warmups else 'sample')
                assert row['audio_seconds']==5.08 and row['pcm_bytes']==325120
                assert row['response_headers']['x-audio-sample-rate']=='32000'
                expected_hash='a3aa7f71c04a7a4b28457225575946c9b29ee7e23bef493ed9de262ecbb4be9d'
                assert row['pcm_sha256']==expected_hash
                result['trials'].append(row)
                print('ACTUAL_CLIENT_RUNTIME',row['role'],i,row['client_response_rtf'],row['client_first_pcm_seconds'],row['required_extra_initial_buffer_seconds'],flush=True)
            response=client.delete('/v1/styles/'+style_id);assert response.status_code==204
            result['delete_status']=204;result['final_health']=client.get('/health').json()
            result['complete']=True
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(60)
            except subprocess.TimeoutExpired:process.kill();process.wait();result['forced_kill']=True
        log.close()
        result['server_stopped']=process.poll() is not None
        result['server_exit_code']=process.returncode
        result['server_log_sha256']=hashlib.sha256(logpath.read_bytes()).hexdigest()
        result['server_graceful_shutdown']='Application shutdown complete.' in logpath.read_text()
        timings=[]
        for line in logpath.read_text().splitlines():
            i=line.find('{"request_id"')
            if i>=0:timings.append(json.loads(line[i:]))
        result['server_segment_reports']=timings
        result['model_files_after']=snapshot(args.manifest)
        result['original_inputs_after']={'s2':sha256(args.s2),'reference':sha256(args.reference)}
        result['native_files_after']={n:sha256(root/n) for n in native_before}
        result['all_assets_and_native_files_unchanged']=(before_assets==result['model_files_after'] and original_inputs==result['original_inputs_after'] and native_before==result['native_files_after'])
        assert result['all_assets_and_native_files_unchanged']
        result['cgroup']={n:(Path('/sys/fs/cgroup')/n).read_text().strip() for n in ('cpu.max','memory.max','memory.swap.max')}
        assert result['cgroup']=={'cpu.max':'200000 100000','memory.max':'8000000000','memory.swap.max':'0'}
        for report in timings:
            assert len(report['segment_timings'])==2
            assert [s['semantic_tokens'] for s in report['segment_timings']]==[44,68]
            assert all(s.get('ar_dispatch')=='compiled_methods_eager_fp32_order' for s in report['segment_timings'])
            assert [s['native_pcm_sha256'] for s in report['segment_timings']]==[
                '48dca061126afdacd1b1f4e96deb62e2ff86cce96e54b0895f3312b99b57a897',
                '719ce751b7b17cd1522a01235578e88739dfc7e79d09fd9628862f9c0b28389a']
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
