"""Temporary actual-client sentence-stream probe; no deployment or audio files."""
import argparse
import hashlib
import json
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
        'tap_finish_probe':args.tap_finish,'command':command,'trials':[],'audio_saved':False,'public_host_port_opened':False}
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
                row=measure(client,request)
                row.update(index=i,role='warmup' if i<args.warmups else 'sample')
                assert row['audio_seconds']==5.08 and row['pcm_bytes']==325120
                assert row['response_headers']['x-audio-sample-rate']=='32000'
                expected_hash=expected_hash or row['pcm48k_sha256']
                assert row['pcm48k_sha256']==expected_hash
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
        for report in timings:
            assert [s['native_pcm_sha256'] for s in report['segment_timings']]==[
                '48dca061126afdacd1b1f4e96deb62e2ff86cce96e54b0895f3312b99b57a897',
                '719ce751b7b17cd1522a01235578e88739dfc7e79d09fd9628862f9c0b28389a']
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
