"""Temporary loopback-only real-model HTTP/cache restart check; no deployment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 's2', 'hubert', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--port', type=int, default=18735)
    args = p.parse_args()
    if args.output.exists():
        p.error('Fresh output required')
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.manifest.read_text())
    case = next(r for r in manifest['listening_report']['cases'] if r['language']=='ko')
    cwd = args.output.parent/'http-working-directory'
    cwd.mkdir(exist_ok=True)
    style_root = cwd/'styles'
    if style_root.exists() and list(style_root.iterdir()):
        p.error('Fresh private style directory required')
    command = [sys.executable, str(root/'server.py'), '--host', '127.0.0.1', '--port', str(args.port),
        '--data-dir', 'styles', '--manifest', str(args.manifest.resolve()),
        '--manifest-sha256', '696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865',
        '--s2', str(args.s2.resolve()), '--hubert', str(args.hubert.resolve()), '--ar-threads', '2', '--threads', '2']
    processes = []
    logfiles = []
    def start(number):
        logpath = args.output.parent/f'http-server-{number}.log'
        log = logpath.open('wb')
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        logfiles.append((logpath, log))
        deadline = time.monotonic()+180
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}', timeout=10) as client:
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Temporary HTTP server startup failed; inspect {logpath}')
                try:
                    response=client.get('/health')
                    if response.status_code==200:
                        assert response.json()['ready']
                        return process, response.json()
                except httpx.RequestError:
                    pass
                time.sleep(.25)
        raise RuntimeError('Temporary server startup timed out')
    def stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError('Temporary server needed forced shutdown')
    def idle(client):
        for _ in range(100):
            h=client.get('/health').json()
            if not h['busy']:
                assert h['ready']
                return h
            time.sleep(.05)
        raise RuntimeError('Worker lease did not release')
    def speech(client, style_id, mode):
        response=client.post('/v1/tts/stream', json=dict(style_id=style_id,text=case['text'],lang='ko',seed=42,latency_mode=mode))
        assert response.status_code==200, response.text
        assert response.headers['x-audio-sample-rate']=='48000'
        assert response.headers['x-audio-channels']=='1'
        assert response.headers['x-audio-format']=='s16le'
        assert response.headers['x-latency-mode-effective']=='fixed_approved_model'
        assert response.headers['x-stream-segments']=='2'
        assert len(response.content)>0 and len(response.content)%2==0 and not response.content.startswith(b'RIFF')
        return dict(latency_mode=mode,status=response.status_code,bytes=len(response.content),
            pcm48k_sha256=hashlib.sha256(response.content).hexdigest(),
            headers={k:v for k,v in response.headers.items() if k.startswith('x-')})
    result={'scope':'Temporary loopback-only actual N1 HTTP contract and persistent prepared-reference restart, not a deployment or new paired RTF experiment',
            'server_processes_sequential':True,'bind':'127.0.0.1','docker_host_port_published':False,'streams':[]}
    try:
        process, health1=start(1)
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',timeout=180) as client:
            with args.reference.open('rb') as audio:
                response=client.post('/v1/styles',files={'audio':('reference.wav',audio,'audio/wav')},
                    data=dict(name='temporary-native-qa',prompt_text=manifest['prompt_text'],prompt_lang=manifest['prompt_language']))
            assert response.status_code==201,response.text
            metadata=response.json();style_id=metadata['style_id']
            result['registration_status']=201
            result['style_id']=style_id
            assert client.get('/v1/styles/'+style_id).status_code==200
            assert style_root.is_dir(), 'Engine changed relative data-directory working directory'
            idle(client)
            with ThreadPoolExecutor(max_workers=1) as pool:
                def first_speech():
                    with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',timeout=180) as c:
                        return speech(c,style_id,'fast')
                future=pool.submit(first_speech)
                observed_busy=False
                for _ in range(100):
                    h=client.get('/health').json()
                    if h['busy']:
                        observed_busy=True
                        assert h['ready']
                        busy=client.post('/v1/tts/stream',json=dict(style_id=style_id,text=case['text'],lang='ko'))
                        assert busy.status_code==429 and busy.headers['retry-after']=='1'
                        break
                    if future.done():
                        break
                    time.sleep(.03)
                assert observed_busy, 'Could not observe real inference busy state'
                result['busy_health_observed']=True
                result['busy_second_request_status']=429
                result['streams'].append(future.result())
            result['first_process_health_after']=idle(client)
        stop(process)
        assert process.poll() is not None
        process, health2=start(2)
        assert health2['styles']==1 and health2['preloaded_from_disk']==1 and health2['rebuilt_on_startup']==0
        result['restart_health']=health2
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',timeout=180) as client:
            assert client.get('/v1/styles/'+style_id).status_code==200
            result['streams'].append(speech(client,style_id,'quality'))
            idle(client)
            result['streams'].append(speech(client,style_id,'balanced'))
            idle(client)
            assert len({r['pcm48k_sha256'] for r in result['streams']})==1
            assert len({r['bytes'] for r in result['streams']})==1
            for unsupported in ({'streaming_mode':'tokens'},{'reference_frames':128}):
                response=client.post('/v1/tts/stream',json=dict(style_id=style_id,text=case['text'],lang='ko',**unsupported))
                assert response.status_code==422
            result['unsupported_modes_rejected_422']=True
            response=client.delete('/v1/styles/'+style_id)
            assert response.status_code==204
            assert client.get('/v1/styles/'+style_id).status_code==404
            assert client.get('/v1/styles').json()['styles']==[]
            result['delete_status']=204
            result['final_health']=idle(client)
            assert not (style_root/style_id).exists()
        stop(process)
        result['complete']=True
    finally:
        for process in processes:
            stop(process)
        for _, log in logfiles:
            log.close()
        result['all_temporary_server_processes_stopped']=all(process.poll() is not None for process in processes)
        result['server_exit_codes']=[process.returncode for process in processes]
        result['cgroup']={n:(Path('/sys/fs/cgroup')/n).read_text().strip() for n in ('cpu.max','memory.max','memory.swap.max')}
        assert result['cgroup']=={'cpu.max':'200000 100000','memory.max':'8000000000','memory.swap.max':'0'}
        timing=[]
        for logpath,_ in logfiles:
            for line in logpath.read_text().splitlines():
                position=line.find('{"request_id"')
                if position>=0:
                    timing.append(json.loads(line[position:]))
        result['actual_stream_segment_reports']=timing
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
