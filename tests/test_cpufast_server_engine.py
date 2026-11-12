"""Reference-cache contract tests; native/model PCM is verified separately on N1."""
import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchaudio.functional  # load resampler before filesystem access is forbidden

from cpufast_server_engine import Engine, REFERENCE_KEYS, _empty_prompt, _encode


class Pipeline:
    def __init__(self):
        self.prompt_cache = _empty_prompt()
        self.t2s_model = SimpleNamespace(model=SimpleNamespace(t2s_transformer=SimpleNamespace(blocks=[None]*17)))
        self.vits_model = SimpleNamespace(dec=SimpleNamespace(resblocks=[None]*11))
    def _invalidate_prompt_runtime_cache(self):
        self.prompt_cache['runtime'] = dict(ref_key=None, refer_audio_spec=None, sv_emb=None,
            decode_ref_key=None, decode_ge=None, decode_ge_text=None)
    def _get_ref_runtime_key(self):
        return ('cpu', 'torch.float32', self.prompt_cache['ref_audio_path'], (), 1, True)
    def set_ref_audio(self, path):
        raise AssertionError('Reference audio must not be read during synthesis')


def engine():
    instance = Engine.__new__(Engine)
    instance._initialized = True
    instance._failed = False
    instance._lock = threading.RLock()
    instance.fingerprint = 'f'*64
    instance.pipeline = Pipeline()
    instance._parameter_versions = []
    instance._current_versions = lambda: []
    instance.native_files_sha256 = {'verified_at_startup': 'source-checked'}
    instance.manifest_sha256 = 'release'
    instance.manifest = dict(reference_sha256='d'*64, prompt_language='ja',
        prompt_text='テスト。', listening_report={'cases': []})
    instance.runtime = SimpleNamespace(run=lambda inputs: synthesis(instance, inputs))
    return instance


def reference():
    result = dict(tokens=torch.tensor([1, 2, 3]), phones=torch.tensor([4, 5]),
        bert_features=torch.zeros(1024, 2), spec=torch.ones(1, 1025, 5),
        reference_audio16=torch.ones(1, 48000), sv_emb=torch.ones(1, 20480),
        decode_ge=torch.ones(1, 512, 1), decode_ge_text=torch.ones(1, 512, 1),
        prompt_text=_encode('テスト。'), prompt_lang=_encode('ja'), norm_text=_encode('テスト。'),
        reference_identity=_encode(json.dumps({'sha256': 'a'*64, 'segment': None})),
        engine_identity=_encode('f'*64), schema_version=torch.tensor([1]))
    assert set(result) == REFERENCE_KEYS
    return result


def synthesis(instance, inputs):
    assert inputs['ref_audio_path'] is None
    assert inputs['fragment_interval'] == .3 and inputs['vits_parallel_infer'] is False
    cache = instance.pipeline.prompt_cache
    runtime = cache['runtime']
    assert runtime['ref_key'] == instance.pipeline._get_ref_runtime_key()
    assert runtime['decode_ref_key'] == runtime['ref_key']
    assert runtime['refer_audio_spec'] and runtime['sv_emb']
    assert runtime['decode_ge'].shape == (1, 512, 1)
    native = np.arange(5120, dtype=np.int16)
    yield 32000, native


def test_disk_reloaded_reference_reproduces_wire_without_reference_file_access(tmp_path, monkeypatch):
    backend = engine()
    prepared = reference()
    cache = tmp_path/'prepared.pt'
    torch.save(prepared, cache)
    reloaded = torch.load(cache, weights_only=True)
    from pathlib import Path
    def no_open(*args, **kwargs):
        raise AssertionError('Synthesis must not reopen reference or model files')
    monkeypatch.setattr(Path, 'open', no_open)
    a, report = backend.synthesize_prepared('테스트입니다.', prepared)
    b, report2 = backend.synthesize_prepared('테스트입니다.', reloaded)
    assert a.dtype == np.float32 and a.ndim == 1 and len(a) == 7680
    assert np.array_equal(a, b)
    assert report['native_pcm_sha256'] == hashlib.sha256(np.arange(5120, dtype=np.int16).tobytes()).hexdigest()
    assert report['native_pcm_sha256'] == report2['native_pcm_sha256']
    assert report['sample_rate'] == 48000 and report['native_sample_rate'] == 32000
    assert report['reference_preprocessing_on_request'] is False
    assert backend.pipeline.prompt_cache['prompt_semantic'] is None
    assert backend.pipeline.prompt_cache['runtime']['decode_ge'] is None
    assert backend.ready


def test_invalid_cache_releases_previous_reference_and_does_not_poison_engine():
    backend = engine()
    good = reference()
    backend._restore_reference(good)
    wrong = dict(good, engine_identity=_encode('another-model'))
    with pytest.raises(ValueError, match='fingerprint'):
        backend.synthesize_prepared('테스트', wrong)
    assert backend.pipeline.prompt_cache['prompt_semantic'] is None
    assert backend.ready
    wrong = dict(good, spec=torch.full((1, 1025, 5), float('nan')))
    with pytest.raises(ValueError, match='finite'):
        backend.synthesize_prepared('테스트', wrong)


def test_upstream_failure_rejects_teacher_reload_reuse_and_clears_reference():
    backend = engine()
    def fail(inputs):
        raise RuntimeError('Upstream may have reloaded original teacher models')
    backend.runtime.run = fail
    with pytest.raises(RuntimeError, match='teacher'):
        backend.synthesize_prepared('테스트', reference())
    assert not backend.ready
    assert backend.pipeline.prompt_cache['prompt_semantic'] is None
    with pytest.raises(RuntimeError, match='recreate'):
        backend.synthesize_prepared('테스트', reference())
