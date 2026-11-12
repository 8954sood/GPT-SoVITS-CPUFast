"""Single-owner approved CPUFast inference adapter with durable tensor references.

Native32kHz PCM16 is sent directly; this is sentence-fragment inference,
not token streaming. No resampling or second quantization is applied.
"""
from __future__ import annotations

import functools
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from cpufast_student_assets import inline_runtime_approval, load_manifest, sha256

APPROVED_MANIFEST_SHA256 = '696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865'
REFERENCE_KEYS = frozenset({'tokens', 'phones', 'bert_features', 'spec', 'reference_audio16',
    'sv_emb', 'decode_ge', 'decode_ge_text', 'prompt_text', 'prompt_lang', 'norm_text',
    'reference_identity', 'engine_identity', 'schema_version'})
TEXT_KEYS = frozenset({'prompt_text', 'prompt_lang', 'norm_text', 'reference_identity', 'engine_identity'})
LANGUAGES = frozenset({'ko', 'ja', 'en'})
REFERENCE_SCHEMA = 1
ROOT = Path(__file__).resolve().parent


def _encode(value):
    return torch.tensor(list(value.encode('utf-8')), dtype=torch.uint8)


def _decode(value):
    if value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError('Prepared text metadata must be a UTF8 uint8 vector')
    try:
        return bytes(value.tolist()).decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError('Invalid prepared UTF8 metadata') from error


def _empty_prompt():
    return dict(ref_audio_path=None, prompt_semantic=None, refer_spec=[], raw_audio=None,
                raw_sr=None, prompt_text=None, prompt_lang=None, phones=None,
                bert_features=None, norm_text=None, aux_ref_audio_paths=[], runtime={})


def _versions(tensors):
    return [(t._version, t.data_ptr(), tuple(t.shape), str(t.dtype)) for t in tensors]


def _ar_tensors(transformer):
    tensors = []
    for cached in transformer.blocks:
        b = cached.block if hasattr(cached, 'block') else cached
        tensors.extend((b.qkv_w_t, b.qkv_b, b.out_w_t, b.out_b, b.mlp.w1_t,
                        b.mlp.b1, b.mlp.w2_t, b.mlp.b2, b.norm_w1, b.norm_b1, b.norm_w2, b.norm_b2))
    return tensors


class Engine:
    reference_keys = REFERENCE_KEYS
    sample_rate = 32000
    native_sample_rate = 32000

    def __init__(self, settings):
        if platform.system() != 'Linux' or platform.machine() != 'aarch64':
            raise RuntimeError('Approved native CPUFast engine requires Linux ARM64')
        if os.environ.get('HUBERT_FRONTEND', '').strip().lower() in ('student', 'distilled'):
            raise ValueError('Approved CPUFast API requires original full HuBERT frontend')
        self.reference_segment = None
        self._initialized = False
        self._lock = threading.RLock()
        self._failed = False
        self._semantic = []
        self._stages = {}
        self.ar_threads = settings.ar_threads
        self.threads = settings.threads
        if self.ar_threads not in (1, 2) or self.threads not in (1, 2):
            raise ValueError('CPUFast stages require one or two CPU threads')
        torch.set_num_threads(self.threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        self.manifest_path = Path(settings.manifest).resolve()
        self.manifest_sha256 = getattr(settings, 'manifest_sha256', APPROVED_MANIFEST_SHA256)
        if self.manifest_sha256 != APPROVED_MANIFEST_SHA256:
            raise ValueError('This API engine requires the exact approved AR17/C24 Vtarget100 release')
        self.manifest = load_manifest(self.manifest_path, self.manifest_sha256)
        inline_runtime_approval(self.manifest, self.manifest_sha256, ROOT)
        s2, hubert = Path(settings.s2).resolve(), Path(settings.hubert).resolve()
        if sha256(s2) != self.manifest['original_s2_sha256']:
            raise ValueError('Original S2 teacher checksum mismatch')
        upstream = str(ROOT/'GPT_SoVITS')
        if upstream not in sys.path:
            sys.path.insert(0, upstream)
        os.environ.setdefault('GPT_SOVITS_DATA_DIR', str(ROOT))
        from TTS_infer_pack.TTS import TTS, TTS_Config
        from student_runtime import StudentGenerator
        from cpufast_engine import optimize
        import cpufast_fast_ops
        from cpufast_fast_ar import build
        import cpufast_cpp_ops

        class NonChineseTTS(TTS):
            def init_bert_weights(self, base_path):
                pass

            def empty_cache(self):
                # This dedicated CPU process has no device cache to flush. Keep
                # automatic generational GC enabled; collect young request cycles
                # without rescanning all loaded models/libraries per sentence.
                gc.collect(0)

        directory = self.manifest_path.parent
        self.pipeline = NonChineseTTS(TTS_Config({'custom': {
            'device': 'cpu', 'is_half': False, 'version': 'v2ProPlus',
            't2s_weights_path': str(directory/'ar.pt'), 'vits_weights_path': str(s2),
            'cnhuhbert_base_path': str(hubert), 'bert_base_path': '.',
        }}))
        spec = json.loads((directory/'student_spec.json').read_text())
        trained = torch.load(directory/'decoder.pt', map_location='cpu', weights_only=True)
        original_ar = self.pipeline.t2s_model.model.t2s_transformer
        if (len(original_ar.blocks) != 17 or self.pipeline.t2s_model.model.num_layers != 17
            or len(spec['ar_keep']) != 17 or spec['decoder_branches'] != self.manifest['decoder_branches']
            or sum(map(len, spec['decoder_branches'])) != 11):
            raise ValueError('Actual approved AR17/11-residual-block geometry required')
        for key in ('decoder_branches', 'decoder_bottlenecks', 'decoder_channel_keep'):
            if trained['student_spec'].get(key) != spec.get(key) or spec.get(key) != self.manifest.get(key):
                raise ValueError('Decoder checkpoint/spec/manifest geometry mismatch: '+key)
        if trained['provenance']['teacher_s2_sha256'] != self.manifest['original_s2_sha256']:
            raise ValueError('C24 decoder teacher checksum mismatch')
        self.pipeline.vits_model.dec = StudentGenerator(self.pipeline.vits_model.dec,
            spec['decoder_branches'], channel_keep=spec.get('decoder_channel_keep'),
            bottlenecks=spec.get('decoder_bottlenecks')).eval()
        self.pipeline.vits_model.dec.load_state_dict(trained['state_dict'], strict=True)
        del trained
        self.runtime = optimize(self.pipeline, head_major=True, cook_phase=True,
            fft_layout=True, packed_fft=True, fft_stream=True, parallel_transforms=True,
            fused_finish=True, tap_small=True, ar_threads=self.ar_threads,
            acoustic_threads=self.threads, ar_fp32_cache=False, ar_prefetch=256)
        approximate, audit = build(original_ar, head_major=True)
        self.pipeline.t2s_model.model.t2s_transformer = approximate
        self.runtime.audit['ar_coefficients'] = audit
        self.runtime_audit = audit
        from cpufast_fft import FFTConv1d
        from cpufast_tap import TapConv1d
        ffts = [m for m in self.pipeline.vits_model.dec.modules() if isinstance(m, FFTConv1d)]
        if (self.runtime.audit['fast_experimental'] or len(approximate.blocks) != 17
            or len(self.pipeline.vits_model.dec.resblocks) != 11
            or any(m.n_fft != (128 if m.weight.shape[1] >= 256 else 256)
                   or m.fast_half or m.half_stream_products for m in ffts)
            or any(getattr(m.forward, '__func__', None) is not TapConv1d.forward for m in self.pipeline.vits_model.dec.modules() if isinstance(m, TapConv1d))):
            raise RuntimeError('Canonical FP32 acoustic/AR runtime geometry changed')
        head_geometry = audit.get('head_geometry', [])
        if (len(head_geometry) != 17 or any(r['heads'] != 16 or r['dim'] != 32 for r in head_geometry)
            or cpufast_fast_ops.manifest.get('head_value_schedule', {}).get('dim32') != 'eight_independent_vectors_preserve_token_FMA_order'):
            raise RuntimeError('Canonical native head-value source/geometry required')
        self._parameter_versions = self._current_versions()
        native_paths = [cpufast_cpp_ops.path, cpufast_fast_ops.path]
        native_paths += [p.with_suffix(p.suffix+'.json') for p in native_paths]
        self.native_files_sha256 = {p.name: sha256(p) for p in native_paths}
        self.source_sha256 = self._source_hashes()
        import sv
        inputs = {'manifest': self.manifest_sha256, 's2': sha256(s2), 'speaker_encoder': sha256(Path(sv.sv_path))}
        inputs.update({'hubert/'+p.relative_to(hubert).as_posix(): sha256(p)
                       for p in sorted(hubert.rglob('*')) if p.is_file() and
                       (p.suffix in ('.bin', '.safetensors') or p.name in ('config.json', 'preprocessor_config.json'))})
        if not any(k.startswith('hubert/') for k in inputs):
            raise ValueError('Local full HuBERT model files required')
        versions = {}
        for package in ('torch', 'torchaudio', 'transformers', 'av', 'pyopenjtalk', 'g2pk2', 'g2p-en'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.input_sha256 = inputs
        recipe = {'schema': REFERENCE_SCHEMA, 'inputs': inputs, 'source': self.source_sha256,
                  'native': self.native_files_sha256, 'libraries': versions,
                  'threads': {'ar': self.ar_threads, 'acoustic': self.threads, 'interop': 1},
                  'cpu_cleanup': 'generation0_per_utterance_automatic_gc_retained',
                  'profile': 'approvedAR17_C24Vtarget100_W8A8_canonicalTap_FFT256_value_schedule'}
        self.fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        self._capture_semantics()
        self.clear_reference()
        self._initialized = True

    @property
    def ready(self):
        return getattr(self, '_initialized', False) and not self._failed

    def _current_versions(self):
        transformer = self.pipeline.t2s_model.model.t2s_transformer
        return _versions(list(self.pipeline.vits_model.parameters()) + list(self.pipeline.t2s_model.model.parameters()) + _ar_tensors(transformer))

    @staticmethod
    def _source_hashes():
        names = ('cpufast_server_engine.py', 'cpufast_engine.py', 'cpufast_fast_ar.py', 'cpufast_kv_cache.py',
            'cpufast_ar_cpp.py', 'cpufast_fft.py', 'cpufast_tap.py', 'cpufast_fused_blocks.py',
            'cpufast_cooktoom.py', 'cpufast_optimizations.py', 'cpufast_predictor.py',
            'cpufast_winograd.py', 'student_runtime.py', 'GPT_SoVITS/TTS_infer_pack/TTS.py',
            'GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py', 'GPT_SoVITS/module/models.py',
            'GPT_SoVITS/AR/models/t2s_model.py', 'tools/audio_utils.py',
            'GPT_SoVITS/sv.py', 'GPT_SoVITS/feature_extractor/cnhubert.py', 'GPT_SoVITS/module/mel_processing.py')
        result = {n: sha256(ROOT/n) for n in names}
        for p in sorted(list((ROOT/'GPT_SoVITS/text').rglob('*.py')) + list((ROOT/'GPT_SoVITS/eres2net').rglob('*.py'))):
            result[p.relative_to(ROOT).as_posix()] = sha256(p)
        return result

    def _capture_semantics(self):
        decoder = self.pipeline.t2s_model.model
        original = decoder.infer_panel_batch_infer
        @functools.wraps(original)
        def infer(*args, **kwargs):
            begin = time.perf_counter()
            result = original(*args, **kwargs)
            self._stages['ar_seconds'] = time.perf_counter()-begin
            self._semantic.extend({'tokens': int(i), 'sha256': hashlib.sha256(t[-int(i):].detach().cpu().numpy().tobytes()).hexdigest()} for t, i in zip(*result))
            return result
        decoder.infer_panel_batch_infer = infer

    def clear_reference(self):
        with self._lock:
            self.pipeline.prompt_cache = _empty_prompt()
            self.pipeline._invalidate_prompt_runtime_cache()

    @torch.inference_mode()
    def prepare_reference(self, path, transcript, lang):
        import soundfile as sf
        from TTS_infer_pack.text_segmentation_method import splits
        if lang not in LANGUAGES or not transcript.strip():
            raise ValueError('Nonempty KO/JA/EN reference transcript required')
        path = Path(path).resolve()
        source_hash = sha256(path)
        segment = self.reference_segment
        transcript = transcript.strip('\n')
        if transcript[-1] not in splits:
            transcript += '.' if lang == 'en' else '。'
        with self._lock:
            if self._failed:
                raise RuntimeError('CPUFast engine failed; recreate the engine')
            try:
                with tempfile.TemporaryDirectory(prefix='cpufast-reference-') as temporary:
                    prepared_path = path
                    if segment is not None:
                        start, end = segment
                        if not np.isfinite(start) or start < 0 or (end is not None and (not np.isfinite(end) or end <= start)):
                            raise ValueError('Invalid reference crop segment')
                        if start != 0 or end is not None:
                            info = sf.info(path)
                            audio, sr = sf.read(path, start=round(start*info.samplerate),
                                stop=None if end is None else round(end*info.samplerate),
                                dtype='float32', always_2d=True)
                            if not len(audio):
                                raise ValueError('Reference crop is empty')
                            prepared_path = Path(temporary)/'crop.wav'
                            sf.write(prepared_path, audio, sr, subtype='FLOAT')
                    self.pipeline.set_ref_audio(str(prepared_path))
                    phones, bert, normalized = self.pipeline.text_preprocessor.segment_and_extract_feature_for_text(transcript, lang, self.pipeline.configs.version)
                    cache = self.pipeline.prompt_cache
                    cache.update(prompt_text=transcript, prompt_lang=lang, phones=phones,
                                 bert_features=bert, norm_text=normalized)
                    self.pipeline._invalidate_prompt_runtime_cache()
                    self.pipeline._get_runtime_refer_audio_spec_and_sv_emb()
                    ge, ge_text = self.pipeline._get_runtime_decode_condition()
                    runtime = cache['runtime']
                    identity = json.dumps({'sha256': source_hash, 'segment': segment}, sort_keys=True)
                    result = {'tokens': cache['prompt_semantic'], 'phones': torch.tensor(phones, dtype=torch.long),
                        'bert_features': bert, 'spec': cache['refer_spec'][0][0],
                        'reference_audio16': cache['refer_spec'][0][1], 'sv_emb': runtime['sv_emb'][0],
                        'decode_ge': ge, 'decode_ge_text': ge_text, 'prompt_text': _encode(transcript),
                        'prompt_lang': _encode(lang), 'norm_text': _encode(normalized),
                        'reference_identity': _encode(identity), 'engine_identity': _encode(self.fingerprint),
                        'schema_version': torch.tensor([REFERENCE_SCHEMA], dtype=torch.long)}
                    result = {k: v.detach().cpu().contiguous().clone() for k, v in result.items()}
                self._validate_reference(result)
                if sha256(path) != source_hash:
                    raise RuntimeError('Reference input changed during preparation')
                return result
            finally:
                self.clear_reference()

    def _validate_reference(self, reference):
        if set(reference) != REFERENCE_KEYS:
            raise ValueError('Invalid prepared reference keys')
        for key, value in reference.items():
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or not value.numel() or not torch.isfinite(value).all():
                raise ValueError('Prepared reference must contain nonempty finite CPU tensors')
            if key in TEXT_KEYS:
                _decode(value)
        if reference['schema_version'].dtype != torch.long or reference['schema_version'].tolist() != [REFERENCE_SCHEMA]:
            raise ValueError('Prepared reference schema mismatch')
        if _decode(reference['engine_identity']) != self.fingerprint:
            raise ValueError('Prepared reference belongs to another model/source fingerprint')
        try:
            identity = json.loads(_decode(reference['reference_identity']))
        except (ValueError, TypeError) as error:
            raise ValueError('Invalid prepared reference identity') from error
        if (not isinstance(identity, dict) or set(identity) != {'sha256', 'segment'}
            or not re.fullmatch(r'[0-9a-f]{64}', identity['sha256'] or '')):
            raise ValueError('Invalid prepared audio identity checksum')
        for name in REFERENCE_KEYS - TEXT_KEYS - {'tokens', 'phones', 'schema_version'}:
            if reference[name].dtype != torch.float32:
                raise ValueError('Prepared acoustic/BERT references require FP32')
        if _decode(reference['prompt_lang']) not in LANGUAGES or reference['tokens'].dtype != torch.long or reference['phones'].dtype != torch.long:
            raise ValueError('Invalid reference language or semantic/phone dtype')
        if reference['tokens'].ndim != 1 or reference['phones'].ndim != 1 or reference['spec'].ndim != 3 or reference['bert_features'].ndim != 2 or reference['bert_features'].shape[0] != 1024 or reference['bert_features'].shape[1] != reference['phones'].numel():
            raise ValueError('Invalid prepared semantic/phone/spec/BERT shapes')

    def _restore_reference(self, reference):
        self._validate_reference(reference)
        cache = _empty_prompt()
        marker = 'prepared://'+hashlib.sha256(reference['reference_identity'].numpy().tobytes()).hexdigest()
        cache.update(ref_audio_path=marker, prompt_semantic=reference['tokens'],
            refer_spec=[(reference['spec'], reference['reference_audio16'])],
            prompt_text=_decode(reference['prompt_text']), prompt_lang=_decode(reference['prompt_lang']),
            phones=reference['phones'].tolist(), bert_features=reference['bert_features'], norm_text=_decode(reference['norm_text']))
        self.pipeline.prompt_cache = cache
        self.pipeline._invalidate_prompt_runtime_cache()
        key = self.pipeline._get_ref_runtime_key()
        cache['runtime'].update(ref_key=key, refer_audio_spec=[reference['spec']], sv_emb=[reference['sv_emb']],
            decode_ref_key=key, decode_ge=reference['decode_ge'], decode_ge_text=reference['decode_ge_text'])
        return cache

    @torch.inference_mode()
    def synthesize_prepared(self, text, reference, *, lang='ko', seed=42):
        if lang not in LANGUAGES or not text.strip():
            raise ValueError('Nonempty KO/JA/EN target text required')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Seed must be a uint32 integer')
        with self._lock:
            if self._failed:
                raise RuntimeError('CPUFast engine failed; recreate the engine')
            started = time.perf_counter()
            try:
                cache = self._restore_reference(reference)
                self._semantic, self._stages = [], {}
                inputs = dict(text=text, text_lang=lang, ref_audio_path=None,
                    prompt_text=cache['prompt_text'], prompt_lang=cache['prompt_lang'],
                    top_k=15, top_p=1., temperature=1., text_split_method='cut0', batch_size=1,
                    seed=seed, parallel_infer=True, vits_parallel_infer=False,
                    fragment_interval=.3, return_fragment=False)
                infer_started = time.perf_counter()
                try:
                    chunks = list(self.runtime.run(inputs))
                except BaseException:
                    self._failed = True  # Upstream reloads teacher models on error; never reuse that fallback.
                    raise
                infer_seconds = time.perf_counter()-infer_started
                if len(chunks) != 1 or chunks[0][0] != 32000 or chunks[0][1].dtype != np.int16:
                    self._failed = True
                    raise RuntimeError('Expected one canonical32kHz PCM16 utterance')
                native = np.ascontiguousarray(chunks[0][1])
                native_hash = hashlib.sha256(native.tobytes()).hexdigest()
                if self._current_versions() != self._parameter_versions or len(self.pipeline.t2s_model.model.t2s_transformer.blocks) != 17 or len(self.pipeline.vits_model.dec.resblocks) != 11:
                    self._failed = True
                    raise RuntimeError('Authoritative model parameters or runtime geometry changed')
                identity = json.loads(_decode(reference['reference_identity']))
                approved_case = next((c for c in self.manifest['listening_report']['cases']
                    if c['language'] == lang and c['text'] == text), None)
                approved_match = (identity['sha256'] == self.manifest['reference_sha256']
                    and identity['segment'] in (None, [0, None]) and seed == 42
                    and cache['prompt_lang'] == self.manifest['prompt_language']
                    and cache['prompt_text'].strip() == self.manifest['prompt_text'].strip()
                    and approved_case is not None)
                if approved_match and (native_hash != approved_case['pcm_sha256'] or self._semantic != approved_case['semantic']):
                    self._failed = True
                    raise RuntimeError('Same approved request did not reproduce native PCM/semantic')
                wire = native
                if wire.ndim != 1 or not wire.size:
                    raise RuntimeError('Invalid API wire audio')
                total = time.perf_counter()-started
                report = {'sample_rate': 32000, 'native_sample_rate': 32000,
                    'native_pcm_sha256': native_hash, 'native_samples': len(native),
                    'wire_samples': len(wire), 'audio_seconds': len(wire)/32000,
                    'native_inference_seconds': infer_seconds, 'total_seconds': total,
                    'rtf': total/(len(wire)/32000), 'semantic': list(self._semantic),
                    'semantic_tokens': sum(r['tokens'] for r in self._semantic),
                    'ar_seconds': self._stages.get('ar_seconds'), 'seed': seed,
                    'manifest_sha256': self.manifest_sha256, 'engine_fingerprint': self.fingerprint,
                    'actual_ar_blocks': 17, 'actual_decoder_residual_blocks': 11,
                    'model_parameter_versions_unchanged': True, 'native_files_sha256': self.native_files_sha256,
                    'reference_preprocessing_on_request': False, 'approved_request_pcm_match': True if approved_match else None,
                    'streaming': 'sentence_fragments', 'cfm': False, 'latency_profile': 'fixed_approved_C24'}
                report['cpu_cleanup'] = 'generation0_per_utterance_automatic_gc_retained'
                return wire, report
            finally:
                self.clear_reference()
