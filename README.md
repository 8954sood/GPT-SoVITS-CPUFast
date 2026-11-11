# ARM64 V2ProPlus optimized fork

This fork adds an **opt-in** CPU inference engine for GPT-SoVITS V2ProPlus. It is tuned for one measured deployment proxy: Linux ARM64, a 0.7426 CPU cgroup quota (7,426 / 10,000 µs), CPUs 0–1, 8,000,000,000 bytes of memory and no swap. It is not a general performance guarantee and it does not emulate or benchmark a Neoverse N1. Measure again on the actual target. The original CPUFast interfaces remain available; the original project documentation follows below.

## Experimental Neoverse N1 tuning

`optimize(..., ar_threads=2, acoustic_threads=2, tap_small=True)` optionally uses two threads per stage and tap GEMMs for eligible decoder FFT convolutions with up to 48 input channels. The default remains one thread per stage with tap GEMMs off. The native build enables OpenMP for the parallel GEMV and packed FFT kernels; use a compiler and OpenMP runtime available for the target PyTorch environment. Rebuild the source-checked library before testing.

On an actual 2-vCPU Neoverse N1 with an 8,000,000,000-byte container limit and no swap, an experimental `head_major`, `cook_phase`, `fft_layout`, `packed_fft`, `fused_finish` configuration with both thread settings at 2 and `GOMP_SPINCOUNT=100000` produced warmed RTF medians of 1.554 without and 1.508 with tap GEMMs, alternating three runs per variant after one warmup each. The 5.98-second output includes 0.3 seconds of appended silence. On this one Japanese sentence, semantic tokens matched and 25 of 191,360 PCM samples differed by at most one 16-bit unit; a second sentence was checked once. These small, exploratory results do not establish general quality or an RTF below 1. The tap route does not change model weights; FP32 accumulation order can differ. It is an opt-in experiment, not a new default.

## Ongoing RTF 0.7 research

The latest experimental checkpoint reached warmed median RTF **0.849** on the first Japanese sentence (three measured runs). The 0.7 target remains open. These experimental options are not enabled by default and have not received a new listening approval. See [the checkpoint and next steps](docs/RTF07_CHECKPOINT.md) and [raw experiment reports](results/rtf07_checkpoint/). The final NEON input-packing edit is not included in that measured performance claim and requires a fresh rebuild/validation.

## Results and scope

With the same Japanese reference, inputs, unchanged checkpoints, full HuBERT, FP32 arithmetic, and seed 42, the following warmed results were recorded on a local ARM64 Docker host. Three warmups were excluded and all subsequent measurements were retained. Loading and installation time are separate.

| Run | Warmed median RTF | Individual RTFs |
| --- | ---: | --- |
| Previous approved phase/layout runtime, sentence 1 | 1.731 | 1.718, 1.731, 1.735 |
| Optimized runtime, sentence 1 | 0.883 | 0.883, 0.854, 0.885, 0.878, 0.976 |
| Optimized runtime, sentence 1, independent repeat | 0.874 | 0.863, 0.856, 0.874, 0.915, 0.884 |
| Optimized runtime, sentence 2 | 0.936 | 0.936, 0.951, 0.902, 0.900, 0.939 |

RTF is elapsed synthesis time divided by generated audio duration; smaller is faster. The second sentence did not reach the 0.8 range. The cgroup proxy was calibrated against a V5 throughput observation and does not reproduce N1 microarchitecture or stage ratios. Different hardware, text, model, CPU limits, competing workloads or cold starts can change results. Raw timings, cgroup settings, hashes and stage measurements are in [`results/arm64_proxy/`](results/arm64_proxy/) and [`config/cpufast_proxy.json`](config/cpufast_proxy.json). The benchmark used the original non-Chinese zero-BERT path; it skipped loading Chinese BERT and was tested with Japanese inputs.

Relative to the prior approved runtime, both test sentences had matching sample rate and duration; maximum PCM difference was one 16-bit integer step (about 90 dB SNR). The user listened to and accepted the before/after pairs. This does not establish equivalence for all text, voices, or languages.

## What changed

- Serialize access to the mutable pipeline and use one PyTorch intra-op thread for both autoregressive (AR) and acoustic inference under this specific quota.
- Rewrite decoder transposed convolutions as equivalent phase convolutions and shuffles, and cache convolution layouts.
- Use overlap-save FFT convolutions and Winograd F(4,3) kernels for selected decoder layers, with fused Linux ARM64 native transforms and FP32 arithmetic.
- Use a native FP32 AR matrix-vector operator and padded predictor layout. A disposable FP16 **storage cache** is constructed only when every original FP32 coefficient round-trips exactly; arithmetic remains FP32 and the original parameters remain authoritative. Unsupported coefficients fail closed.
- Verify in-memory AR and acoustic parameter hashes and checkpoint hashes before and after inference. No training, pruning, lossy conversion of model weights, or checkpoint rewriting was performed. Floating-point evaluation order can still change the waveform slightly.

The native operator and caches are specialized to this profile; the optimized engine is not automatically enabled for the original WebUI or API. Changing weights requires creating a fresh pipeline and engine. Keep other inference workloads in separate processes because PyTorch thread settings are process-wide.

## Build and use the optimized engine

Install CPUFast dependencies and obtain the V2ProPlus checkpoints, speaker verification model, full HuBERT and language resources using the upstream setup below. Models and generated audio are not included. On **Linux ARM64**, with the same PyTorch environment used for inference and a C++ compiler, build the native operator from this repository:

```sh
python build_cpufast_ar.py --output "$PWD/libcpufast_ar_torch.so"
```

A manifest is written alongside the library. The loader checks the library, source hashes and PyTorch version. Rebuild after changing PyTorch or native source files. `CPUFAST_AR_LIBRARY` can select a different library path.

After constructing a CPU `TTS` pipeline with `v2ProPlus` and `is_half=False`, opt in explicitly:

```python
import torch
from cpufast_engine import optimize

# Set once at process startup, before starting parallel PyTorch work.
torch.set_num_interop_threads(1)
# pipeline is a fully initialized, evaluation-mode CPUFast TTS instance.
engine = optimize(pipeline)
for sample_rate, audio in engine.run(request):
    pass  # Consume the normal CPUFast audio chunks.
```

The engine checks the model and parameter hashes during installation. Use `engine.run(...)`, not the original pipeline's `run(...)`, to obtain the serialized scheduling. This engine is experimental; its WebUI/API integration and streaming behavior have not been validated. The benchmark entry point accepts `--engine`; run `python benchmark_cpufast.py --help` for required model and reference paths. Run inside your own Linux ARM64 container with the quota above, appropriate model mounts and available fast-language-detection/OpenJTalk assets. The image used for the recorded experiment was a locally built V5 image, identified by its SHA in the profile, and is not distributed here, so an independently assembled environment may not reproduce identical timing.

For example, with model and reference paths already available inside the container (and `GPT_SoVITS` on `PYTHONPATH`):

```sh
mkdir -p local-results
PYTHONPATH="$PWD:$PWD/GPT_SoVITS" python resource_guard.py \
  --report local-results/guard.json -- python benchmark_cpufast.py \
  --engine --s1 "$S1_PATH" --s2 "$S2_PATH" --hubert "$HUBERT_DIR" \
  --reference "$REFERENCE_WAV" --prompt-text "$PROMPT_TEXT" \
  --text "$SYNTHESIS_TEXT" --lang ja --warmups 3 --samples 5 \
  --output local-results/measurement.json
```

Supply the paths and texts yourself. `resource_guard.py` adds a sampled process-tree watchdog; use Docker cgroup flags as well for the measured hard quota and memory limit. The generated report includes every warmup and measured run. A fresh output filename is required.

## Attribution

Based on [baicai-1145/GPT-SoVITS-CPUFast](https://github.com/baicai-1145/GPT-SoVITS-CPUFast), pinned at `6a5ab49409a4e9e131e6c9d4c1b7ce4ad910fc61`. Original project license and upstream attribution remain in this repository.

---

# GPT-SoVITS CPU Inference Fork

Inference-only GPT-SoVITS fork focused on CPU deployment and CPU-side optimization on Windows, Linux, and macOS.

[中文简体](./docs/cn/README.md)

## What This Repo Is

- Inference-only fork of GPT-SoVITS.
- Designed around CPU usage rather than GPU-first features.
- Current practical focus is the S2 `v2Pro` / `v2ProPlus` path, while keeping versioned pretrained downloads for `v1`, `v2`, `v2Pro`, and `v2ProPlus`.

## What Was Removed

This repository no longer keeps most training and dataset-preparation features from upstream.

- Training entrypoints and training-only utilities
- Dataset slicing / denoise / ASR / labeling workflows
- UVR5 and other non-inference WebUI tools

The remaining goal is straightforward: run GPT-SoVITS inference on CPU with less installation friction and a smaller runtime surface.

## What Still Works

- `webui.py`: minimal inference launcher
- `GPT_SoVITS/inference_webui_fast.py`: high-performance CPU inference WebUI
- `api.py` and `api_v2.py`: inference APIs

## Quick Start

### 1. Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a C/C++ compiler toolchain, then sync the locked CPU inference dependencies into `.venv`:

```bash
uv sync --locked
```

The project supports Python 3.10-3.12. CPU PyTorch is selected by the uv project configuration. For a one-command pip setup with CPU PyTorch, run `python -m pip install -r requirements.txt` from the repository root. Model weights remain separate downloads.

### 2. Download Inference Weights

CPU example with ModelScope and `v2ProPlus`:

```bash
bash install.sh --source ModelScope --version v2ProPlus
```

Windows PowerShell:

```powershell
.\install.ps1 -Source ModelScope -Version v2ProPlus
```

Available versions:

- `v1`
- `v2`
- `v2Pro`
- `v2ProPlus`
- `all`

### 3. Launch

Recommended:

```bash
uv run --locked gpt-sovits-webui
```

Direct high-performance inference WebUI:

```bash
uv run --locked python GPT_SoVITS/inference_webui_fast.py
```

After installing the project with pip, the same launcher is available as `gpt-sovits-webui`. Set `GPT_SOVITS_DATA_DIR` when model files should live outside the current directory:

```bash
GPT_SOVITS_DATA_DIR=$HOME/.local/share/gpt-sovits gpt-sovits-webui
```

The package contains Python code and small text assets. Inference weights, BERT/CN-HuBERT files, G2PW, NLTK data, and the Open JTalk dictionary remain separate downloads because of their size and licensing/source choices.

## Notes

- This fork is aimed at CPU inference, not training.
- Chinese inference is still heavier than English / Japanese / Korean because text preprocessing needs extra frontend work such as `g2pw` and BERT features.
- `install.sh` and `install.ps1` use `uv sync --locked` and download inference assets by version; model weights are not bundled in the Python package.
- A platform C/C++ compiler toolchain is needed when a dependency has to build from source. Windows users should install Visual Studio Build Tools with the C++ workload.
- `NLTK` and `OpenJTalk` dictionary downloads remain enabled by default.

## Optional Lightweight Frontends

Two inference options can be toggled at runtime (WebUI checkboxes, or environment variables for API deployments). Both default to the original behavior.

### 1. `g2pS` polyphone disambiguation (lighter replacement for `g2pw`)

- Toggle: WebUI checkbox `多音字消歧 g2pS` or `G2P_BACKEND=g2ps` (default `g2pw`; `pypinyin` = dictionary only).
- Deps: `pip install g2ps pypinyin-g2pS` — the `g2ps` wheel bundles a 2.1 MB numpy-only `nano` model (vs 637 MB `g2pw.pth`), no ONNX/torch runtime needed.
- Measured on the real `clean_text` path (CPU, 8 threads, min of 10 runs): `g2pw` ~600+ ms / 180 chars vs `g2pS` **~83 ms / 180 chars (0.46 ms/char, ~7x faster)**; engine cost is ~7 ms/sentence (numpy transformer, per-sentence context needed for disambiguation); polyphone outputs identical to `g2pw` on test sentences (银行/行踪/重量级 all correct).
- Includes a word-level pinyin result cache (`text/pinyin_cache.py`) that also speeds up the default `g2pw` path (tone-sandhi merge passes and phrase-override lookups reuse memoized results; outputs verified identical).
- Uses [pypinyin-g2pS](https://pypi.org/project/pypinyin-g2pS/) as the polyphone decision engine; monophonic characters keep pypinyin's dictionary path.

### 2. Distilled HuBERT frontend (~32x smaller)

- Toggle: WebUI checkbox `蒸馏版 HuBERT` or `HUBERT_FRONTEND=distilled` (`hubert` is the default).
- Weights ship with this repo: `GPT_SoVITS/pretrained_models/distilled_hubert/student_v0.pt` (11.9 MB, 2.97M params vs 95M).
- Distilled from `chinese-hubert-base` on edge-tts + AISHELL-1; e2e paired synthesis is at the teacher's own cross-seed noise level (token norm_lev 0.843 vs null 0.815, CER median 0, speaker cosine 0.913 > teacher cross-seed 0.845). Report: upstream `distill_lab/REPORT_student.md`.

## Speed Summary

Without changing recognition behavior, prosody, speaker identity, or audio quality, this CPU fork currently delivers:

- Chinese end-to-end `zh_pure`: `wall_sec 15.136264 -> 8.309471`, about `-45.1%`
- End-to-end: `wall_sec 10.431826 -> 7.046743`, about `-32.4%`
- Preprocessing: `frontend_sec 0.867065 -> 0.569571`, about `-34.3%`
- T2S: `t2s_sec 4.023657 -> 2.268571`, about `-43.6%`
- VITS: `vits_sec 5.137876 -> 3.969286`, about `-22.7%`

Largest landed stage wins so far:

- Preprocessing:
  - Chinese warm multi-sentence BERT frontend: `0.855s -> 0.131s`, about `-84.7%`
  - Korean cold frontend path: `0.791s -> 0.155s`, about `-80.4%`
- T2S: `stable_batch_remap` 7-case comparison `1.986543 -> 1.684446`, about `-15.2%`
- VITS: `remove_weight_norm` `vits_only` comparison `4.256119 -> 4.102004`, about `-3.6%`

## How The Speedups Were Achieved

These gains do not come from sentence-level caching, quantization, or quality tradeoffs. The main work was removing CPU-side Python overhead, repeated preparation, repeated copies, and unnecessary cold-start costs from the real inference path.

Chinese shows an even larger end-to-end gain than the overall average because it started from the heaviest path: it pays for both `g2pw` and BERT text features, so frontend reductions propagate directly into total latency.

### What Was Deliberately Not Used

- ONNX / ORT was not adopted. `dec-only ORT`, `flow + dec ORT`, and larger graph-level ORT experiments were all tried, but on the current machine and dependency stack they did not produce a solution that was simultaneously quality-safe, faster, and lighter than the PyTorch path, so the runtime stayed on pure PyTorch and the ONNX compatibility path was removed.
- The Chinese path was not simplified by dropping quality-critical frontend pieces. `g2pw` stayed, Chinese BERT stayed, and the project did not switch to lighter but lower-quality replacements such as `g2pm`, which are more likely to cause noticeable G2P errors on harder text, especially literary or classical material.
- The main path also does not rely on secondary splitting just to manufacture larger batches. That direction was explored in benchmark-only form, but the results did not stay stable, and the `repeats=3` verification did not justify moving it into the runtime path.
- VITS parallel synthesis is not exposed in the CPU WebUI. Local CPU measurements showed it can add overhead and slow inference, so the high-performance WebUI keeps VITS synthesis serial while preserving T2S parallel inference.

### Preprocessing / Frontend

- English frontend work focused on cold start. Heavy first-request dependencies around `g2p_en.G2p` were reduced, `wordsegment.load()` was made lazy, `nltk.pos_tag` was narrowed to heteronym cases, and import-time overhead from `inflect/typeguard` was trimmed.
- Korean frontend work removed an unrelated cold-start chain. `g2pk2` used to pull in `nltk/cmudict` even for pure Korean input; that path is now stubbed lazily so Korean requests no longer pay for the English dictionary on first use.
- The biggest Chinese frontend win came from changing pure-Chinese multi-sentence BERT extraction from sentence-by-sentence serial execution to a batched path. The current path batches tokenization, BERT forward, and `word2ph` alignment for pure Chinese multi-sentence requests.
- Chinese `g2pw` cold start was also reduced by lazily importing `requests`, shrinking tokenizer initialization down to direct `tokenizer.json` loading, and avoiding the larger `transformers` auto-dispatch chain during startup.
- Chinese segmentation / POS loading was further reduced with local static assets so `jieba_fast.posseg` initialization does less repeated work, without introducing any sentence-level cache tied to user input.
- Non-Chinese zero-BERT paths also gained simple length-based zero-tensor reuse so repeated all-zero feature allocation does less work.

### T2S

- The first layer of gains came from removing pure Python overhead from the hot path, including `tqdm` and hot-loop prints during decoding. These changes do not alter model numerics, but they do matter on CPU.
- The next layer came from the shrink path. Previously, when some rows in a batch finished early, the code copied whole future-capacity buffers with `index_select`, including token buffers, KV caches, and masks that were never going to be used. The current path compacts only the valid prefix.
- Another landed gain is `stable_batch_remap`. It is not a “never shrink” hack. It keeps exact behavior while stabilizing how active rows are remapped inside the batch, which reduces unnecessary compaction churn.
- The deeper T2S gains came from addressing the actual `addmm` hotspots. The main path now uses exact-safe hybrid linear execution for high-frequency layers such as `MLP`, `qkv_proj`, and `out_proj`: `rows == 1` keeps the original `F.linear`, while larger cases use a more CPU-friendly `torch.addmm` path.
- That backend work only became safe after fixing the load path as well. `t2s_transformer` is rebuilt after checkpoint loading so cached transposed weights are bound to the real loaded parameters instead of the pre-load initialization weights.

### VITS

- VITS optimization stayed conservative. The main strategy was to remove work that was being repeated for every batch instead of rewriting the model structure.
- The first landed step was a run-level runtime cache in `TTS.run()` for reference-side objects that do not depend on the current input text, such as `refer_audio_spec`, `sv_emb`, `prompt_semantic_tokens`, and `prompt_phones`.
- The second landed step moved decode-condition preparation out of repeated decode calls. `build_decode_condition()` lets `ge / ge_text` be computed once before the batch loop and then reused across decode calls.
- The third landed step applies `remove_weight_norm()` directly to non-vocoder `Generator.dec`, removing weight-norm reparameterization overhead during inference. This is the main clearly logged stage-local VITS win currently kept in the codebase.
- The worklog also records more aggressive VITS experiments that were tested but not kept, such as traced `dec` and more aggressive decode layout variants. The README only describes the parts that actually landed.

### Load Path And Memory

- `t2s_only` benchmark loading used to instantiate the full `TTS()` pipeline and then trim unused objects, which pushed peak memory far too high. That was replaced with a true lightweight load path that initializes only the minimum T2S-side objects.
- The main inference load path was also reordered to avoid rebuilding large transposed-weight structures while the checkpoint dictionary was still alive in memory.
- The biggest memory-side change is that inference no longer builds `self.h` at all on the main T2S path. The runtime now rebuilds `t2s_transformer` directly from the state dict and only keeps what inference actually uses.
- This is not benchmark-only machinery. It is wired into the real inference path and mainly helps reduce steady-state RSS and peak RSS so CPU machines are less likely to stall or be reclaimed under memory pressure.

## Upstream and Credits

This project is based on and uses code from:

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

This fork keeps upstream credits and referenced projects below.

## Referenced Projects

### Theoretical Research

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)

### Main Model / Training / Vocoder Related

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [SoVITS](https://github.com/voicepaw/so-vits-svc-fork)
- [GPT-SoVITS-beta](https://github.com/lj1995/GPT-SoVITS/tree/gsv-v2beta)
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

### Text Frontend for Inference

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)

### Inherited Upstream Tool References

These projects were referenced by upstream GPT-SoVITS. Some related modules are removed in this inference-only fork, but credits are preserved here.

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)
