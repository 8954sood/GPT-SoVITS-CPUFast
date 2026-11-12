# ARM64 V2ProPlus optimized fork

This fork adds an **opt-in** CPU inference engine for GPT-SoVITS V2ProPlus. It was initially tuned against a local Linux ARM64 proxy with a 0.7426 CPU cgroup quota and has since been measured and further tuned on the target 2-vCPU Neoverse N1 server with an 8,000,000,000-byte memory limit and no swap. The proxy does not emulate N1 microarchitecture; results are specific to each environment. The original CPUFast interfaces remain available; the original project documentation follows below.

## Experimental Neoverse N1 tuning

The FP32 engine now defaults to AR coefficient prefetch distance 256, which preserves the original parameters and changes only memory access hints. On the real 2-vCPU server, independent five-run comparisons of the tap profile improved median RTF from **1.5787 to 1.5394** and **1.5870 to 1.5552** for two Japanese sentences. Every candidate PCM waveform was bit-identical to its respective tap baseline; one candidate run was a slow outlier and is included in the median. Model checkpoint and parameter hashes remained unchanged. This has **not** reached the immediate RTF < 1 target; results depend on the inputs and hardware. See [actual-server research and measurements](docs/TARGET_RESEARCH.md) and the [benchmark workflow guide](ARM_BENCHMARK.md). To disable prefetch, pass `ar_prefetch=0` to `optimize` or `--ar-prefetch 0` to the benchmark. Rebuild the native extension after updating.


`optimize(..., ar_threads=2, acoustic_threads=2, tap_small=True)` optionally uses two threads per stage and tap GEMMs for eligible decoder FFT convolutions with up to 48 input channels. The default remains one thread per stage with tap GEMMs off. The native build enables OpenMP for the parallel GEMV and packed FFT kernels; use a compiler and OpenMP runtime available for the target PyTorch environment. Rebuild the source-checked library before testing.

On an actual 2-vCPU Neoverse N1 with an 8,000,000,000-byte container limit and no swap, an experimental `head_major`, `cook_phase`, `fft_layout`, `packed_fft`, `fused_finish` configuration with both thread settings at 2 and `GOMP_SPINCOUNT=100000` produced warmed RTF medians of 1.554 without and 1.508 with tap GEMMs, alternating three runs per variant after one warmup each. The 5.98-second output includes 0.3 seconds of appended silence. On this one Japanese sentence, semantic tokens matched and 25 of 191,360 PCM samples differed by at most one 16-bit unit; a second sentence was checked once. These small, exploratory results do not establish general quality or an RTF below 1. The tap route does not change model weights; FP32 accumulation order can differ. It is an opt-in experiment, not a new default.

## Separate CPU student research

Separate research tests structured AR layer and decoder branch reduction, followed by CPU-only distillation. Original checkpoints and engine defaults remain intact. Both the untrained students and the first short posttraining pilots were rejected in owner listening tests. After 400-update recovery experiments, the owner accepted combined candidate G (AR20/decoder13) on the Korean comparison, with a small phrase-ending difference. Its first actual-N1 Korean comparison showed almost unchanged RTF: optimized original **1.44398**, G **1.44139**; elapsed synthesis was8.346 and8.504 seconds for5.78 and5.90 seconds of audio. Parameter reduction alone was not a speed success: general FP32 student caches lost the original model's compact coefficient-storage advantage.

H2 reduces AR to18 layers and adds400 updates of original-teacher hidden-feature supervision after400 incremental KL updates. H3 is a separate inference copy that rounds only73 AR linear matrices to FP16-representable values, restoring compact AR cache storage while retaining FP32 multiplication/accumulation. Biases, normalization, embeddings and the approved G decoder13 remain unchanged. H3 is a lossy conversion relative to H2. The owner separately accepted H2 and H3 on the same Korean reference/text/seed; this is limited listening approval, not broad zero-shot or Japanese/English validation. On actual N1, the same H3 coefficients improved from RTF **1.35895** with FP32 cache/prefetch 0 to **1.23130** with exact FP16 coefficient storage/prefetch 256 (6.9306 to 6.2796 seconds, 9.39% less wall time). This measures the combined cache and prefetch profile; all repeated outputs matched exactly between those H3 runtime profiles. Further structural reduction is guided by actual server latency/RTF and listening quality. See the [student method and V5 execution-log review](experiments/student/README.md) and [training, conversion, approval and backup reports](results/student_pilot/).

K conservatively reduces the decoder to 12 residual branches after CPU recovery training. L keeps those weights and uses dynamically quantized AR inputs with per-output-channel INT8 derived weights and native FP32 attention; acoustic operations remain FP32. In the paired actual-N1 Korean test (three warmups and five measured runs each), K reached **1.22759** and L **1.14142** RTF, with median synthesis time **6.2607 to 5.7299 seconds**. Output duration changed from 5.10 to 5.02 seconds because AR generation changed, so both duration and latency are reported. The immediate **RTF < 1 goal remains open**. Owner listening approval applies to this Korean reference/text/seed, not all voices or languages. See [raw L measurements](results/n1_student_l_20261112/).

The head-major Torch SDPA replacement, temporal tap tiling and explicit im2col/SGEMM did not improve total target-server latency and were not adopted. A 640-width decoder was rejected in listening and was not sent to N1. A calibrated 672-width decoder was also rejected in Korean listening after 200 CPU updates; it was not sent to N1. A separate additional200-update recovery completed with log-mel0.186844, but has not received new listening approval. The global-width path is suspended for the pair-only candidate below. P, an owner-approved Korean runtime experiment, keeps the same K weight files and L AR runtime but uses FP16 inputs, derived coefficients and accumulation only for the large FFT spectral products; transforms and the other acoustic operations remain FP32. Its first paired actual-N1 test reached L **1.16173** versus P **1.13344** RTF (5.8319 to 5.6899 seconds for the same 5.02-second output). Decoder time improved only 0.83%, while unchanged AR time varied by 7.4%, and sampled process-tree peak RSS rose from 5.059 to 6.371 GB. The apparent 2.44% total gain is not yet a stable speed conclusion; the target remains unmet. The [raw P results](results/n1_student_p_20261112/) and memory cost are retained. Important approved student files are separately backed up in a private Hugging Face repository with immutable revisions and fresh-download hash verification.

R follows a separate, conservative pair-bottleneck design. It preserves K12 branches, AR18 depth, all outer decoder widths/upsampling/residual boundaries and the final C24 stage. Only two C48/K11 Conv pairs use real **48→36→48** internal channels, initialized by JA/KO/EN training activation/consumer scores with omitted-input ridge reconstruction folded into Conv2 weights. Original V2 teacher supervision uses both original-AR and approved-L-AR inputs on identical latent/conditioning pairs (60training/12validation contexts), same-width pair/stage boundaries and waveform mel/STFT losses. CPU200-update validation log-mel improved **0.117717→0.103439**. The owner accepted R versus L on the Korean comparison and authorized branch push and actual N1 testing. JA/EN token/duration/integrity checks are synthesis checks only. The new source is on `codex/decoder-bottleneck`, not a default-model replacement. Its actual N1 comparison is pending; no speed success is inferred from parameter count or loss. See [R training and approval](results/student_pilot/approved_r.json) and [verified private backup](results/student_pilot/hf_approved_r_release.json).

## Ongoing RTF 0.7 research

An earlier local ARM64 quota-proxy experiment reached warmed median RTF **0.849** (this is not an actual-N1 result) on the first Japanese sentence (three measured runs). The 0.7 target remains open. These experimental options are not enabled by default and have not received a new listening approval. See [the checkpoint and next steps](docs/RTF07_CHECKPOINT.md) and [raw experiment reports](results/rtf07_checkpoint/). The final NEON input-packing edit is not included in that measured performance claim and requires a fresh rebuild/validation.

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
- Verify in-memory AR and acoustic parameter hashes and checkpoint hashes before and after inference. In this unchanged-model engine path, no training, pruning, lossy conversion of model weights, or checkpoint rewriting was performed. The separate student and approximate-runtime research described above has different arithmetic and quality scope. Floating-point evaluation order can still change the waveform slightly.

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

## ARM experimental benchmark

See [ARM_BENCHMARK.md](ARM_BENCHMARK.md) for the opt-in fast path, quality caveats, host requirements, and manual GitHub Actions measurements.

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
