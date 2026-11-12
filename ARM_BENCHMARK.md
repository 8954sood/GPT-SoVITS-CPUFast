# ARM64 experimental RTF benchmark

The manual **ARM RTF benchmark** workflow builds the FP32 native runtime on the target host, runs native and tap-kernel checks, then measures a baseline and the small-channel tap candidate on two Japanese sentences in serial. Each uses the same reference, seed and model checkpoints. JSON artifacts contain individual warmup and measured runs, guard results, CPU quota, timings and parameter audits. Generated WAV files for the four runs are also uploaded for listening; the input reference and model files are never uploaded. Model loading is excluded from RTF, and output includes 0.3 seconds of appended silence.

The FP32 engine uses AR prefetch distance 256 by default, based on two actual-server comparisons with PCM-exact outputs; pass `--ar-prefetch 0` to disable it. This workflow uses the FP32 path with value-preserving AR weight caches; it does not select the separate experimental W8A8/FP16 path. Checkpoints and authoritative AR/acoustic parameter hashes are checked before and after each run. Numerical rounding can differ and the generated audio still needs listening review. The default is the configured **actual 2-vCPU ARM server**, not the local 0.7426-CPU proxy; never compare their raw RTFs as if they were the same environment. The workflow is manually dispatched only.

## Configure a target host

The workflow runs on a GitHub-hosted runner and connects over SSH to the ARM server; the runner itself does not need to be ARM. It requires a prepared, trusted host with Docker (direct access or passwordless `sudo -n docker`), at least 8 GB available to Docker, and the following private assets. Do not commit the SSH key, model assets, or reference audio.

```
$HOME/gpt-sovits-benchmark/
  reference_ja_4s.wav
  project/models/s1v3.ckpt
  project/models/chinese-hubert-base/
  project/.venv/lib/python3.12/site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11/
  cpufast-cefae22/assets/pretrained_models/v2Pro/s2Gv2ProPlus.pth
  cpufast-cefae22/assets/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt
  cpufast-cefae22/assets/pretrained_models/fast_langdetect/lid.176.bin
```

The default image `cpufast-n1:cefae22` is a **locally prepared image**, not published by this repository. An alternate host needs an equivalent Linux ARM64 image with Python 3.12, PyTorch 2.8.0, the project inference dependencies, OpenMP, and a C++ compiler capable of building PyTorch extensions. The image must run `python` by default and make the dependencies available to a non-root user. The mounted Open JTalk dictionary and language identification cache must be readable. Model weights and the reference must match to compare results. This workflow does not download or provision models or the image.

In repository **Settings → Secrets and variables → Actions**, configure these repository secrets for the target: `ARM_SSH_HOST`, `ARM_SSH_USER`, `ARM_SSH_PRIVATE_KEY`, `ARM_SSH_PASSPHRASE` (empty if unencrypted), and `ARM_SSH_KNOWN_HOSTS` (a verified known_hosts entry). Restrict repository write access: anyone allowed to change and run the workflow can use its secrets. To change hosts, update all relevant secrets, especially the verified host key.

Optional repository variables: `ARM_BENCH_ROOT` (absolute path to the layout above), `ARM_BENCH_IMAGE` (local image name), `ARM_BENCH_CPUS` (Docker CPU quota, default `2`), and `ARM_BENCH_MEMORY` (bytes, default `8000000000`). The experiment uses two inference threads even if a different CPU quota is selected; record that distinction when comparing hosts. Docker mounts the asset directory read-write for runtime caches, with the S2 and speaker-verification checkpoints individually mounted read-only; the project and reference are also mounted read-only.

## Run

Select `compare` for the two-sentence audio comparison , `diagnostics` for a separate operator profile and synthetic FP32 kernel probes, `attention` to compare the approved tap path with short-context native attention on two sentences, `small` for synthetic small-channel convolution probes, `scheduling` for an active/passive OpenMP wait comparison, `packed48` to compare the selected 48-channel convolution candidate against tap on two sentences, `fft` for synthetic FFT-size tuning, `fftcandidate` for an end-to-end FFT-size comparison and AR cache probe, `arcache` for full synthesis with FP32 AR coefficient caches, `prefetch` for model-sized AR cache traversal, `prefetchcandidate` for the selected full-synthesis prefetch comparison, `winograd` to probe three-tap decoder methods, `direct3` to compare a selected direct-convolution candidate in full synthesis, or `locality` to probe AR prefetch locality. Profiling runs after the measured synthesis and is excluded from RTF; synthetic timings are not end-to-end performance results.

Open **Actions → ARM RTF benchmark → Run workflow**, choose the trusted branch and warmup/sample counts. The summary reports median and individual RTF; download the JSON and WAV artifacts from the run for details and listening. A temporary checkout is removed from the server after the run. The default workload uses the Japanese reference and text specified in `scripts/run_arm_benchmark.sh`, with seed 42. Hardware load, CPU quota, memory, runtime image and thermal conditions can affect the result.

To run directly on an already prepared target from a checkout:

```sh
bash scripts/run_arm_benchmark.sh "$HOME/gpt-sovits-benchmark" cpufast-n1:cefae22 8000000000 2 2 3
```

The outputs are `ci-results/{baseline,tap}_{1,2}.json`, their guard reports and corresponding WAV files. The script rebuilds the FP32 native extension in that checkout.

`mode=fftstream` compares the independently verified FP32 tap/prefetch-256
baseline against an opt-in four-frequency streaming spectral GEMM. It retains
the same ACL FP32 kernel and complex reconstruction order, but writes the
complex inverse-FFT layout from bounded scratch storage rather than retaining
all three frequency-product planes. The job first checks bit-exact synthetic
products and convolution epilogues, then reports a synthetic probe and full
synthesis of both fixed sentences. This is a candidate, with no N1 performance
claim until the action completes; the default engine remains unchanged.

`mode=transforms` separately tests channel-parallel Winograd input/output and
FFT finishing (`--parallel-transforms`). Independent channels retain the same
FP32 arithmetic and output order; this is not branch-parallel synthesis.
The default remains serial transforms until target full-TTS repeats justify a
change. Native tests compare serial/parallel outputs exactly, including
non-contiguous residual channel rows.

`mode=inference` compares the same original FP32 profile under its existing
`no_grad` path and the benchmark's explicit `torch.inference_mode()` context.
Model loading and installation remain outside that context, preserving normal
parameter version counters and the cache-validity checks. Both variants use
fresh processes, with every warmup/sample recorded. Local checks establish
correctness only; any performance claim must use the target action.

`mode=runtime` batches an independent original-FP32 baseline, parallel transforms,
inference mode, and their combination with FFT streaming on both fixed
sentences. The non-baseline order is reversed for the second sentence. Use
three excluded warmups and five retained samples for confirmation. Every
candidate is compared to that sentence's baseline WAV, and the combined RTF is
measured directly. Individual improvements are not added together.
