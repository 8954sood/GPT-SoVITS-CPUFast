# ARM64 experimental RTF benchmark

The manual **ARM RTF benchmark** workflow builds the FP32 native runtime on the target host, runs native and tap-kernel checks, then measures a baseline and the small-channel tap candidate on two Japanese sentences in serial. Each uses the same reference, seed and model checkpoints. JSON artifacts contain individual warmup and measured runs, guard results, CPU quota, timings and parameter audits. Generated WAV files for the four runs are also uploaded for listening; the input reference and model files are never uploaded. Model loading is excluded from RTF, and output includes 0.3 seconds of appended silence.

This workflow uses the FP32 path with value-preserving AR weight caches; it does not select the separate experimental W8A8/FP16 path. Checkpoints and authoritative AR/acoustic parameter hashes are checked before and after each run. Numerical rounding can differ and the generated audio still needs listening review. The default is the configured **actual 2-vCPU ARM server**, not the local 0.7426-CPU proxy; never compare their raw RTFs as if they were the same environment. The workflow is manually dispatched only.

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

Select `compare` for the two-sentence audio comparison , `diagnostics` for a separate operator profile and synthetic FP32 kernel probes, or `attention` to compare the approved tap path with short-context native attention on two sentences. Profiling runs after the measured synthesis and is excluded from RTF; synthetic timings are not end-to-end performance results.

Open **Actions → ARM RTF benchmark → Run workflow**, choose the trusted branch and warmup/sample counts. The summary reports median and individual RTF; download the JSON and WAV artifacts from the run for details and listening. A temporary checkout is removed from the server after the run. The default workload uses the Japanese reference and text specified in `scripts/run_arm_benchmark.sh`, with seed 42. Hardware load, CPU quota, memory, runtime image and thermal conditions can affect the result.

To run directly on an already prepared target from a checkout:

```sh
bash scripts/run_arm_benchmark.sh "$HOME/gpt-sovits-benchmark" cpufast-n1:cefae22 8000000000 2 2 3
```

The outputs are `ci-results/{baseline,tap}_{1,2}.json`, their guard reports and corresponding WAV files. The script rebuilds the FP32 native extension in that checkout.
