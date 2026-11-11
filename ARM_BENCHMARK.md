# ARM64 experimental RTF benchmark

The manual **ARM RTF benchmark** workflow builds the native kernels on the target host, checks their numerical behavior, warms the model, and measures non-streaming file-generation inference. The JSON artifact contains per-run timings, stage timings, resource limits, software versions, input and model hashes, and a parameter audit. Model loading is excluded; output duration includes the 0.3-second trailing silence. No generated audio is saved by this workflow.

The opt-in `--fast-experimental` path uses approximate W8A8 autoregressive linear layers and FP16 acoustic matrix multiplication. It requires ARM64 Linux with `asimddp` and `asimdhp`; it does not change the default inference path. Quality and generated tokens can differ, so evaluate the intended voices and texts before using it in production.

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

Optional repository variables: `ARM_BENCH_ROOT` (absolute path to the layout above), `ARM_BENCH_IMAGE` (local image name), `ARM_BENCH_CPUS` (Docker CPU quota, default `2`), and `ARM_BENCH_MEMORY` (bytes, default `8000000000`). The experiment uses two inference threads even if a different CPU quota is selected; record that distinction when comparing hosts. Docker mounts the asset directory read-write because runtime tools can populate caches; the project and reference are mounted read-only.

## Run

Open **Actions → ARM RTF benchmark → Run workflow**, choose the trusted branch and warmup/sample counts. The summary reports median and individual RTF; download the JSON artifact from the run for details. A temporary checkout is removed from the server after the run. The default workload uses the Japanese reference and text specified in `scripts/run_arm_benchmark.sh`, with seed 42. Hardware load, CPU quota, memory, runtime image and thermal conditions can affect the result.

To run directly on an already prepared target from a checkout:

```sh
bash scripts/run_arm_benchmark.sh "$HOME/gpt-sovits-benchmark" cpufast-n1:cefae22 8000000000 2 2 3
```

The output is `ci-results/benchmark.json` in that checkout. The script rebuilds native extensions there.
