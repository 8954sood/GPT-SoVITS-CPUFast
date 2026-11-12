#!/usr/bin/env bash
set -euo pipefail
root=/home/ubuntu/gpt-sovits-benchmark
repo=$root/ssh-api-runtime-07a4203-20261113
canonical=$root/ssh-head-integrated-c356245-20261112
manifest_sha=696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865
exec 9>"$root/.cpufast-arm-benchmark.lock"
flock -n 9 || exit 1
cp "$canonical"/libcpufast_{ar_torch,fast}.so* "$repo/"
mkdir -p "$repo/.experiments" "$repo/ci-results"
cp -a "$canonical/.experiments/nltk_data" "$repo/.experiments/"
args=(run --rm --user "$(id -u):$(id -g)" --platform linux/arm64 --cpus 2 --memory 8000000000 --memory-swap 8000000000
 --mount "type=bind,src=$repo,dst=/repo"
 --mount "type=bind,src=$root/cpufast-cefae22/assets/pretrained_models,dst=/repo/GPT_SoVITS/pretrained_models"
 --mount "type=bind,src=$root/project,dst=/project,readonly"
 --mount "type=bind,src=$root/reference_ja_4s.wav,dst=/reference.wav,readonly"
 --mount "type=bind,src=$root/student-assets/$manifest_sha,dst=/student,readonly"
 --mount "type=bind,src=$root/ssh-api-http-b1e1838-20261113/.experiments/server-python,dst=/server-python,readonly"
 -e PYTHONPATH=/server-python:/repo:/repo/GPT_SoVITS
 -e OPEN_JTALK_DICT_DIR=/project/.venv/lib/python3.12/site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11
 -e GPT_SOVITS_DATA_DIR=/repo -e NLTK_DATA=/repo/.experiments/nltk_data
 -e OMP_WAIT_POLICY=ACTIVE -e GOMP_SPINCOUNT=100000
 -e CPUFAST_AR_LIBRARY=/repo/libcpufast_ar_torch.so -e CPUFAST_FAST_LIBRARY=/repo/libcpufast_fast.so
 cpufast-n1:cefae22)
sudo -n docker "${args[@]}" python resource_guard.py --report /repo/ci-results/scripted_ar_guard.json -- python scripts/benchmark_api_runtime.py --manifest /student/manifest.json --s2 /repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth --hubert /project/models/chinese-hubert-base --reference /reference.wav --output /repo/ci-results/scripted_ar.json --gc-policy default --warmups 3 --runs 3 --script-ar --expected-wait-policy ACTIVE --expected-spincount 100000 --expected-preload /build/native/libgsv_mimalloc.so
