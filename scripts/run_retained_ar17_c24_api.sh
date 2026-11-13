#!/usr/bin/env bash
# Start or verify the single approved runtime retained after server cleanup.
set -euo pipefail
root=${CPUFAST_BENCH_ROOT:-/home/ubuntu/gpt-sovits-benchmark}
mode=${1:-serve}
sha=696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865
repo=$root/current-runtime
assets=$root/runtime-assets
args=(run --rm --platform linux/arm64 --user "$(id -u):$(id -g)"
  --cpus 2 --memory 8000000000 --memory-swap 8000000000 --workdir /repo
  --mount "type=bind,src=$repo,dst=/repo"
  --mount "type=bind,src=$assets/pretrained_models,dst=/repo/GPT_SoVITS/pretrained_models,readonly"
  --mount "type=bind,src=$assets/chinese-hubert-base,dst=/hubert,readonly"
  --mount "type=bind,src=$assets/open_jtalk_dic_utf_8-1.11,dst=/open-jtalk,readonly"
  --mount "type=bind,src=$assets/server-python,dst=/server-python,readonly"
  --mount "type=bind,src=$root/student-assets/$sha,dst=/student,readonly"
  --mount "type=bind,src=$root/reference_ja_4s.wav,dst=/reference.wav,readonly"
  -e PYTHONPATH=/server-python:/repo:/repo/GPT_SoVITS
  -e OPEN_JTALK_DICT_DIR=/open-jtalk -e GPT_SOVITS_DATA_DIR=/repo
  -e NLTK_DATA=/repo/.experiments/nltk_data
  -e OMP_WAIT_POLICY=ACTIVE -e GOMP_SPINCOUNT=100000
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2
  -e CPUFAST_AR_LIBRARY=/repo/libcpufast_ar_torch.so
  -e CPUFAST_FAST_LIBRARY=/repo/libcpufast_fast.so)
case "$mode" in
  serve)
    args+=(--name cpufast-ar17-c24-api -p 127.0.0.1:9880:9880 cpufast-n1:cefae22)
    command=(python server.py --host 0.0.0.0 --port 9880 --data-dir /repo/data/styles
      --manifest /student/manifest.json --manifest-sha256 "$sha"
      --s2 /repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth --hubert /hubert)
    ;;
  verify)
    mkdir -p "$repo/cleanup-validation"
    args+=(--name cpufast-ar17-c24-cleanup-verify cpufast-n1:cefae22)
    command=(python resource_guard.py --report /repo/cleanup-validation/resource_guard.json --
      python scripts/benchmark_api_client_runtime.py --manifest /student/manifest.json
      --s2 /repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth --hubert /hubert
      --reference /reference.wav --output /repo/cleanup-validation/http.json --warmups 0 --runs 1)
    ;;
  *) printf 'Usage: %s [serve|verify]\n' "$0" >&2; exit 2 ;;
esac
exec sudo -n docker "${args[@]}" "${command[@]}"
