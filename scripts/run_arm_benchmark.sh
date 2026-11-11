#!/usr/bin/env bash
# Run in a private, disposable source checkout on the target Linux ARM64 host.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench_root=${1:-"$HOME/gpt-sovits-benchmark"}
image=${2:-cpufast-n1:cefae22}
memory=${3:-8000000000}
cpus=${4:-2}
warmups=${5:-2}
samples=${6:-3}
project="$bench_root/project"
assets="$bench_root/cpufast-cefae22/assets/pretrained_models"
reference="$bench_root/reference_ja_4s.wav"
for path in "$project/models/s1v3.ckpt" "$project/models/chinese-hubert-base" "$assets/v2Pro/s2Gv2ProPlus.pth" "$assets/sv/pretrained_eres2netv2w24s4ep4.ckpt" "$reference"; do
  if [[ ! -e "$path" ]]; then printf 'Required benchmark asset missing: %s\n' "$path" >&2; exit 1; fi
done
if [[ "$(uname -m)" != aarch64 ]]; then echo 'Target must be Linux ARM64.' >&2; exit 1; fi
if docker info >/dev/null 2>&1; then docker_cmd=(docker); else docker_cmd=(sudo -n docker); fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null
mkdir -p "$repo_dir/ci-results"
docker_args=(run --rm --user "$(id -u):$(id -g)" --platform linux/arm64 --cpus "$cpus" --memory "$memory" --memory-swap "$memory"
  --mount "type=bind,src=$repo_dir,dst=/repo"
  --mount "type=bind,src=$assets,dst=/repo/GPT_SoVITS/pretrained_models"
  --mount "type=bind,src=$project,dst=/project,readonly"
  --mount "type=bind,src=$reference,dst=/reference.wav,readonly"
  -e PYTHONPATH=/repo:/repo/GPT_SoVITS
  -e OPEN_JTALK_DICT_DIR=/project/.venv/lib/python3.12/site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11
  -e GPT_SOVITS_DATA_DIR=/repo -e OMP_WAIT_POLICY=ACTIVE -e GOMP_SPINCOUNT=100000
  -e CPUFAST_AR_LIBRARY=/repo/libcpufast_ar_torch.so -e CPUFAST_FAST_LIBRARY=/repo/libcpufast_fast.so
  "$image")
run_container() { "${docker_cmd[@]}" "${docker_args[@]}" "$@"; }
run_container python build_cpufast_ar.py --output /repo/libcpufast_ar_torch.so
run_container python build_cpufast_fast.py --output /repo/libcpufast_fast.so
run_container python -m unittest discover -s tests -p test_fast_operators.py -v
run_container python benchmark_cpufast.py --engine --fast-experimental --ar-threads 2 --threads 2 \
  --s1 /project/models/s1v3.ckpt --s2 /repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth \
  --hubert /project/models/chinese-hubert-base --reference /reference.wav \
  --prompt-text 'スマートフォンからフィーチャーフォンまで、マルチデバイスに対応。' \
  --text '今日は穏やかな天気ですね。少し休んでから、また一緒に続きを進めましょう。' \
  --output /repo/ci-results/benchmark.json --warmups "$warmups" --samples "$samples" --no-save-audio
