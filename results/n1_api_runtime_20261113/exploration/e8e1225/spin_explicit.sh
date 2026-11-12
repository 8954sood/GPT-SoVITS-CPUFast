#!/usr/bin/env bash
set -euo pipefail
root=/home/ubuntu/gpt-sovits-benchmark
repo=$root/ssh-api-runtime-e8e1225-20261113
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

 -e CPUFAST_AR_LIBRARY=/repo/libcpufast_ar_torch.so -e CPUFAST_FAST_LIBRARY=/repo/libcpufast_fast.so
 cpufast-n1:cefae22)
for mode in active100000 active1000 active10000 passive; do
 OMP_WAIT_POLICY=ACTIVE
 GOMP_SPINCOUNT=${mode#active}
 if [[ $mode == passive ]]; then OMP_WAIT_POLICY=PASSIVE; GOMP_SPINCOUNT=0; fi
 sudo -n docker "${args[@]:0:${#args[@]}-1}" -e "OMP_WAIT_POLICY=$OMP_WAIT_POLICY" -e "GOMP_SPINCOUNT=$GOMP_SPINCOUNT" "${args[-1]}" python -c "import os,json; expected={'OMP_WAIT_POLICY':'$OMP_WAIT_POLICY','GOMP_SPINCOUNT':'$GOMP_SPINCOUNT'}; assert all(os.environ.get(k)==v for k,v in expected.items()); print(json.dumps({k:os.environ.get(k) for k in ['OMP_WAIT_POLICY','GOMP_SPINCOUNT','LD_PRELOAD','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}))" > "$repo/ci-results/explicit_${mode}_environment.json"
 sudo -n docker "${args[@]:0:${#args[@]}-1}" -e "OMP_WAIT_POLICY=$OMP_WAIT_POLICY" -e "GOMP_SPINCOUNT=$GOMP_SPINCOUNT" "${args[-1]}" python resource_guard.py --report "/repo/ci-results/explicit_${mode}_guard.json" -- python scripts/benchmark_api_runtime.py --manifest /student/manifest.json --s2 /repo/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth --hubert /project/models/chinese-hubert-base --reference /reference.wav --output "/repo/ci-results/explicit_${mode}.json" --gc-policy default --warmups 2 --runs 3
done
