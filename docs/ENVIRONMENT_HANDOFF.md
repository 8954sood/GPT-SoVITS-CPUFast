# GPT SoVITS V2ProPlus ARM CPU 최적화 인수인계

## 2026-11-12 직접 SSH 재개 후 현재 상태

사용자가 이 환경에서 직접 SSH 작업과 지속적인 최적화를 다시 요청했다.
이 절이 아래의 이전 일시정지/미측정 상태보다 우선한다. Main은 모델·CPU 학습·
품질을, `cpu_performance` 서브 에이전트는 추론·native·서버 실측을 맡고,
서버 작업은 하나의 직렬 큐로 실행한다. 현재 코드/모델 작업은
`codex/ssh-resume-20261112`, 측정 원래 소스는 worker의
`codex/ssh-v-n1-20261112`에 보존한다. 자세한 재개 기록은
[`SSH_RESUME_20261112.md`](SSH_RESUME_20261112.md)와
[`TARGET_RESEARCH.md`](TARGET_RESEARCH.md)를 따른다.

- 잘못 기록됐던 V 청취의 AR24/AR18 불일치를 수정하고 실제 AR18+원래 V400
  음성을 새로 생성해 사용자 승인을 받았다. 그 비교의 실제 N1 중앙 RTF는
  L1.160668 / AR18+V1.059591. 원자료는 `results/n1_student_v_20261112/`.
- 사용자가 새 AR17 QAT200+동일 V400 음질도 승인했다. 별도5쌍에서 AR18/AR17
  중앙 RTF1.057507/1.012739, AR/token16.3397/14.8897ms. 발화 길이가
  5.02/5.62s라 절대 생성 시간은5.309/5.692s다. 원자료는
  `results/n1_ar17_v400_20261112/`. 이 결과까지는 RTF1 미만이 아니다.
- 승인 AR17 실행 manifest SHA는
  `e30cdb7521a7e58053c554a2d78bf6f4d187d856b13083ff2d9e15987dc85bbb`.
  승인 범위는 같은 참조음/한국어 문장/seed뿐이다. JA/EN 전체 품질이나 원본
  교사와의 동등성을 주장하지 않는다. 원래 V400과 recovery100은 다른 모델이다.
- 출력 동일 attention value scheduling의5쌍 비교는 중앙RTF1.016054/0.992652,
  후보 범위0.990021~1.008962(4/5가1미만)다. 기본native경로에 동일루프를 채택한
  독립5fresh 재확인은 중앙1.010217, 범위0.991603~1.017123이다. 승인PCM은
  모두 같지만 안정적인1미만 재현은 아직 미달이다. 원자료는
  `results/n1_head_v32_20261112/{full_pcm,integrated}/`에 보존한다.
- QK4의 전체5쌍은 baseline/candidate 중앙RTF0.996747/0.996269로 사실상
  같았고 AR시간도 일관되게 줄지 않아 채택하지 않았다. 원자료와 재현용
  workflow `studentheadqk4`만 보존한다. worker는 승인AR17+V400와 채택된
  value scheduling을 고정하고 FFN2-only row8 INT8 후보를 다음으로 시험한다.
- 사용자는 새 AR16의 음질을 거부해 해당 모델을 보류했고 성능 측정은 없다.
  V recovery200은 같은 구조에서 CPU 추가학습을 마쳤고 고정입력 logmel 오차가
  원래V보다2.35% 낮지만 청취/RTF 미승인 후보다. 현재서버의 V400을 바꾸지 않는다.
- Main의 AR16, history 보강, inactive-FFN slice는 별도 미승인 연구 후보다.
  승인 모델 파일을 덮어쓰거나 이들 후보를 현재 서버 비교에 섞지 않는다.

아래는 재개 전 인계 당시 기록이며 위의 현재 상태와 구분한다.

기준일: 2026-11-12. 이 환경의 최적화 작업은 사용자 요청으로 일시정지한다. **V 실제 N1 시험은 여기서 실행하지 않고 새 환경으로 넘긴다.** 목표는 모델 파일 축소 자체가 아니라 실제 배포 서버의 RTF를 낮추는 것이다. 최근 목표는 RTF 0.8대이며, 먼저 실제 N1에서 1 미만을 달성해야 한다. 0.8대 달성을 주장할 결과는 아직 없다.

## 1. 지금 이어받아야 할 상태

| 항목 | 상태와 판단 |
| --- | --- |
| 성능 기준 L | 승인된 K 계열 AR18/디코더12 + W8A8 AR decode, native FP32 attention, FP32 acoustics. 최근 실제 N1 RTF 약 1.139. 계속 비교 기준으로 유지 |
| R | 한국어 청취 승인. L/R 실제 N1 1.145699/1.141252. 총시간 차이 0.39%로 작아 성능 기준으로 채택하지 않음 |
| U | 한국어 청취 승인. L/U 실제 N1 1.1386905/1.1377801. 총시간 0.08% 차이, 디코더는 1.38% 느림. 채택하지 않음 |
| V | R에서 큰 FFT 내부 연산 폭을 줄이고 CPU 400회 후학습 완료. 사용자: “품질 저하 조금 있긴해 근데 점차 모든 모델들에서 조금씩 낮아지고 있는거라 이정도는 감안해야할듯”. 약간의 저하를 감안한 평가이며 원본 동급 품질로 기록하면 안 됨. **실제 N1 미실행, 새 환경에서 수행** |
| W | 동일 K/L 가중치, prefill FFN만 genuine multirow INT8 GEMM. 한국어 semantic/PCM은 L과 정확히 같음. JA/EN 출력은 달라 별도 청취 필요. 기존 run `43195474472`가 인계 전 이미 실행되어 마지막 확인은 in_progress. source `b88177079aff850861e5ab3848e190032e369b41`. 재실행 없이 먼저 회수. 상세는 `W_PREFILL_HANDOFF.md` |
| QAT | `distill_ar.py --qat-decode` 선택 기능과 테스트만 준비. 실제 QAT 학습/청취/RTF 검증 없음 |

기존 `RTF07_CHECKPOINT.md`의 0.8~0.9 수치는 **로컬 quota 근사치**다. 이 문서의 실제 N1 1.14와 혼동하면 안 된다. V5의 quota 7426/10000 보정 역시 다른 모델의 과거 throughput 근사이며 실제 N1 시험에는 적용하지 않는다.

## 2. 코드와 자료 위치

원격 저장소: https://github.com/8954sood/GPT-SoVITS-CPUFast

- 새 환경 시작 브랜치: `codex/optimization-handoff`. 아래 인계 소스와 V/W/QAT 준비 코드를 모은다. 실제 W 시험의 source SHA는 W 운영 문서의 별도 고정값을 따른다.
- 이전 공개 U 소스: `codex/decoder-u-optimization`의 `5a486bc`, 실제 U 결과: `codex/cpu-u-performance`의 `422e191`.
- 원래 로컬 연구 브랜치: `codex/ar-ffn-optimization`. 아직 승인되지 않은 과거 연구도 포함하므로 전체를 main에 무작정 병합하지 않는다. 별도 로컬 Git bundle로 보존한다.
- 주 저장소의 `results/student_pilot/`, `results/n1_student_u_20261112/`, `results/rtf07_v_fft_correctness/`에 승인·실험·정합성 증거가 있다.
- V5 참고 저장소의 작업 중 변경은 source bundle + patch + untracked source archive로 별도 보존한다. CPUFast 코드로 덮어쓰지 않는다.

기존 Mac 경로:

```text
/Users/byungjun/Desktop/python/GPT-SoVITS-CPUFast
/Users/byungjun/Desktop/python/gpt-sovits-v5-optimization
/Users/byungjun/Desktop/python/audio8-tts-optimization
/Users/byungjun/Desktop/python/tts/handoff-20261112
```

인계 폴더의 `TRANSFER_RECEIPTS.json`에 전송 파일 크기와 SHA256을 기록한다. 토큰·SSH 키는 포함하지 않는다.

| 전송 파일 | 새 환경에서의 용도 |
| --- | --- |
| `cpufast-private-research.tar.gz` | V best/spec/history, 원본 AR 교사 문맥36개, decoder A/B 문맥72개, 교사 decoder state, AR teacher features, 생성 음성, L/V/W 비교 음성. Git checkout 루트에 복원 |
| `cpufast-original-assets.tar.gz` | 원본 S1/S2, HuBERT, SV, 고정 참조음, 언어 감지·일본어·NLTK 자료. 별도 `local-assets` 디렉터리에 복원 |
| `cpufast-research-source.bundle` | 정지 시점 로컬 연구 브랜치의 전체 Git 이력. 공개 인계 브랜치에 없는 연구 확인용 |
| `v5-source.bundle` | V5 main의 커밋 이력 |
| `v5-working-tree.patch` | V5 tracked 파일의 미커밋 변경 |
| `v5-untracked-source.tar.gz` | V5의 미추적 소스와 작은 결과 JSON |

전송 묶음에는 개인 참조음과 생성 음성 문맥이 있으므로 **로컬 전송용으로 보관한다. Git/공개 데이터셋에 올리지 않는다.** 승인된 중요 모델 백업용 HF 저장소는 별도로 아래에 명시한다. 새 파일을 계속 중복 보관하지 말고 다른 환경으로 복사한 뒤 전송본 필요 여부를 정리한다.

## 3. 새 환경의 첫 실행

### 기존 원격 서버를 이용한 성능 작업

새 컴퓨터에 모델이나 Docker 이미지를 먼저 복사하지 않아도 기존 Actions 서버 시험은 이어갈 수 있다. GitHub 저장소 권한과 CLI 인증을 준비한다.

```bash
git clone --branch codex/optimization-handoff https://github.com/8954sood/GPT-SoVITS-CPUFast.git
cd GPT-SoVITS-CPUFast
gh auth login
gh run list --repo 8954sood/GPT-SoVITS-CPUFast --workflow arm-rtf.yml --limit 5 --json databaseId,status,conclusion,headSha,url
```

W 실행이 이미 있다면 먼저 해당 run을 확인하고 결과를 내려받는다. 같은 실험을 중복 dispatch하지 않는다. 실행은 반드시 `--repo 8954sood/GPT-SoVITS-CPUFast`를 지정한다. upstream 저장소를 기본 대상으로 잘못 선택한 전례가 있다. GitHub CLI의 [`workflow run`](https://cli.github.com/manual/gh_workflow_run), [`run list`](https://cli.github.com/manual/gh_run_list), [`run download`](https://cli.github.com/manual/gh_run_download) 문서를 기준으로 한다.

```bash
gh run download RUN_ID --repo 8954sood/GPT-SoVITS-CPUFast --dir local-results/RUN_ID
```

### 로컬 CPU 후학습 자료 복원

```bash
# 현재 위치는 새 CPUFast checkout 루트다. 기존 파일과 겹치면 먼저 비교한다.
python3 scripts/package_research_handoff.py --output /전송경로/cpufast-private-research.tar.gz --verify
tar -xzf /전송경로/cpufast-private-research.tar.gz
mkdir -p ../cpufast-local-assets
python3 scripts/package_research_handoff.py --output /전송경로/cpufast-original-assets.tar.gz --verify
tar -xzf /전송경로/cpufast-original-assets.tar.gz -C ../cpufast-local-assets
```

Python 3.12를 사용한다. 성공한 로컬 학습 환경은 Torch/torchaudio 2.8.0, numpy 1.26.4, x-transformers 1.43.4, transformers 4.56.2, torchmetrics 1.5.0, librosa 0.10.2였다. 나머지 관측 버전은 `results/student_pilot/handoff_host_versions.json`에 있다. 이는 설치 lock의 완전 재현을 보장하는 파일이 아니라 실제 사용 버전 기록이다. Linux는 Torch CPU wheel을 설치하고 CUDA/MPS로 학습을 돌리지 않는다. Mac에서도 `.cpu()`로 실행한다.

```bash
python3.12 -m venv .venv-research
.venv-research/bin/python -m pip install 'torch==2.8.0' 'torchaudio==2.8.0' 'numpy==1.26.4' 'x-transformers==1.43.4' 'transformers==4.56.2' 'torchmetrics==1.5.0' 'librosa==0.10.2' 'soundfile==0.14.0' 'einops==0.8.2' 'psutil==7.2.2' pytest
export PYTHONPATH="$PWD:$PWD/GPT_SoVITS"
.venv-research/bin/python -c 'from experiments.student.distill_decoder import decoder_from_teacher; from experiments.student.distill_ar import ARStudent; print("training imports OK")'
```

Linux CPU wheel의 설치 index는 해당 OS에 맞춰 지정한다. 전체 음성 전처리/추론은 프로젝트 의존성과 언어별 사전도 필요하다. 위 명령은 학습 코드의 시작점이며 새 OS에서 음성 추론까지 검증된 설치 명령으로 간주하면 안 된다. 로컬 native ARM 커널과 `.so`는 새 아키텍처에서 다시 빌드해야 한다.

예전 `.experiments/student_run.py`는 Mac 절대 경로 및 로컬 전용 이미지 `gpt-sovits-v5:cpu-arm64`에 의존한다. 이 이미지는 registry에 게시되지 않았다. 파일을 복사하거나 같은 이름으로 `docker pull`하는 것만으로 재현되지 않는다. 성능 판정은 기존 N1 Actions를 쓰고, 로컬 학습은 위의 host Python 경로로 분리한다. 로컬 음성 합성이 필요하면 새 자산 경로/의존성/ARM native build를 먼저 연결한다.

## 4. 승인 모델 복원

HF private repo: `babihoba/gpt-sovits-v2proplus-arm-students`. 토큰은 새 환경에서 비공개 입력/자격증명 저장소로 제공한다. 문서·Git·명령행 인수에 토큰 문자열을 넣지 않는다.

| 모델 | 고정 revision | manifest 경로 | manifest SHA256 |
| --- | --- | --- | --- |
| K (L의 가중치) | `475ac97f4a503ea9c6581333c8b936fd5c9f6dcd` | `approved-k-20261112/manifest.json` | `761a3dd80e40211d469823a2fffd8c3c5d153162790029bae12cf152cc7a17b4` |
| R | `d213cf026d492364950de5077b4c2631fc48e910` | `approved-r-20261112/manifest.json` | `dbb4a69d2b92120fce7bbb685c8c2f78590d5ca46abd5753b26088b93626c635` |
| U | `9287f847c7272ba2f45d0bd2a18512abec93a6e1` | `approved-u-20261112/manifest.json` | `8d0329df24fa14eaf36e07b265c3111a592dec962870ec392fe0c58896832751` |

K의 AR SHA256은 `4051df84ee71e52eeb9858a53a060c4785eef912cf25a51051069ddd99041335`。decoder は `a2fe6478aae00a8a481209f32aba85eab97198cd30aca0ce1ff63d9ec31c3c1d`。L/Wはこの同じモデルファイルを使い、実行方式が異なる。

`scripts/download_student_assets.py` は環境変数 `HF_STUDENT_REPO`, `HF_STUDENT_REVISION`, `HF_STUDENT_MANIFEST_PATH`, `HF_STUDENT_MANIFEST_SHA256`, `HF_STUDENT_TOKEN` を受け取り、manifestと3ファイルを検証する。固定 revision と manifest SHAを省略しない。`.hf-cache`にもダウンロードコピーが作られるので検証後の重複に注意する。[Hugging Face公式の固定リビジョンダウンロード説明](https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/guides/download.md)も参照。

VはまだHF bundle化していない。**転送アーカイブの best.pt が正本**で、次の環境で必要なら private HFへ重要候補として登録する。既存モデルの原本を上書きしない。

V5教師の参考情報: private `babihoba/gpt-sovits-v5-turbo-arm-student-full` の12層 revision `e5b0511fde3e1f40e02cb1a69181b8b893c36d32`。元のV2モデルは変更していない。V5音声をV2のsemantic logits教師として取り違えない。V5は音声/conditioning素材として使用し、AR logits/decoder同一入力教師はV2原本から取る。

## 5. Vの次の作業

現時点で追加後学習やN1試験は開始しない。新環境で再開指示を受けてから以下を進める。

1. V best/spec と転送索引のハッシュを検証する。
2. 元の教注明瞭度を維持できているか、**原本A・基準L・V**を同じ参照/テキスト/seedで比較する。ユーザーはVの若干の低下を認識している。さらに縮める前に累積劣化を確認する。
3. サーバー用bundleを作る際、manifestはK/R/Uの検証形式を踏襲し、LのARファイルを共有、V decoder/specだけ差し替える。specの `decoder_channel_keep` が null でも manifest の検証形式と整合させる。全 tensor の export 前後完全一致を確認する。
4. `load_manifest`、元S1/S2・参照音・学生パラメータhash、3言語 synthesis、semantic token/長さ/PCM を記録する。新たな音声変化はユーザー確認を挟む。
5. **V単独/L単独**を同じsource SHA・同じHF revisionで交互に実測する。初回からW/P/R/Uを追加合成しない。
6. 実測が速くなった場合だけ、独立した再試験と追加文/長文で再現性を確認する。

V設計: 外幅/upsampling/残差/枝数は保存。測定された15 FFT pairについて C384→240、C192→120または96、C96→60 の内部幅を採用し、RのC48→36 pairも保持。decoder params 30,708,456→22,811,196。既存FFTサイズheuristic込みの対象 spectral MAC削減は40.116%。**全体RTF40%削減を意味しない。** C384から240へ落ちるconv2はFFTサイズ128→256に変わる点も計算済み。

Vファイル:

```text
.experiments/student/budget40/best.pt
  SHA256 7d4c184c638b09880a7e61b088b999542598644d22166e84e32137c5649b9524
.experiments/student/budget40_init/student_spec.json
  SHA256 cb411065f3e69c2725791fc00272940589c309a67bd5122abcd26499cfde6a16
.experiments/student/budget40/history.json
.experiments/student/budget40_host.json
.experiments/student/listening_V/{ko_D,ja_D,en_D}.wav
results/student_pilot/budget40_grouped_validation.json
results/rtf07_v_fft_correctness/real_v_in_memory.json
```

60train/12validation、A(original AR)とB(approved L AR)双方のlatent/conditioningで元V2 decoderを教師にした。400 CPU更新、8threads、grad accumulation3、lr5e-6→5e-7、full context、log-mel/STFT、stage features、pair boundary loss。検証log-melは0.248019→0.162211、最終log-STFT0.379436。Native in-memory検証は3言語×5block×1/2threadの30条件を通過したが、これは速度や聴感の承認ではない。

追加学習の例（新しい出力先を使う。**再開依頼前には実行しない**）:

```bash
export PYTHONPATH="$PWD:$PWD/GPT_SoVITS"
.venv-research/bin/python experiments/student/distill_decoder.py \
  --s2 ../cpufast-local-assets/assets/v2Pro/s2Gv2ProPlus.pth \
  --spec .experiments/student/budget40_init/student_spec.json \
  --manifest .experiments/student/bottleneck_contexts72/manifest.json \
  --resume .experiments/student/budget40/best.pt \
  --output .experiments/student/budget40_next \
  --steps 400 --threads 8 --lr 0.000002 --grad-accum 3 \
  --lr-final-ratio 0.1 --validate-every 50 --full-context --log-mel
```

上のlr/追加400回は次の実験例で、今回実施したものではない。heldoutを層選択用に使わず、語尾・発音・こもりの改善をユーザーが判断する。decoder B corpusは `ar_distillation_allowed:false`。BのAR出力を原本AR24の教師logitsとして使わない。

## 6. 実際のN1実験規則

- Workflow: `.github/workflows/arm-rtf.yml`。GitHub Actionsから設定済みSSH先の実際のNeoverse N1で実行する。2vCPU、8,000,000,000B、swapなし。ローカルMac Dockerは正しさ/音声生成の確認用。
- 一度にremote jobは1本。concurrency `cpufast-arm-benchmark`、cancel-in-progress false。複数agentが同時にdispatchしない。
- 設定済みGitHub Secrets: `ARM_SSH_HOST`, `ARM_SSH_USER`, `ARM_SSH_PRIVATE_KEY`, `ARM_SSH_PASSPHRASE`, `ARM_SSH_KNOWN_HOSTS`, `HF_STUDENT_TOKEN`。新PCに中身を転記する必要はない。
- Vars: `ARM_BENCH_ROOT`, `ARM_BENCH_IMAGE`, `ARM_BENCH_MEMORY`, `ARM_BENCH_CPUS`。サーバー既存imageの既定値は `cpufast-n1:cefae22`。Git archiveされた source SHAから別checkoutを作り、native DSOをそのソースでビルドする。
- paired比較は5 fresh processesずつAB/BA、各3warmup除外+1sample。profileは別プロセスで測定に混ぜない。全sample・outlier・guard・元ファイルhash・paramhash・semantic/PCMhashを残す。
- RTF=実行wall秒/実際の出力音声秒。無音追加、音声長変更、時間の差し引きで数値を調整しない。wall/AR/decoder/CPU/RSS/token数/長さを同時に読む。
- 変えていない段階の揺れだけで全体が速く見える結果を採用しない。CPU profiler scope時間の合計をwallや削減余地に換算しない。

最新U実測: [run 43189203151](https://github.com/8954sood/GPT-SoVITS-CPUFast/actions/runs/43189203151)、source `2915db55e6fc8baba7d87d1e9f11d3c109b979f4`、`results/n1_student_u_20261112/`。

Lの固定韓国語: `오늘은 날씨가 참 좋네요. 잠깐 쉬었다가 함께 이야기를 이어 가요.`、seed42、118token、5.02秒、32kHz。
reference SHA256 `d88a56e042cecc419fceaef1bb73cbcf21d7e0f9122b2206a8f5045b8f577839`。
L semantic SHA256 `de58113b4925307206f189dd71945298f00c64b97cec964501f98b8f738d50d9`、PCM SHA256 `92a135e36f8a0a0fd111cfbc42b1075e0ae3a421401fcc823d509e67be7edd9f`。

## 7. 메인과 서브 에이전트 운영

**메인 + 실제 서브 에이전트1명**으로 시작한다. 역할 이름만 문서에 쓰고 실제로 생성하지 않는 운영은 하지 않는다. 새 채팅에서 “서브 에이전트 1명을 실제 생성해서 런타임/실제서버 검증을 맡겨”라고 명시한다. 이미 살아 있는 worker가 있으면 새로 늘리지 말고 재사용한다. 모델/추론 설정은 별도 요구가 없으면 부모 설정을 상속한다. 공식 [Codex subagents 안내](https://learn.chatgpt.com/docs/agent-configuration/subagents)에 따라 명시적으로 위임하고, 각 작업의 범위와 반환 자료를 지정한다.

| 담당 | 소유 범위 | 반환할 자료 |
| --- | --- | --- |
| 메인 | 모델/CPU 후학습/원본 대비 품질/자료·HF bundle/사용자 청취/최종 채택 | 정해진 checkpoint/spec/source hash, 음성 A/B, 품질 피드백, 실험 우선순위 |
| cpu_performance worker | native C++/실행 경로/빌드/워크플로/profile/실제 N1 | isolated branch와commit, 정합성 증거, 정확한 dispatch명령/run ID, raw results, stage별 해석 |

Git worktree를 분리하고 두 agent가 같은 파일을 동시에 수정하지 않는다. 공통 파일 변경은 소유자를 정한 뒤 전달하고, cherry-pick/선택적반영 후 다시 diff를 검토한다. GitHub Actions dispatch는 worker 한 명이 맡고, 메인은 원격 대기 중 다음 가설의 코드/자료/정합성 검증을 준비한다. 새 N1 결과가 오기 전에 여러 후보를 합쳐 원인 판별을 어렵게 만들지 않는다. 이 환경은 정지하므로 자동 다음 실험은 실행하지 않는다.

worker에 줄 최초 요청:

```text
docs/ENVIRONMENT_HANDOFF.md와 docs/W_PREFILL_HANDOFF.md를 읽어라.
네 담당은 native runtime, build, profile, GitHub Actions를 통한 실제 N1 검증이다.
먼저 기존 W run 상태와 source SHA를 확인하고 이미 실행된 시험을 중복하지 마라.
메인과 별도 worktree/브랜치를 사용하고, 원격실험 큐는 한 번에 1개만 소유하라.
V는 현재 best/spec를 고정한 뒤 메인이 bundle/품질 범위를 전달하면 L/V 단독 비교를 준비하라.
로컬 Docker RTF를 N1 성능으로 보고하지 마라.
구조/정밀도 변경은 정합성→실제 실행 음성→청취 확인→N1 실측 순서를 지켜라.
완료 시 source commit, manifest hashes, run ID, 모든 raw 결과 경로, stage별 분석을 반환하라.
```

## 8. 다음에 시도할 가치가 있는 가설

1. **최우선은 V의 실제 비용 확인**이다. 큰 FFT pair를 줄였는데 end-to-end decoder가 빨라지는지 확인해야 한다. 크기와MAC만 작아지고 real backend·FFT-size·packing비용이 늘어날 수 있다. U가 이 반례다.
2. W의 genuine prefill GEMM 효과를 확인한다. KO가 PCM-exact여도 prefill 비중이 작으면 전체 RTF 이득은 작다. JA/EN은 품질 재평가가 필요하다. W와V 결합은 각각의 단독 검증 이후다.
3. 실제 N1에서 남은 decoder의 FFT packing/transform/spectral GEMM/finish와 메모리 트래픽을 분리한다. U/R처럼 작은 pair만 줄이는 미세 최적화보다 큰 C384/C192 실측 hotspot을 우선한다. 형상별 microbench로 가설을 거른 뒤 full inference로 검증한다.
4. AR FFN 축소가 필요한 경우 준비된 QAT 경로로 train/runtime quantization 간격을 줄이는 가설을 검증한다. 현재 QAT는 float STE surrogate로 native attention/norm/int32 누산 전체와 bitwise 동일하지 않다. 정합성/음질/실속도 검증이 필요하며 학습 완료 모델은 없다.
5. V 음질 보완은 원본 teacher와의 mel/STFT/feature/pair-boundary 감독 및 끝부분 발음/억양 검증을 우선한다. 무조건 더 줄이거나 같은 단문 하나의 loss만 낮추지 않는다. 더 긴 문장/다른 참조음은 validation leakage 없이 추가한다.
6. V5의 후학습·원본 대비 mel 검증과 supertonic3/audio8의 성공/실패는 가설 참고로만 사용한다. 다른 모델의 커널·quota 속도나 GPU/FP16 이득을 N1 V2에 그대로 대입하지 않는다.

실패/미채택: 무학습 B/C/D 품질 저하, S FFN1536 품질 저하, N640/O672 외폭 축소 품질 저하. P FP16 FFT product는 한국어 승인됐지만 stage 흔들림과 RSS 증가 때문에 기본 채택하지 않았다. FP32 packed prefill은 AR가 더 느려져 채택하지 않았다. 같은 실험을 이름만 바꿔 재실행하기 전에 원래 보고서를 읽는다.

## 9. 용량 정리와 원본 보존

`results/student_pilot/model_cleanup_20261112.json`에 삭제한67개 파일의 정확한 hash/백업검증을 남겼다. 당시 CPUFast 6.09→1.55GiB, V5 8.61→5.49GiB로 줄였다. 이후 인계 archive를 만들었으므로 현재 디스크 사용량은 이 숫자와 다르다.

중요 K/R/U와V best, 원본교사, 현재 corpus는 유지한다. 삭제된 G/H2/H3/과거16층 모델은 HF고정백업을 사용한다. `.experiments/student/approved_k_20261112/ar.pt`가 현재AR정본이며 삭제된 옛 `h3_half` 경로를 참조하면 안 된다. 거절한 학생/optimizer/중복 캐시를 다시 대량 복원하지 않는다. 새 모델이 중요하면 private HF백업의 fresh-download SHA검증을 한 뒤 작업중 모델 여부를 확인하고 정리한다.

## 10. 새 채팅에 붙여넣을 재개 요청

```text
GPT-SoVITS-CPUFast ARM CPU 최적화를 이어서 진행해.
codex/optimization-handoff의 docs/ENVIRONMENT_HANDOFF.md와 docs/W_PREFILL_HANDOFF.md를 먼저 읽어.
인계 폴더의 TRANSFER_RECEIPTS.json을 검증하고 필요한 corpus/model만 복원해.
실제 서브 에이전트 1명을 생성해 native/runtime/Actions/N1 실측을 맡기고,
메인은 모델/CPU후학습/원본대비음질/청취승인/통합을 맡아.
기존 W run을 먼저 회수하고, V는 이전 환경에서 N1을 실행하지 않았으므로 현재 best/spec로 단독 L/V 검증을 준비해.
V는 약간의 품질 저하를 사용자가 인지한 상태다. 누적저하를 피하려고 원본A/L/V를 함께 비교해.
원본 모델은 보존하고 학생 후학습은 로컬CPU만 사용해. 모델 크기보다 실제 N1 RTF 감소가 목적이야.
실제 Neoverse N1 2vCPU/8GB/no-swap 서버의 RTF 1 미만, 이어서0.8대가 목표야.
Actions 대기 중에는 다음 가설/정합성검증을 준비하되 remote 실험은 직렬로 실행해.
출력/정밀도가 바뀌면 음성을 들려주고 내 품질 검증 후 실제 N1 시험을 진행해.
이번 문서 작성 후에는 이전 환경은 일시정지 상태로 유지해.
```

## 11. 이관 직전 검증

공개 인계 checkout에서 host tests **63 passed / 19 skipped**를 확인했다. ARM 전용 검사는 host에서 skip되므로 W의 별도 Linux ARM64 primitive/loader 증거와 V의 30조건 native 수치 검증 보고서를 함께 읽는다. 전송 research archive 206파일과 original-assets archive 29파일의 전체 SHA256 읽기 검증을 완료했다. V N1은 실행하지 않았다. 이미 진행 중이던 W 외에 새 remote 실험을 시작하지 않는다.
