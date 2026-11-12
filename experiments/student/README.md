# V2ProPlus CPU Student 실험

원본 체크포인트와 기본 실행 경로를 유지한 별도 연구 코드입니다. 학습 없이 축소한 B/C/D는 사용자가 음색·먹먹함·억양 저하로 모두 거부했습니다. 후학습 체크포인트도 별도 청취 승인 전에는 채택하지 않습니다.

배포 성능 기준은 GitHub Actions에 연결된 실제 Neoverse N1의 2 vCPU / 8,000,000,000 bytes / swap 없음입니다. 로컬 Mac CPU 학습과 로컬 Docker 음성 생성은 N1 RTF 근거가 아닙니다. CPU quota로 보정한 이전 프록시 결과와도 구분합니다.

## V5 실행 기록에서 가져온 판단

`gpt-sovits-v5-optimization`의 실제 학습 보고서, `training.json`, WORK_LOG를 읽었습니다. 요약과 원본 보고서 SHA256은 `results/student_pilot/v5_execution_lessons.json`에 있습니다.

- 폭 1024를 유지한 21레이어 LoRA 학습은 원본 교사를 고정하고 검증용 endpoint RMSE 0.0138442를 얻었으며, 별도 레퍼런스 한국어 청취 승인을 받았습니다.
- 16레이어 adapter만 학습한 경로는 충분히 회복하지 못했습니다. 이후 전체 파라미터를 lr 2e-5로 200회 학습한 후보는 새 레퍼런스 청취 승인을 받았습니다.
- 승인된 16레이어에서 12레이어로 단계적으로 줄인 뒤 CPU 200회 후학습한 후보도 청취 승인을 받았습니다. V5의 로컬 Docker RTF는 실제 N1 결과로 옮겨 쓰지 않습니다.
- 폭을 크게 줄인 초기 학생과 보코더 분기 축소는 실패했습니다. 2분기 보코더는 600회 학습 후에도 미채택입니다. 손실 감소만으로 품질 회복을 선언하지 않습니다.

V5의 CFM과 V2의 AR/VITS는 구조가 다릅니다. 동일한 손실이나 latent를 이식하지 않습니다. 두 프로젝트의 AR은 같은 s1v3 원본을 사용하므로 V5 AR을 더 강한 교사라고 부르지 않습니다. 현재 V5의 역할은 검증된 원본 교사 trajectory에서 생성한 음성 레퍼런스를 제공하는 것입니다. 토큰 로짓과 디코더 타깃은 호환되는 원본 V2ProPlus에서 수집합니다.

## 분석과 후보

`analyze_teacher.py`는 AR 레이어를 하나씩 제외한 teacher-forcing KL과 각 디코더 분기 출력의 상대 오차를 한·일·영에서 측정합니다. hook 출력은 이후 inplace 연산 전에 clone합니다. 원본 파라미터 digest가 실행 전후 동일한지 확인합니다.

원본 AR은 24레이어, 폭512, 16헤드, FFN2048입니다. 첫/끝 레이어를 보호하고 16·17·18·20을 제외한 AR20을 첫 파일럿으로 선택했습니다. 디코더는 5개 업샘플링 단계의 폭과 비율을 유지하며 중간 두 단계에서 분기 하나씩을 제외한 13분기 후보입니다. 마지막 단계는 유지합니다. 초기 3언어 분석 데이터는 후보 선택에 사용되었으므로 독립 최종 검증이라고 주장하지 않습니다.

- A: 원본 AR24 + 원본 디코더15
- B: AR20 + 원본 디코더15
- C: 원본 AR24 + 학생 디코더13
- D: AR20 + 학생 디코더13

학습 전과 후는 별도 출력 디렉터리를 사용합니다. 원본과 학생 체크포인트를 덮어쓰지 않습니다.

## 데이터와 첫 CPU 후학습

`prepare_v5_voices.py`는 기존 V5 trajectory의 전체 길이와 원본 vocoder 출력 길이를 비교합니다. V5 vocoder는 48kHz입니다. 일부 공개 화자 trajectory는 잘린 데이터여서 전체 문장 레퍼런스로 사용하지 않습니다. 기존 원본 녹음 두 개도 복사해 시험했으나 V2 교사 출력 길이가 짧은 문장에 대해 11.7~22.22초로 길어졌습니다. 이들은 청취 확인 전까지 학습에서 제외했으며 원본 수집 결과는 보존합니다.

첫 회복 파일럿은 확인된 V5 생성 레퍼런스 한 개, 한·일·영 각 학습1문장/검증1문장, 총 학습3/검증3 context로 제한합니다. 화자 일반화 증거가 아닙니다. 원본 청취 비교 레퍼런스는 학습용 V5 생성 문장과 다른 녹음입니다. 데이터와 음성은 `.experiments/` 아래에 보존하고 Git에 올리지 않습니다.

학습용 AR24를 원본과 비교한 로짓 최대 오차는 약 2.48e-5, RMSE 3.83e-6입니다. 이를 확인한 후 AR20의 Transformer와 predictor만 CPU FP32로 학습했습니다. teacher-forced KL(교사||학생), 온도2, 토큰 평균, lr2e-5, gradient norm1을 사용합니다. 고정된 입력 embedding/context와 교사 로짓을 사용하므로 자유 생성의 누적 오차는 별도 청취로 확인해야 합니다.

디코더는 원본 교사를 고정하고, 16 latent frame crop에서 다중 STFT 손실과 단계별 특징 손실로 CPU FP32 학습합니다. 원본 파형 타깃은 같은 crop에 대한 원본 V2 디코더 출력입니다. 작은 crop의 수치 개선이 전체 문장 품질을 보장하지 않습니다.

첫 파일럿은 AR50회, 디코더30회이며, 결과와 정확한 체크포인트 SHA256은 `results/student_pilot/cpu_recovery_pilot.json`에 있습니다. 학습과 검증 경로/내용 중복을 검사합니다. 별도 화자·문장으로 최종 품질을 검증하기 전에는 일반 성능으로 해석하지 않습니다.

## 실행

학습 환경에는 CPU PyTorch와 원본 프로젝트 의존성이 필요합니다. `PYTHONPATH`에 저장소와 `GPT_SoVITS`를 넣습니다. 각 출력은 새 경로여야 합니다.

```sh
python experiments/student/observe_host.py --report /fresh/ar-host.json -- \
  python experiments/student/distill_ar.py \
  --teacher /assets/s1v3.ckpt --spec /research/student_spec.json \
  --manifest /research/manifest_reference_pilot.json \
  --output /fresh/ar20 --steps 50 --threads 8

python experiments/student/observe_host.py --report /fresh/decoder-host.json -- \
  python experiments/student/distill_decoder.py \
  --s2 /assets/s2Gv2ProPlus.pth --spec /research/student_spec.json \
  --manifest /research/manifest_reference_pilot.json \
  --output /fresh/decoder13 --steps 30 --threads 8
```

`observe_host.py`는 CPU/RAM 사용량을 관측하며 학습 자원을 2 CPU/8GB로 제한하지 않습니다. 학습 프로그램은 CPU FP32를 확인합니다. 배포 성능 검증은 별도로 실제 N1 제약에서 수행합니다.

후학습된 AR 가중치는 일반 FP32 값입니다. 원본의 exact-FP16 저장 패킹을 강제로 적용하면 정상적으로 거부됩니다. `synthesize_student.py --ar-fp32-cache`를 명시해 기존 FP32 캐시 경로를 사용하며 prefetch=0을 설정합니다. 반올림이나 양자화를 하지 않습니다. 저장량 증가가 속도 이득을 상쇄할 수 있으므로 원본 캐시 결과와 학생 성능을 섞지 않습니다.

`--variant B/D`에는 별도로 내보낸 AR 학생을 `--s1`으로 전달합니다. C/D의 학습된 디코더는 `--decoder-student`로 전달합니다. 원본 sample rate, 텍스트, seed, 레퍼런스를 유지하고 자유 생성 음성·semantic token·파라미터 무결성을 기록합니다.

## 채택 순서

1. 파라미터/원본 파일 무결성, 유한한 원래 샘플레이트 출력 확인.
2. 같은 조건의 원본/학생 음성을 사용자에게 제공해 음색·발음·억양·잡음 평가.
3. 승인된 학생만 실제 N1 Actions에서 원본과 비교. 전체 문장 생성, 오디오 길이, AR/디코더 시간, RTF, RSS 및 실패 로그를 함께 보존.
4. 품질이 부족하면 후학습 또는 더 보수적인 깊이/분기 축소로 돌아감. 작은 검증 손실만으로 기본 모델을 교체하지 않음.

원본 모델 커널 최적화는 별도 `codex/cpu-performance` 작업 트리에서 진행합니다. 그 경로의 N1 측정은 원본 모델에 대한 결과이며 학생 승인이나 성능으로 간주하지 않습니다.

## 400회 후학습 재검토와 확장 단계

첫 CPU 후학습 B/C/D도 사용자가 모두 품질 저하로 거부했습니다. `listening_trained_pilot.json`에 기록했습니다. V5 채팅을 다시 읽고 실제 `round12_refined_train_guard.json`과 `distill_flow.py`를 대조했습니다. 그 프로젝트에는 1,500회 초기 학습 뒤 full-context, lr2e-5, 4회마다 종단 mel rollout 손실을 넣은 400회 보정이 실제 존재합니다. 별도 Vocos 보정에는 log-mel + .25 log-STFT + .5 spectral 목표의 600회 학습도 있었습니다. 두 경로 모두 수치 개선만으로 청취 승인을 받지 못했습니다. 근거 파일 SHA와 결과는 `v5_400_recheck.json`에 있습니다.

V2의 다음 확장 실험은 다음과 같습니다.

- `recovery_texts.json`: 한·일·영 각 학습10/검증2문장, 총30/6. 학습 문장은 기존 청취 비교 문장을 포함하지 않습니다.
- `collect_contexts.py --speakers reference --texts ...`: 확인된 한 개의 V5 생성 레퍼런스만 사용. 마지막에 교사 무결성과 수집 완료를 확인하며, 부분 수집 파일을 학습하지 않습니다. 다화자 일반화 검증은 여전히 필요합니다.
- AR: 첫 best를 초기값으로 400회 추가 학습, lr2e-5, 언어를 순환하며 KL 감독. 모든25회 검증하고 마지막 결과 대신 검증값이 가장 낮은 체크포인트를 보존합니다.
- 디코더: 첫 best에서 시작해 `--full-context --log-mel --steps 400 --validate-every 25`. 동일 길이의 원래32kHz 파형에서 512/1024/2048 FFT의 Slaney log-mel 및 log-STFT를 비교합니다. 목표는 log-mel + .25 log-STFT + .5 기존 spectral + .5 단계별 feature입니다. sample rate나 재생 속도를 바꾸지 않습니다.
- 학습 전체 문맥에서 고정 교사의 파형/특징을 캐시하지만 학생 파라미터/연산은 그대로이며 학습은 CPU FP32입니다. V5의 CFM mel endpoint 손실을 V2 AR에 억지로 적용하지 않습니다.
- 이 단계는 새 품질 평가를 위한 실험입니다. 아직 청취 승인이나 실제 N1 학생 RTF 결과가 없습니다. 학습 횟수와 청취 품질은 같은 지표가 아닙니다.

학습용 특징 수집 forward는 stock teacher 및 내보내는 StudentGenerator와 정확히 같은 출력을 내는 테스트를 통과했습니다. 로그 손실의 동일 입력/무음/유한한 gradient도 검사합니다. 첫 확장 수집에서 누락된 영어 NLTK tagger로 실패한 부분 결과는 보존했고, 기존 로컬 리소스를 읽기 전용 mount한 새 수집으로 재시도했습니다. 새로운 데이터 다운로드는 없었습니다.

400회 확장 실험은 AR/디코더 모두 정상 종료했습니다. AR은 추가25회에서 KL0.042746이 최선이고 마지막400회는0.075788이므로 최선만 선택했습니다. 디코더는 추가175회에서 종합0.339259, log-mel0.098504, spectral0.352913이 최선이며 마지막400회는 각각0.360572/0.102587/0.381209입니다. 마지막 가중치를 무조건 채택하지 않습니다. 상세 history와 CPU 관측/체크포인트 SHA는 `cpu_recovery400.json`에 있습니다. 수집된73개 타깃 파일의 SHA도 학습 후 모두 일치했습니다.

새 청취 표시 E는 최선 AR + 원본 디코더, F는 원본 AR + 최선 디코더, G는 두 학생을 결합한 경로입니다. 한·일·영9개 음성이32k/finite/원본 파일 및 실행 중 파라미터 해시 검사를 통과했습니다. F는 원본 semantic token과 길이를 유지하며, E/G는 자유 생성 token과 길이가 달라집니다. 전체 문장 누락·음색·억양은 사용자 청취로 평가하며 품질 승인은 아직 없습니다. 결과는 `recovery400_listening.json`에 있습니다.

사용자는 E/F/G의 먹먹함 개선을 확인했고 G의 첫 문장 끝 차이는 허용 가능한 수준이라고 평가했습니다. G를 이후 비교 기준으로 별도 보존했습니다. 이 승인은 한국어 한 레퍼런스 비교에 한정하며 원본 기본 모델을 바꾸지 않습니다. 정확한 가중치 해시와 피드백은 `results/student_pilot/approved_g.json`에 있습니다.

## 다음 축소와 실제 속도 판정

목적은 실제 N1의 RTF 감소입니다. 파라미터 감소와 검증 손실 개선은 속도 성과가 아닙니다. 첫 G N1 측정에서 한국어 원본 최적화 경로 RTF1.44398, G 결합 경로1.44139로 실질 개선이 거의 없었습니다. 생성 시간은8.346초→8.504초, 출력 길이는5.78초→5.90초입니다. 원본의 exact-FP16 캐시 대신 일반 FP32 캐시를 사용한 비용이 AR 층 축소 이득을 상쇄했습니다. 따라서 실제 서버의 생성 시간·RTF·단계별 시간과 청취 품질을 함께 확인한 뒤 다음 축소를 결정합니다.

`prune_approved.py`는 승인된 G에서 학습 context 한·일·영 각1개로 민감도를 검사해 AR 원본12/14번 층을 추가 제거합니다. AR18을400회, 누적3문장(각 언어1개), cosine lr1e-5→1e-6로 보정한 H는 사용자가 원본 교사 대비 부족하다고 평가했습니다. `collect_ar_features.py`로 원본 AR24의6/13/23번 층 semantic 구간 특징을 별도 수집하고, `distill_ar.py --feature-dir ... --feature-weight 0.1`로 KL에 정규화 특징 MSE를 더해400회(lr3e-6→3e-7) 보완했습니다. H2는 검증 KL0.046241→0.043376, 특징 오차0.019814→0.014714이며 사용자가 한국어 비교에서 사용 가능으로 승인했습니다. 기존 context와 원본 교사 파일은 변경하지 않습니다.

`prepare_half_cache.py`는 H2를 별도 복사해 각 AR 층의4개 선형 행렬 및 predictor 행렬만 FP16 표현 가능한 값으로 반올림합니다. 18층에서 총73행렬입니다. bias·LayerNorm·embedding·디코더는 그대로입니다. H3는 해당 복사본을 기존 exact-FP16 저장 캐시로 읽으며 FP32 곱셈/누산을 유지합니다. **H2 대비 계수 변환은 손실 변환**이며 H2 원본과 파형 일치를 주장하지 않습니다. 사용자는 H3도 한국어 비교에서 사용 가능으로 승인했습니다. H2/H3는 같은 N1에서 비교하며 속도 결과 전에는 개선을 주장하지 않습니다. `approved_h2.json`, `approved_h3.json`, `h3_half_conversion.json`, `incremental_training.json`에 provenance와 범위가 있습니다.

디코더 첫 단계11-tap 분기를 제거한12분기 모델은200회 후에도 G 대비 log-mel 오차가 커서 미채택입니다. 더 보수적인3-tap 제거는 `prune_decoder_branch.py --stage 0 --original-branch 0`으로 재현하며7/11-tap 분기를 유지합니다. 정해진100회만 시험하고, 이후 구조 탐색은 실제 서버 시간 감소 근거를 우선합니다. 두 경로 모두 기본 모델을 교체하지 않습니다.

보수적인 디코더는100회에서 log-mel0.326903→0.117517로 종료했습니다. H3 AR과 결합한 K는 H3와 한·일·영 semantic token/출력 길이가 같으며, 사용자가 한국어에서 사용 가능으로 승인했습니다. `approved_k.json`과 `conservative_decoder12_listening.json`이 해당 범위를 기록합니다. G/H2/H3/K는 비공개 HF 저장소에 중요 모델만 보존하고 fresh-download SHA256까지 확인합니다. 미채택/중복 모델은 삭제 기록을 남기며 원본·학습 타깃·로그·청취 음성은 유지합니다. K의 추가 디코더 축소가 실제 속도 이득인지 H3와 같은 N1에서 측정하기 전에는 속도 성과로 주장하지 않습니다.
