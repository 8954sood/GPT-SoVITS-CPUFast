# CPUFast 스타일·PCM 스트리밍 API

기존 `gpt-sovits-v5-optimization` 작업 폴더의 API를 가져와 승인된
**AR17+C24 Vtarget100** 추론 엔진에 연결했다. 원래 프로젝트는 수정하지 않았다.
복사한 원본 파일의 SHA와 당시 소스 버전은 [API_PORT_SOURCE.json](docs/API_PORT_SOURCE.json)에 있다.

## API 호환 범위

| 메서드·경로 | 기능 |
| --- | --- |
| `GET /health` | 준비 상태, 작업 중 여부, 캐시·모델 정보 |
| `GET /v1/styles` | 등록된 스타일 목록 |
| `GET /v1/styles/{style_id}` | 스타일 메타데이터 |
| `POST /v1/styles` | multipart WAV·원문·언어 등록 |
| `DELETE /v1/styles/{style_id}` | 등록한 WAV·캐시 삭제, 성공 204 |
| `POST /v1/tts/stream` | 스타일 ID로 문장별 PCM 스트리밍 |

요청 필드, 스타일 ID, 기본 포트9880, 오류 상태, 영구 저장, 단일 추론 소유권,
backpressure와 연결 중단 시 정리 방식을 유지한다. 등록은 최대8MiB/60초의
mono·stereo WAV에서3–10초 구간을 선택한다. `ja`, `ko`, `en`을 지원한다.
합성·등록·삭제 작업은 한 번에 하나이며 다른 변경 요청은429/Retry-After:1이다.
health와 목록 조회는 합성 중에도 응답한다.

응답은 모델의 원래 출력인 **32kHz / mono / PCM signed16 little endian**이다.
`application/octet-stream`이며 WAV 헤더가 없다. 사용자 요청에 따라 이전 API의
48kHz 변환을 제거해 모델이 만든 PCM을 재변환 없이 보낸다.
클라이언트는 `X-Audio-Sample-Rate`를 읽어 재생 속도를 정한다.
`stream_client.py`의 명령은 같으며 저장할 WAV의 샘플레이트를 응답에서 읽는다.

`latency_mode`의 `quality`/`balanced`/`fast`는 요청 호환을 위해 모두 받지만
이 엔진에서는 **같은 승인 모델과 전체 레퍼런스**를 사용한다. 기본값은 `fast`다.
응답의 `X-Latency-Mode-Effective: fixed_approved_model`과 health에서 이를
표시한다. V5의 CFM 구간 축소를 실행한다고 표시하지 않는다.
`reference_frames`를 지정하거나 `streaming_mode: tokens`를 요청하면422다.
`max_segment_chars`는20–240, 기본120이며 seed는 문장마다1 증가(uint32 wrap)한다.

## 시작

Linux ARM64에서 기존 CPUFast 추론 의존성, 승인 모델, HuBERT·SV·언어 자료를
준비한다. 네이티브 라이브러리는 같은 PyTorch 환경과 현재 소스로 빌드한다.
실제 사용한 모델과 호스트 준비는 [재현 지침](docs/C24_N1_REPRODUCTION.md)을 따른다.

```bash
python -m pip install -r requirements-server.txt
python build_cpufast_ar.py --output "$PWD/libcpufast_ar_torch.so"
python build_cpufast_fast.py --output "$PWD/libcpufast_fast.so"

export PYTHONPATH="$PWD:$PWD/GPT_SoVITS"
export CPUFAST_AR_LIBRARY="$PWD/libcpufast_ar_torch.so"
export CPUFAST_FAST_LIBRARY="$PWD/libcpufast_fast.so"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=100000

python server.py --host 127.0.0.1 --port 9880 \
  --manifest /path/to/approved-student/manifest.json \
  --manifest-sha256 696966debccf516f2af6ef0f87be2c2cf52548d313d70f8f467731dbf1f7a865 \
  --s2 /path/to/s2Gv2ProPlus.pth \
  --hubert /path/to/chinese-hubert-base \
  --data-dir data/styles
```

`--manifest`, `--s2`, `--hubert`는 실제 파일 경로로 지정한다. 로컬 사설 경로와
SSH 키를 소스에 넣지 않는다. API 문서는 `/docs`에서 확인한다.
SV 체크포인트는 기본적으로 저장소의
`GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`에 둔다.
다른 데이터 루트를 사용하면 시작 전에 `GPT_SOVITS_DATA_DIR`를 지정하고 그 아래
동일한 상대 경로에 SV 파일을 준비한다. 이 파일도 스타일 캐시의 모델 식별에 포함된다.
프로세스는 하나이고 Torch 음향/AR 스레드는 각각2다. 다른 서비스와 별도
프로세스에서 실행하며 기존 서버를 덮어쓰지 않는다. 다른 포트가 필요하면
`--port`를 지정한다. 실제 N1 검증과 같은 하드 제한은 Docker의2CPU·8GB·swap0
등 호스트 설정으로 적용한다. 스타일 텐서64MiB 한도는 전체 RSS 제한이 아니다.

## 등록과 합성

```bash
curl --fail-with-body http://127.0.0.1:9880/v1/styles \
  -F 'audio=@reference.wav' -F 'name=my_voice' \
  -F 'prompt_lang=ja' \
  -F 'prompt_text=スマートフォンからフィーチャーフォンまで、マルチデバイスに対応。'

python stream_client.py styles
python stream_client.py tts STYLE_ID \
  '오늘은 날씨가 참 좋네요. 잠깐 쉬었다가 함께 이야기를 이어 가요.' \
  --lang ko --output results/streamed.wav

python stream_client.py delete STYLE_ID
```

`STYLE_ID`는 등록 응답의 ID로 바꾼다. raw PCM을 직접 받으려면:

```bash
curl --fail-with-body --no-buffer http://127.0.0.1:9880/v1/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"style_id":"STYLE_ID","text":"こんにちは。","lang":"ja","seed":42}' \
  --output streamed.pcm
ffplay -nodisp -autoexit -f s16le -ar 32000 -ac 1 streamed.pcm
```

## 저장과 수명 관리

`data/styles/<style_id>/` 아래 원본 `reference.wav`, `metadata.json`,
`prepared.pt`를 보존한다. 정상 재시작 후 ID가 유지되고, 호환되는 전처리
텐서를 메모리로 읽어 합성에 바로 사용한다. 모델·전처리·라이브러리 fingerprint나
캐시 스키마가 달라지면 원본으로 다시 준비한다. 삭제 중단의 tombstone을
재시작 때 정리하며, 손상된 원본은 조용히 누락하지 않고 시작을 실패시킨다.
새 API의 데이터 디렉터리를 별도로 사용한다. 음성·캐시 파일은 Git에서 제외한다.

추론은 문장 하나가 완성되면 그 음성을 먼저 보내고 다음 문장을 합성한다.
네트워크 청크는 최대9600바이트다. 첫 PCM 생성에 성공한 뒤 HTTP200을 연다.
클라이언트가 끊겨도 실행 중인 네이티브 호출이 끝나기 전에는 다음 모델 작업을
받지 않는다. 전송 시작 뒤 실패하면 해당 스트림을 끊고 요청 ID를 로그에 남긴다.

실시간 경로는 모델의32kHz PCM을 직접 전송하고, 문장마다 전체 모델 힙을
강제 스캔하던 정리를 줄인다. 자동 메모리 관리는 유지한다. AR은 실제로
컴파일된 클래스 인스턴스를 실행하되, FP32 계산의 반올림이 달라지는 JIT
그래프 재작성은 AR 메서드 안에서만 끈다. 가중치·참조 음성·문장 분할·seed와
모델 출력 길이는 바꾸지 않는다. 토큰 캐시는 유효한 프롬프트만 복사하며
최대 생성 용량은 그대로 보존한다.

### 실제 스트림 RTF 측정

스타일을 등록한 뒤 다음처럼 측정한다. 생성한 음성 파일은 저장하지 않는다.

```bash
python scripts/measure_streaming_api.py STYLE_ID \
  '오늘은 날씨가 참 좋네요. 잠깐 쉬었다가 함께 이야기를 이어 가요.' \
  --lang ko --warmups 3 --samples 5
```

RTF는 **클라이언트 요청 시작부터 응답 수신 완료까지의 시간 / 받은 PCM 길이**다.
첫 음성 대기를 빼지 않으며, 워밍업과 측정 샘플을 모두 출력한다.
첫 PCM 도착 시간, 샘플 수·SHA, 요청별 RTF와 최악값도 남긴다.
문장 단위 스트림은 전체 RTF가1 미만이어도 즉시 재생하면 문장 사이에
버퍼가 부족할 수 있어, 관측한 도착 시각에 필요한 추가 초기 대기 시간도
`required_extra_initial_buffer_seconds`로 구분한다. 기기 재생 지연은 포함하지 않는다.

이전 최악RTF0.962389는 워밍업 후 전체 파일 생성 실측이다. 새 HTTP 경로의
등록 비용·첫 오디오 지연과 같은 지표로 취급하지 않는다.

## 32kHz 직접 스트리밍 최종 ARM 실측

2026-11-13 실제 Neoverse N1에서 **2CPU / 8,000,000,000B / swap0**으로
정식 API를 검증했다. 측정 소스는 `139eaef`이며 현재 API 실행 코드와 동일하다.
HTTP 클라이언트는 서버와 별도 프로세스로 같은 컨테이너의 localhost에 연결했다.
인터넷을 통한 사용자 기기까지의 지연을 측정한 결과는 아니다.

먼저 한 서버에서3회 워밍업 뒤5회를 측정하고, 이어서 서버 프로세스를5번
새로 띄워 각각3회 워밍업 뒤1회를 측정하도록 계획을 고정했다. 서버는 순서대로
하나씩 실행했다. 워밍업 기록도 보존했으며, 아래10회에 워밍업은 포함하지 않는다.
모델 로드·스타일 등록 비용은 제외하고, **요청 시작·첫 음성 대기·전체 PCM 수신
완료를 포함**한다. 출력은 매번 같은5.08초/325,120바이트다.

| 측정 묶음 | 실제 클라이언트 RTF |
| --- | --- |
| 같은 서버5회 | 0.936370, 0.957164, 0.965698, 0.954824, 0.987373 |
| 별도 서버5회 | 0.952538, 0.953778, 0.959658, 0.956887, 0.953233 |

**10회 모두 RTF<1, 최악0.987373**이다. 특정 한국어 두 문장·일본어 전체
레퍼런스·seed42/43의 관측값이며, 다른 입력이나 cold inference까지 보장하지 않는다.
RTF<1과 문장 사이 무중단 재생은 별개다. 이 API는 문장 단위로 출력한다.
수신 시각과 필요한 추가 초기 버퍼를 원시 기록에서 함께 확인할 수 있다.

승인 AR17+C24 모델·레퍼런스·문장 분할·seed·네이티브4개 파일의 해시를
전후 확인했다. 두 문장 의미 토큰은44+68이며, 모든 요청의32kHz PCM SHA256은
`a3aa7f71c04a7a4b28457225575946c9b29ee7e23bef493ed9de262ecbb4be9d`로
최적화 전 모델 원래 출력과 정확히 일치한다. 음성 파일은 저장하지 않았다.

채택한 변경은32kHz 직접 전송, 문장별 전체 힙 강제 정리 제거,
유효 프롬프트만 복사하는 KV 캐시, 실제 컴파일된 AR 메서드 실행이다.
AR 안의 FP32 그래프 재작성은 꺼서 기존 샘플링 결과를 보존했다.
별도 INT8 packed4 커널은 단독 연산에서 빨랐지만 전체 API의 확실한 이득을
확인하지 못해 채택하지 않았다. Tap finish·스레드/대기/affinity 변경도
정식 경로에 추가하지 않았다.

호스트 통합 테스트93개 통과/19개 ARM 전용 skip이며 실제 ARM 검증은
위 측정으로 별도 수행했다.
[최종 요약](results/n1_api_runtime_20261113/final/summary.json),
[소스·자산·실행 출처](results/n1_api_runtime_20261113/final/provenance.json),
[HTTP 원시 기록](results/n1_api_runtime_20261113/final/raw/)과
[미채택·무효 탐색 기록](results/n1_api_runtime_20261113/exploration/)을 보존했다.

## 이식 직후48kHz 경로의 ARM 서버 검증

아래는32kHz 직접 전송과 정리 비용 최적화 전의 기록이다.
2026-11-13 N1에서2CPU·8,000,000,000바이트·swap0으로 실제 엔진과 임시
localhost HTTP 서버를 검증했다. 전체 문장 엔진 출력은 승인된32kHz PCM과
일치했고, 준비한 레퍼런스를 저장한 뒤 다시 읽어도48kHz PCM이 같았다.
합성 중 레퍼런스 파일·HuBERT·SV·참조 문장 전처리를 다시 호출하면 실패하도록
설정한 검증도 통과했다.

HTTP는 등록201, 두 문장 PCM 스트리밍200, 합성 중 health 응답과 중복 요청429,
재시작 후 같은 스타일 ID 복원(`preloaded_from_disk=1`, `rebuilt_on_startup=0`),
삭제204를 확인했다. 세 latency 별칭의 응답은 모두487,680바이트로 동일했다.
서버 내부 첫 PCM 준비 시간은2.228–2.396초였다. 클라이언트 도착 지연은
별도로 측정하지 않았다. 출력5.08초에 서버 로그 전체 시간은5.427–5.721초로,
이번 문장 분할 API의 관측RTF는**1.068–1.126**이다. 기존 파일 생성RTF<1을
API에서도 달성했다고 표현하지 않는다. 고정된 한 입력의 세 요청 관측이며
부하 시험이나 최악 지연 보장은 아니다. HTTP는 문장 분할과 문장별 seed 때문에
전체 문장을 한 번에 생성한 엔진 검증과 출력 SHA가 다르다.
검증용 서버는 모두 종료했다.
실측 소스·실행 스크립트·요청별 원시 기록은
[검증 출처](results/n1_api_20261113/provenance.json),
[엔진 검증](results/n1_api_20261113/backend/adapter_validation.json),
[HTTP 검증](results/n1_api_20261113/http/http_validation.json)에 보존했다.
