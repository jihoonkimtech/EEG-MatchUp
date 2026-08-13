# EEG-MatchUp

숏폼 영상 시청 중의 뇌파를 측정하고, 공개 EEG 데이터셋과 교차 검증해 **영상 유형별 인지·정서 반응 차이**를 정량화한 분석 프로젝트다.

광운대학교 2025-2 매치업 심화과정 경진대회 출품작이다.

---

## 개요

숏폼 콘텐츠가 시청자에게 어떤 인지 상태를 유발하는지를 EEG 주파수 대역 파워로 측정한다. 자체 측정 데이터만으로는 지표의 타당성을 담보하기 어렵기 때문에, 정서·음악 자극에 대한 공개 EEG 데이터셋 두 종을 동일한 파이프라인으로 처리해 지표가 조건에 따라 실제로 갈리는지를 함께 확인했다.

세 갈래 모두 **전처리 → 에폭 분할 → PSD 산출 → 지표 계산 → 조건 간 비교**라는 같은 흐름을 따른다.

| 갈래 | 데이터 | 조건 |
| --- | --- | --- |
| `short-form-EEG/` | 자체 측정 (6채널) | base / info / highlight / challenge / 120s |
| `gameEMO/` | GAMEEMO 공개 데이터셋 | Resting / Tranquility / Horror / Fun |
| `openMIIR/` | OpenMIIR 공개 데이터셋 | 음악 자극 (Stimuli_Meta 기준) |

---

## 자체 측정 설계

숏폼 시청 상황을 다섯 조건으로 나눠 측정했다.

- `base` — 기준 상태
- `info` — 정보 전달형 영상
- `highlight` — 하이라이트 편집형 영상
- `challenge` — 챌린지형 영상
- `120s` — 장시간 연속 시청

측정 장비는 6채널 구성이며, `Ch6`을 자극 트리거 채널로 사용해 영상 재생 시점을 기록했다. 임피던스 체크 결과는 `short-form-EEG/impedance.png`에 남겨뒀다.

---

## 파이프라인

### 1단계 — 원시 데이터 병합 (`raw_combiner.py`)

측정 장비가 내보낸 `.mat` 파일을 읽어 MNE `Raw` 객체로 변환하고, 한 조건에 속한 세션들을 하나로 병합한다. `.mat` 안의 `SR` 키에서 샘플링 주파수를, `y` 키에서 신호 배열을 꺼내 쓴다. 결과는 `combined_{조건}_raw.fif`로 저장한다.

### 2단계 — 전처리 (`pre-processor.py`)

트리거 채널에서 자극 시점을 뽑아 에폭으로 자른다.

- `Ch6`을 `misc` 타입으로 지정한 뒤 `mne.find_events()`로 상승 엣지를 검출한다
- `min_duration=2.0`으로 짧은 스파이크성 노이즈를 걸러낸다
- 검출된 이벤트 중 간격이 `MIN_EVENT_SEPARATION_SEC`보다 좁은 것은 중복으로 보고 제거한다
- `annotate_muscle_zscore()`로 근전도 아티팩트 구간을 표시한다

전처리 전후 신호와 PSD를 각각 플롯으로 남겨 육안 검증이 가능하도록 했다.

### 3단계 — 분석 (`analyzer.py`)

Welch 방법으로 1–45 Hz 구간의 PSD를 계산한 뒤(`n_fft=128`), 세 대역으로 나눠 파워를 적분한다.

| 대역 | 범위 |
| --- | --- |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |

자극 제시 직후 구간은 반응 지연을 고려해 `REACTION_DELAY_SECONDS`만큼 잘라내고 분석한다.

---

## 산출 지표

전두엽(F3, F4)과 후두엽(O1, O2) 채널에서 네 가지 지표를 계산한다.

| 지표 | 정의 | 해석 |
| --- | --- | --- |
| **FAA** | `log(F4 alpha) - log(F3 alpha)` | 전두엽 알파 비대칭. 접근–회피 성향의 대리 지표 |
| **FAP** | `(F3 alpha + F4 alpha) / 2` | 전두엽 알파 파워. 이완·비활성 정도 |
| **OASI** | `(O1 alpha + O2 alpha) / 2` | 후두엽 알파. 시각 주의 억제 정도 |
| **TBR** | `frontal theta / frontal beta` | 세타-베타 비율. 주의 집중 상태의 대리 지표 |

로그 계산 시 0 이하 값은 `NaN`으로 처리하고, TBR 분모에는 `1e-10`을 더해 0 나눗셈을 막는다.

조건별 결과는 `EEG_features_{조건}.csv`로 떨어지며, `compare_csv.py`가 이들을 모아 조건 간 비교표를 만든다.

---

## 시각화 산출물

각 조건마다 아래 세 종류를 생성한다.

- **Topomap** — Theta / Alpha / Beta 대역별 두피 분포도
- **PSD curve** — 채널별 주파수-파워 곡선 (`10*log10` 스케일)
- **Summary bar** — 네 지표를 한눈에 비교하는 막대그래프

---

## 실행

```bash
pip install mne numpy scipy pandas matplotlib h5py

# 1단계: 원시 .mat 병합
python raw_combiner.py

# 2단계: 이벤트 검출 및 에폭 분할
python pre-processor.py

# 3단계: PSD 및 지표 산출
python analyzer.py

# 조건 간 비교표 생성
python compare_csv.py
```

각 스크립트 상단의 `target_folder`, `SUBJECT`, `CHANNELS_TO_PICK` 등을 처리할 조건에 맞게 수정한 뒤 실행한다.

---

## 디렉토리 구조

```
EEG-MatchUp/
├── short-form-EEG/          자체 측정 데이터
│   ├── raw/                 조건별 원시 .mat 및 병합 .fif
│   ├── pre-processed/       에폭 분할 결과 (-epo.fif)
│   └── analyze/             지표 CSV, topomap, PSD 플롯
├── gameEMO/                 GAMEEMO 데이터셋 교차 검증
│   ├── preprocessed/        조건별 전처리 결과
│   └── analyze/             조건별 플롯 및 Summary
└── openMIIR/                OpenMIIR 데이터셋 교차 검증
    ├── eeg/                 원시 EEG
    ├── meta/                자극 메타데이터, 전극 정보
    └── analyze/             그룹별 PSD 곡선
```

---

## 사용 도구

`MNE-Python` · `NumPy` · `SciPy` · `pandas` · `Matplotlib` · `h5py`

---

## 참고

- 공개 데이터셋(GAMEEMO, OpenMIIR)은 각 배포처의 라이선스를 따른다. 이 저장소에는 분석 스크립트와 산출물이 포함되며, 원본 데이터의 재배포를 의도하지 않는다.
- 자체 측정 데이터는 6채널 구성의 제한된 표본이므로, 결과는 경향성 확인 수준으로 해석해야 한다.
