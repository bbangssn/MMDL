# MMDL: Qwen3-VL on MMMU

Qwen3-VL-4B-Instruct의 MMMU/MMMU-Pro baseline을 재현하고, 이후 failure analysis와 fine-tuning 실험으로 연결하기 위한 프로젝트입니다.

## 구조

```text
.
├── README.md
├── assignment/                # 과제 보고서 및 제출 문서
├── code/
│   ├── 00_mmmu_vllm.py        # 독립 vLLM MMMU Validation 평가
│   ├── 01_mmmu_pro.py         # MMMU-Pro 평가
│   ├── 02_mmmu_vllm_judge.py  # vLLM 결과의 Llama 3.1 8B fallback 판정
│   ├── 03_modality_ablation.py # MMMU-Pro 200문항 modality ablation
│   ├── 04_bottleneck_probes.py # 24문항 paired 병목 진단
│   ├── 05_conflict_control.py  # 무관 이미지 경고문 대조 조건
│   ├── 06_mmmu_openai_judge.py # vLLM 결과의 OpenAI API fallback 판정
│   ├── 07_mmmu_modality_ablation.py # MMMU Validation 200문항 modality ablation
│   ├── 08_mmmu_modality_llama_judge.py # 실험 07 fallback의 local-Llama 판정
│   ├── 09_mmmu_option_permutation.py # MMMU 객관식 전수 선지 순서 반복 실험
│   ├── 10_mmmu_option_permutation_llama_judge.py # 실험 09 local-Llama 판정
│   ├── 11_mmmu_llama_judge_prompt_ablation.py # Llama judge 출력 형식 실험
│   ├── 12_mmmu_pro_resolution_scale.py # 120문항 paired 해상도 확장 실험
│   ├── 13_mmmu_pro_resolution_llama_judge.py # 실험 12 strict Llama 판정
│   ├── analyze_bottleneck_probes.py # paired CI/fallback 제외 분석
│   ├── utils.py               # 모델 registry, inference, parser, 결과 I/O
│   ├── reporting.py           # 환경 기록과 category/subject/runtime 집계
│   ├── vllm_compat.py          # GPU/CUDA별 FlashInfer sampler 호환 설정
│   ├── download_assets.py
│   ├── figurefactory.ipynb
│   └── requirements.txt
└── results/                   # prediction, summary, figure
```

실험 설정은 각 실험 파일 상단의 대문자 변수에서 수정합니다.

### 설치

모든 Qwen 추론 실험은 `vllm>=0.11.0`을 공통 backend로 사용합니다.

```bash
pip install -r code/requirements.txt
```


## 실행

```bash
pip install -r code/requirements.txt

python code/download_assets.py
python code/run_mmmu_baseline.py  # 00 vLLM + 02 local judge
python code/01_mmmu_pro.py
python code/03_modality_ablation.py
python code/04_bottleneck_probes.py
python code/05_conflict_control.py
python code/analyze_bottleneck_probes.py
python code/07_mmmu_modality_ablation.py
python code/08_mmmu_modality_llama_judge.py
python code/09_mmmu_option_permutation.py
python code/10_mmmu_option_permutation_llama_judge.py
python code/11_mmmu_llama_judge_prompt_ablation.py
python code/12_mmmu_pro_resolution_scale.py
python code/13_mmmu_pro_resolution_llama_judge.py
```

OpenAI API judge를 사용할 때는 Git에서 제외된 `openai_key.json`에 키를
입력하고 실행합니다. API 모델과 반복 횟수는
`code/06_mmmu_openai_judge.py` 상단의 대문자 변수에서 설정합니다.

```json
{
  "OPENAI_API_KEY": "sk-...",
  "OPENAI_BASE_URL": "https://api.openai.com/v1"
}
```

```bash
python code/06_mmmu_openai_judge.py
```

이 스크립트는 [Qwen3-VL 공개 MMMU 평가 코드](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)의
judge prompt, temperature 0, API 오류 재시도 및 최대 25회 answer-extraction
재질의 구조를 따른 뒤, 끝까지 unresolved인 경우 seeded random fallback을 적용합니다.
기본 judge는 공식 설정과 같은 legacy 모델 `gpt-3.5-turbo-0125`이며, 계정에서
사용할 수 없다면 상단의 `JUDGE_MODEL`을 `gpt-4o-mini`로 변경할 수 있습니다.

MMMU 중심의 modality 분석은 `07_mmmu_modality_ablation.py`가 Validation
200문항에 대해 full, text-only, shuffled-image, blank-image 조건을 비교합니다.
30개 과목에서 6~7문항씩 균형 표집하며, full 조건은 실험 00의 동일 문항
출력을 재사용합니다. 조건별 raw accuracy와 fallback 제외 accuracy를 함께
저장합니다.

선지 위치 민감도 실험은 `09_mmmu_option_permutation.py`가 MMMU Validation의
객관식 847문항 전부를 사용합니다. 원본 조건은 다시 추론하지 않고, 문항마다
선지를 원래 순서와 다르게 무작위 재배열한 복수의 `option_permuted_XX`
조건만 실행합니다. 이동된 정답 문자를 원래 선택지 의미로 역변환하여
반복별 accuracy와 전체 평균을 저장합니다. 이어서
`10_mmmu_option_permutation_llama_judge.py`가 두 조건의 parser fallback을
동일한 local Llama로 판정합니다. 실험 10은 Llama가 설명 뒤에 명시한 마지막
`Output: X`도 인식하므로, 기존 실험 02/08의 절대 점수와 직접 비교하지 않고
shuffle 반복 사이의 분산을 확인합니다. 반복 수는 두 파일 상단의
`PERMUTATION_REPETITIONS`에서 동일하게 지정하며 기본값은 3입니다.

기본 데이터 소스 대신 로컬 MMMU dataset script/path를 사용하려면 CLI 인자 대신 환경변수로 덮어씁니다.

```bash
MMDL_MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct MMMU_DATA_PATH=/path/to/MMMU python code/00_mmmu_vllm.py
MMMU_PRO_DATA_PATH=/path/to/MMMU_Pro python code/01_mmmu_pro.py
MMMU_DATA_PATH=/path/to/MMMU python code/02_mmmu_vllm_judge.py
MMMU_PRO_DATA_PATH=/path/to/MMMU_Pro python code/03_modality_ablation.py
MMMU_PRO_DATA_PATH=/path/to/MMMU_Pro python code/04_bottleneck_probes.py
MMMU_PRO_DATA_PATH=/path/to/MMMU_Pro python code/05_conflict_control.py
python code/analyze_bottleneck_probes.py
```

MMMU-Pro 설정은 `code/01_mmmu_pro.py`의 `SETTING`에서 `vision` 또는 `standard10`을 선택합니다.

## 결과

Assignment 01 최종 제출 보고서는 [`assignment/assignment01.md`](assignment/assignment01.md)에 있습니다.

한 모델·설정당 다음 고정 경로에 누적됩니다.

```text
results/01_mmmu_pro/<setting>/<model>/
├── config.json
├── environment.json
├── predictions.jsonl
├── progress.json
└── summary.json
```

`predictions.jsonl`에는 sample ID, category, subject, 문제 유형, 정답, raw model output, parsed answer, 정오 및 sample runtime이 저장됩니다. `summary.json`에는 overall/category/subject accuracy와 runtime 집계가 저장됩니다. `environment.json`에는 GPU/VRAM, CPU, RAM, OS, Python, CUDA 및 주요 package version이 기록됩니다.

vLLM 전용 judge는 `results/02_mmmu_vllm_judge/<run-tag>/<source-model>/<judge-model>/<judge-run-tag>/`에 별도로 저장되며, `02_mmmu_vllm_judge.py`의 `SOURCE_RUN_TAG`로 입력 vLLM 실행을 선택합니다.

재실행하면 완료된 sample ID를 건너뛰고 자동 재개합니다. 결과에 영향을 주는 설정이 기존 `config.json`과 다르면 서로 다른 조건의 결과가 섞이지 않도록 실행을 중단합니다. 새 조건으로 실험할 때는 해당 결과 디렉터리를 먼저 백업하십시오.

vLLM 실험은 `results/00_mmmu_vllm/<run-tag>/<model>/`에 저장됩니다. `code/00_mmmu_vllm.py` 상단의 `LIMIT`으로 smoke test 범위를 정할 수 있으며 `None`은 전체 900문항을 실행합니다. 기본 `PRESENCE_PENALTY=1.5`는 Qwen 공식 vLLM recipe를 따른 값이므로, 실행 중에는 GPU 0 전체 사용량을 0.5초 간격으로 polling하며, idle baseline·절대 peak·baseline 대비 증가량을 `peak_vram.json`, `progress.json`, `summary.json`에 기록합니다. vLLM worker process를 포함하기 위한 방식이므로 같은 GPU의 다른 프로세스가 있으면 그 사용량도 포함됩니다.

## vLLM 결과

- MMMU Validation rule-only: **217/900 (24.11%)**; local Llama judge와 seeded random fallback 적용 후 **469/900 (52.11%)**
- MMMU-Pro standard10 test: **724/1,730 (41.85%)**
- MMMU-Pro modality ablation (각 200개): full **48.0%**, text-only **24.0%**, shuffled-image **21.0%**, blank-image **23.5%**
- 24문항 병목 probe: direct **29.2%**, oracle rationale **45.8%**, shuffled-image **16.7%**, 3-seed prediction stability **45.8%**

04/05는 방향성 pilot이며 정식 benchmark 점수로 해석하지 않습니다. 상세 CI와 fallback 제외 분석은 `results/04_bottleneck_probes/qwen3-vl-4b-instruct/analysis.json`에 있습니다.

## 실험 결정 기록

이 절은 최종 보고서를 작성할 때 선택 근거를 되짚기 위한 간단한 기록입니다.

- **주 평가 split — MMMU Validation:** 현재 assignment가 실제 pipeline을 구축해 900개 Validation sample의 baseline을 재현하고 상세 분석하는 것을 요구합니다. Dev는 few-shot 예시용이며 사용하지 않습니다. MMMU Test는 최종 프로젝트 평가 단계로 남겨둡니다.
- **MMMU-Pro — 보조 평가:** 첫 assignment의 필수 대상은 아니지만 최종 프로젝트의 일반화 성능 확인을 위해 실험 01로 분리했습니다. `vision`과 `standard10` 결과는 별도 디렉터리에 저장합니다.
- **모델 — Qwen3-VL-4B-Instruct:** 과제 지정 base model이며 공개 성능(MMMU 67.4, MMMU-Pro 53.2)과 비교합니다. 공개 수치는 split·backend·parser 차이가 있을 수 있으므로 동일 조건의 절대 재현값으로 단정하지 않습니다.
- **평가 프로토콜:** Qwen3-VL 공개 MMMU prompt와 rule parser, MMMU-Pro 공식 multiple-choice parser를 우선 따릅니다. 모든 Qwen 실험은 vLLM과 `presence_penalty=1.5`를 사용합니다.
- **한 개 GPU:** `CUDA_VISIBLE_DEVICES`로 RTX 4090 한 장만 노출합니다. 물리 GPU 번호는 각 실험의 `VISIBLE_GPU`에서 선택합니다.
- **길이 설정:** `MAX_MODEL_LENGTH=9048`, `MAX_NEW_TOKENS=2048`입니다. vLLM의 `max_model_len`으로 입력과 생성 token 합계를 9,048 이하로 제한합니다. 이는 24GB GPU에서 안정적으로 평가하기 위한 자원 제한이며 Qwen 공식 수치의 고유 설정이라고 주장하지 않습니다.
- **이미지 해상도:** Qwen 공개 설정을 참고해 `MIN_PIXELS=1280*28*28`, `MAX_PIXELS=5120*28*28`을 기록하고 적용합니다. 다만 MMMU-Pro의 35-image 문항 1개는 입력만 약 36,046 token이므로 `MAX_MODEL_LENGTH=9048`을 지키기 위해 각 이미지를 160 visual-token 상당(`160*28*28` pixels)으로 축소합니다. 해당 여부와 실제 pixel bound는 sample row에 기록합니다.
- **Sampling:** `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.0`, `presence_penalty=1.5`를 사용합니다. Qwen 공식 Instruct recipe와 같고, 측정 seed는 42로 고정합니다(공식 공개 seed 3407과 다름).
- **Raw 결과와 runtime:** 최종 점수만으로는 failure 원인을 검증할 수 없으므로 raw response와 sample runtime을 보존합니다. runtime이 없는 행은 임의로 추정하지 않고 `untimed_legacy_samples`로 분리합니다.
- **Fallback 점수:** 분석 노트북의 기본 accuracy는 parser fallback 문항을 제외합니다. 전체 raw accuracy와 fallback 제외 accuracy를 구분해 보고해야 합니다.
- **LLM-as-a-Judge:** 반복 실험은 로컬 Llama 3.1 8B Instruct에 strict one-label prompt, greedy decoding, `MAX_NEW_TOKENS=8`, robust final-label parser를 사용합니다. 실험 11의 596개 fallback 전수 비교에서 99.66%를 명시적으로 판정했고 GPT-3.5와 양쪽 모두 resolved인 label 일치율은 92.53%였습니다. 추출 실패 시에만 유효 선택지와 `Z` 중 하나를 seeded random fallback으로 선택합니다. 공식 성적 비교에는 별도로 GPT-3.5-Turbo judge 결과를 사용합니다.
- **병목 pilot:** 실험 04는 공식 설명이 있고 이미지가 하나인 MMMU-Pro 문항을 대분야별 4개씩 뽑은 24문항 진단입니다. benchmark 점수가 아니라 prompt, 해상도, 선택지 순서, 이미지 충돌, sampling 반응을 빠르게 비교하는 방향성 실험입니다. 동일 표본 paired 비교와 fallback 제외 정확도를 함께 보고합니다.

## 분석 원칙

보고서에서는 관찰과 원인 가설을 분리합니다.

```text
Overall/category/subject 성능
→ 취약 영역 식별
→ 해당 영역의 실제 오답 확인
→ 반복되는 failure pattern 분류
→ 원인 가설
→ 개선 target과 training data
→ fine-tuning 및 재평가 계획
```

OCR, visual understanding, chart interpretation, spatial/mathematical/multi-step reasoning, domain knowledge, answer-format failure 등의 분석이 진행된 상태입니다.

## 비교 제한

- MMMU 평가는 공식 기본 judge인 `gpt-3.5-turbo-0125` 대신 Llama 3.1 8B와 출력 형식을 제한한 strict prompt를 사용했으므로 완전히 동일한 공식 점수는 아닙니다.
