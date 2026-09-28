# Assignment 01: MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: _(기입 필요)_
- **팀원**: _(기입 필요)_
- **작성일**: 2026-09-28
- **재현 커맨드**: `python code/run_mmmu_baseline.py`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` |
| 모델 revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 데이터셋 / revision | `MMMU/MMMU` / `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` |
| 평가 split | validation, 30과목 × 30문항 = 900문항 |
| 추론 백엔드 | vLLM 0.30.0 / PyTorch 2.13.0 / CUDA 13.0 |
| 사용 GPU | NVIDIA RTX PRO 5000 Blackwell, 47.23 GiB, physical GPU 0 only |
| 실측 peak VRAM | Qwen vLLM: 39.412 GiB whole-device peak on a 48GB GPU (reserved KV cache included); local judge: 16.041 GiB reserved |
| 총 소요 시간 | Qwen vLLM 00:15:53 (900문제), local judge 추가 00:16:57; total 00:32:50 |
| 의존성 | [`code/requirements.txt`](../code/requirements.txt) |
| 실행 커맨드 | 아래 코드 블록 참조 |

```bash
conda activate MMDL
pip install -r code/requirements.txt
python code/run_mmmu_baseline.py
```

`run_mmmu_baseline.py`는 Qwen vLLM 추론과 local judge를 서로 다른 프로세스로 순서대로 실행한다. 기본값은 pin된 Hugging Face 모델·데이터를 사용한다. 실행 중에는 `predictions.jsonl`과 `progress.json`이 갱신되며, 중단 후 같은 명령으로 자동 재개된다. 모델 또는 데이터의 로컬 경로가 필요하면 각각 `MMDL_MODEL_PATH`, `MMMU_DATA_PATH` 환경변수로 지정한다.

추가 환경: Python 3.14.7, vLLM 0.30.0, Datasets 5.0.1, Accelerate 1.14.0, qwen-vl-utils 0.0.14, RAM 187.85 GiB, OS Linux-6.8.0-138-generic-x86_64-with-glibc2.35.

추론 백엔드는 vLLM을 사용하였다. 900문항을 batch 32로 처리해 실행 시간을 줄이고, sample별 JSONL 저장과 ID 기반 자동 재개를 함께 적용할 수 있기 때문이다. 정확도를 높이기 위한 선택이 아니라 처리량과 재현성을 위한 선택이며, `tensor_parallel_size=1`과 `CUDA_VISIBLE_DEVICES=0`으로 단일 GPU만 사용한다. 모델과 데이터 revision은 코드에 pin되어 있으며, 실행 결과의 `config.json`과 `environment.json`에도 함께 저장된다.

## 2. 프롬프트

**Multiple-choice 문항에 실제로 입력한 prompt 전문**:

```text
Question: {question}
Options:
A. {option_A}
B. {option_B}
...
Please select the correct answer from the options above.
```

**Short-answer 문항에 실제로 입력한 prompt 전문**:

```text
Question: {question}
```

이미지는 Qwen chat template의 image content로 질문과 함께 전달한다.

- **출처**: [Qwen3-VL 공식 MMMU evaluation의 prompt 구성](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)
- **선택 이유**: 강의자료의 공개 성능과 비교할 때 prompt 차이를 줄이고, 임의의 정답 형식 지시로 점수를 변경하지 않기 위해 공식 공개 구현의 문구를 유지하였다.


### Local LLM judge 프롬프트

Rule parser가 해석하지 못한 응답에는 다음 Qwen 공개 judge prompt를 사용하였다. `{options}`에는 `There are several options:`와 `A. ...` 형식의 전체 선택지가, `{answer}`에는 Qwen의 raw response가 들어간다.

```text
You are an AI assistant who will help me to match an answer with several options of a single-choice question. You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. If the meaning of all options are significantly different from the answer, output Z. Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z.
Example 1:
Question: What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
Answer: a cute teddy bear
Your output: A
Example 2:
Question: What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
Answer: Spider
Your output: Z
Example 3:
Question: {question}?
Options: {options}
Answer: {answer}
Your output:
```

- **출처**: Qwen3-VL 공개 MMMU `eval_utils.build_prompt()`를 문구 수정 없이 사용하였다.
- **Judge model 및 decoding**: `meta-llama/Llama-3.1-8B-Instruct`, greedy decoding, `max_new_tokens=4096`, `max_model_length=9048`.
- **선택 이유**: 공식 기본 judge인 `gpt-3.5-turbo-0125`는 로컬 checkpoint로 내려받을 수 없으므로, 단일 GPU에서 실행 가능한 instruction-tuned 8B 모델로 대체하였다. 따라서 prompt는 공식과 같지만 judge 모델은 공식 조건과 다르다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---:|
| `do_sample` | `True` |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 42 |

- **출처**: [Qwen3-VL 공식 MMMU evaluation README](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/README.md)와 [공식 evaluation reproduction 설정](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction)의 Instruct recipe를 사용하였다. `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.0`, `presence_penalty=1.5`는 공식 값과 같다. 측정 실행의 seed는 재실행을 고정하기 위해 42로 설정했으며 공식 공개 seed 3407과 다르다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---:|
| `max_model_length` | 9,048 (입력+출력 합계) |
| `max_new_tokens` | 2,048 |
| `min_pixels` | 1,003,520 = 1280 × 28² |
| `max_pixels` | 4,014,080 = 5120 × 28² |

**선택 근거:** 단일 GPU에서 900문항을 안정적으로 완료하기 위해 context와 생성 길이를 제한하고, 이미지 pixel budget은 Qwen 공개 평가 설정을 참고하였다. 공개 vLLM 예시보다 생성 예산이 작아 장문 reasoning이 잘릴 수 있지만, VRAM과 runtime을 통제하는 대신 생기는 trade-off로 기록하였다.

## 4. 채점(파싱) 방식

- **사용한 로직**: [Qwen3-VL 공식 MMMU evaluation](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)의 rule-based option extractor와 judge prompt를 참고해 구현하였다.
- **1차 rule parser**: 거부 응답을 `Z`로 처리한 뒤, 구두점을 제거한 token에서 유일한 option letter를 찾는다. letter가 모호하면 응답에 유일하게 등장하는 option text를 찾는다. 후보가 없거나 여러 개면 unresolved로 남긴다.
- **Short-answer 처리**: Qwen 공개 전처리와 같이 ground-truth answer를 A, `Other Answers`를 B로 놓은 판정 문제로 변환한다.
- **Fallback judge**: unresolved 596개에 공식 judge prompt를 적용하였다. API 모델을 내려받을 수 없어 `gpt-3.5-turbo-0125` 대신 로컬 `meta-llama/Llama-3.1-8B-Instruct`를 greedy decoding으로 사용했다.
- **공식 구현과의 차이**: local judge는 1회만 호출하며, 실패한 207개는 unresolved로 유지한다. 공식 코드의 25회 API retry 및 최종 random fallback은 사용하지 않았다. 따라서 아래 재현 점수는 완전한 공식 protocol 점수가 아니다.

## 5. 결과

아래 표는 **rule parser 결과에 local judge가 성공적으로 해석한 fallback만 합친 결과**이다. 모든 subject가 30문항이므로 macro average와 900문항 micro accuracy가 같다.

| No. | Subject | Data Num | Acc |
|---|---|---:|---:|
| 1 | Accounting | 30 | 53.3 |
| 2 | Agriculture | 30 | 36.7 |
| 3 | Architecture_and_Engineering | 30 | 36.7 |
| 4 | Art | 30 | 46.7 |
| 5 | Art_Theory | 30 | 73.3 |
| 6 | Basic_Medical_Science | 30 | 46.7 |
| 7 | Biology | 30 | 36.7 |
| 8 | Chemistry | 30 | 46.7 |
| 9 | Clinical_Medicine | 30 | 50.0 |
| 10 | Computer_Science | 30 | 43.3 |
| 11 | Design | 30 | 43.3 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 26.7 |
| 13 | Economics | 30 | 53.3 |
| 14 | Electronics | 30 | 56.7 |
| 15 | Energy_and_Power | 30 | 36.7 |
| 16 | Finance | 30 | 60.0 |
| 17 | Geography | 30 | 43.3 |
| 18 | History | 30 | 53.3 |
| 19 | Literature | 30 | 50.0 |
| 20 | Manage | 30 | 56.7 |
| 21 | Marketing | 30 | 73.3 |
| 22 | Materials | 30 | 36.7 |
| 23 | Math | 30 | 46.7 |
| 24 | Mechanical_Engineering | 30 | 56.7 |
| 25 | Music | 30 | 23.3 |
| 26 | Pharmacy | 30 | 53.3 |
| 27 | Physics | 30 | 73.3 |
| 28 | Psychology | 30 | 50.0 |
| 29 | Public_Health | 30 | 56.7 |
| 30 | Sociology | 30 | 33.3 |
| | **Overall (macro avg)** | **900** | **48.4** |

계산식: `Overall = mean(30개 과목 accuracy)`. 정답 수 기준으로는 **436/900 = 48.4%**이다.

참고로 외부 judge 없이 rule parser만 적용하면 **217/900 = 24.1%**이며, unresolved가 596개였다. Local judge는 이 중 389개를 해석하고 정답 219개를 복구했으며, 207개는 unresolved로 남겼다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---:|
| [공식 (Qwen3-VL Technical Report)](https://arxiv.org/abs/2511.21631) | 67.4 |
| 우리 재현 결과 | 48.4 |
| 차이 (Δ) | -19.0%p |

우리 재현 결과는 local-judge final이다. Rule-only 24.1%는 unresolved가 지나치게 많아 공식 수치와의 주 비교값으로 사용하지 않았다.

## 7. 격차 분석

측정 점수는 공식 67.4보다 19.0%p 낮았다. 현재 실행에서 rule parser가 596/900개(66.2%)를 unresolved로 분류했고, local judge가 이 중 389개를 해석하여 정답 219개를 복구했다. 그 결과 정확도는 rule-only 24.1%에서 judge-final 48.4%로 증가했으므로 출력 형식과 파싱이 주요 측정 병목이다. 응답 길이 상위 180개에서도 fallback 89.4%, rule 정확도 6.1%, judge-final 정확도 36.7%로 같은 경향이 나타났다. 남은 격차의 가능한 원인은 공식 GPT-3.5 judge 대신 Llama-3.1-8B를 사용한 점, retry/random fallback을 생략한 점, 공식보다 작은 9,048 context와 2,048 generation budget, seed 42 사용이다. Judge-final 기준 Tech and Engineering 43.3%, Diagnostics and Laboratory Medicine 26.7%로 낮아 출력 형식 외에도 도면·전문 영상·계산 추론의 취약성이 관찰된다.

## 8. 기타 특이사항 / 한계 (Optional)

- **Runtime:** Qwen vLLM inference 세션은 00:15:53, 기록된 평균 요청 시간은 0.961초/문항이었다. Category/subject별 시간은 [`summary.json`](../results/00_mmmu_vllm/official-penalty-1p5/qwen3-vl-4b-instruct/summary.json)에 저장하였다. Sample별 기록은 로컬 `predictions.jsonl`에 생성되며 용량 때문에 Git에서는 제외한다.
- **취약 영역:** local-judge final 기준 Tech and Engineering 43.3%가 가장 낮았고, Diagnostics and Laboratory Medicine이 26.7%였다. 실제 오답에는 병리 영상의 단정적 오인, resource-allocation graph 관계 반전, 항공사진 계산의 단위 변환 실패, 장문 계산 후 최종 답 누락이 있었다.
- **다음 단계:** 짧은 reasoning과 `Answer: X` 형식, 더 긴 generation budget을 먼저 ablation한 뒤, 약한 기술·의료 분야의 visual grounding 및 계산 데이터를 이용한 LoRA/QLoRA SFT를 계획한다.
- **Peak VRAM 측정:** vLLM 전체 실행에서 장치 전체 peak 39.412 GiB가 측정되었다. 이는 48GB GPU에서 `gpu_memory_utilization=0.8`로 예약한 KV cache를 포함하므로 모델의 최소 요구 VRAM이 아니다. RTX 4090에서는 동일 비율 기준 약 19.2 GiB 예산으로 동작한다. 상세 기록은 Qwen [`peak_vram.json`](../results/00_mmmu_vllm/official-penalty-1p5/qwen3-vl-4b-instruct/peak_vram.json)에 저장하였다.
