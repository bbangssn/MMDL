# Assignment 01: MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 13조
- **팀원**: 강영선, 김윤나, 송민수, 임희진
- **작성일**: 2026-09-28 (2026-10-13 결과 반영 및 재작성)
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
| 사용 GPU | NVIDIA RTX PRO 5000 Blackwell, 47.23 GiB |
| 실측 peak VRAM | Qwen vLLM: 39.412 GiB (KV cache 예약 공간 포함); local judge: 16.041 GiB |
| 총 소요 시간 | Qwen vLLM 00:15:53; strict local judge 00:02:16 (모델 로딩 포함); 총 약 00:18:09 |
| 의존성 | [`code/requirements.txt`](../code/requirements.txt) |

`run_mmmu_baseline.py`는 Qwen vLLM 추론과 local judge를 서로 다른 프로세스로 순서대로 실행한다. 실행 중 결과와 로그를 갱신하며, 중단된 경우 저장된 sample 이후부터 재개한다.

환경 상세: Python 3.14.7, vLLM 0.30.0, Datasets 5.0.1, Accelerate 1.14.0, qwen-vl-utils 0.0.14, RAM 187.85 GiB, OS Linux-6.8.0-138-generic-x86_64-with-glibc2.35.

실험 환경에 GPU가 다수 존재하므로 `tensor_parallel_size=1`과 `CUDA_VISIBLE_DEVICES=0`으로 단일 GPU만 사용하였다. 모델과 데이터 revision은 코드에 명시했으며 결과의 `config.json`과 `environment.json`에도 저장하였다.

## 2. 프롬프트

**Multiple-choice 문항 입력 prompt**:

```text
Question: {question}
Options:
A. {option_A}
B. {option_B}
...
Please select the correct answer from the options above.
```

**Short-answer 문항 입력 prompt**:

```text
Question: {question}
```

이미지는 Qwen chat template의 image content로 질문과 함께 전달하였다. 모델 입력 prompt는 [Qwen3-VL 공식 MMMU evaluation](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)의 구성을 따랐다.

### Local LLM judge 프롬프트

Rule parser가 해석하지 못한 응답은 `meta-llama/Llama-3.1-8B-Instruct`로 판정하였다. 공식 judge prompt를 그대로 사용했을 때 Llama가 설명문을 생성하여 unresolved가 늘어나는 문제가 있어, 판정 목적은 유지하면서 출력 형식을 한 글자로 제한한 다음 prompt를 사용하였다.

```text
You are a classification function. Match the proposed answer to the closest option.
Return Z only if none of the options match.

OUTPUT RULES:
- Return exactly one uppercase letter.
- Do not explain your reasoning.
- Do not repeat the question, options, or answer.
- Do not add punctuation, formatting, or any other words.
- The only valid outputs are: {valid_labels}

Question: {question}
Options:
{options}
Proposed answer: {answer}

Output:
```

- **Judge 설정**: greedy decoding, `max_new_tokens=8`, `max_model_length=9048`, 1회 호출.
- **선택 이유**: 공식 평가는 `gpt-3.5-turbo-0125`를 사용하지만, API 비용과 외부 서비스 의존성을 줄이기 위해 로컬 Llama-3.1-8B로 대체하였다.
- **설정 검증**: strict prompt는 fallback 596건 중 594건(99.7%)을 해석하였다. `max_new_tokens=128` 조건과 모든 판정 label이 같았으므로 8-token 제한이 결과를 바꾸지 않았다. GPT-3.5 Turbo와 두 judge가 모두 해석한 442건에서는 label이 409건, **92.5%** 일치하였다.

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

`temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.0`, `presence_penalty=1.5`는 [Qwen3-VL 공식 MMMU evaluation README](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/README.md)의 Instruct recipe를 사용하였다. 재실행의 일관성을 위해 seed는 42로 고정하였다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---:|
| `max_model_length` | 9,048 (입력+출력 합계) |
| `max_new_tokens` | 2,048 |
| `min_pixels` | 1,003,520 = 1280 × 28² |
| `max_pixels` | 4,014,080 = 5120 × 28² |

단일 GPU에서 전체 문항을 안정적으로 완료하도록 context와 생성 길이를 제한하였다. 이미지 pixel budget은 Qwen 공개 평가 설정을 참고하였다.

## 4. 채점(파싱) 방식

- **Rule parser**: Qwen3-VL 공개 MMMU 평가 코드의 option extractor를 따라, 응답에서 유일한 option letter 또는 option text를 찾는다. 후보가 없거나 여러 개이면 unresolved로 남긴다.
- **Short-answer 처리**: Ground-truth answer를 A, `Other Answers`를 B로 둔 판정 문제로 변환한다.
- **Fallback judge**: Rule parser가 처리하지 못한 문항에 strict local judge를 greedy decoding으로 1회 적용한다.
- **Random fallback**: Judge도 해석하지 못한 문항은 seed 42의 random fallback으로 답을 선택한다.
- **공식 구현과의 차이**: `gpt-3.5-turbo-0125` 대신 local Llama-3.1-8B와 strict prompt를 사용하며, 최대 25회 재질의 대신 1회만 호출하였다.

## 5. 결과

아래 표는 rule parser, strict local judge, 최종 seeded random fallback을 차례로 적용한 결과이다. 모든 subject가 30문항이므로 macro average와 900문항 micro accuracy가 같다.

| No. | Subject | Data Num | Acc (%) |
|---:|---|---:|---:|
| 1 | Accounting | 30 | 63.3 |
| 2 | Agriculture | 30 | 50.0 |
| 3 | Architecture_and_Engineering | 30 | 46.7 |
| 4 | Art | 30 | 60.0 |
| 5 | Art_Theory | 30 | 86.7 |
| 6 | Basic_Medical_Science | 30 | 76.7 |
| 7 | Biology | 30 | 56.7 |
| 8 | Chemistry | 30 | 50.0 |
| 9 | Clinical_Medicine | 30 | 66.7 |
| 10 | Computer_Science | 30 | 56.7 |
| 11 | Design | 30 | 73.3 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.7 |
| 13 | Economics | 30 | 83.3 |
| 14 | Electronics | 30 | 73.3 |
| 15 | Energy_and_Power | 30 | 53.3 |
| 16 | Finance | 30 | 76.7 |
| 17 | Geography | 30 | 66.7 |
| 18 | History | 30 | 66.7 |
| 19 | Literature | 30 | 80.0 |
| 20 | Manage | 30 | 66.7 |
| 21 | Marketing | 30 | 80.0 |
| 22 | Materials | 30 | 33.3 |
| 23 | Math | 30 | 60.0 |
| 24 | Mechanical_Engineering | 30 | 46.7 |
| 25 | Music | 30 | 36.7 |
| 26 | Pharmacy | 30 | 66.7 |
| 27 | Physics | 30 | 80.0 |
| 28 | Psychology | 30 | 76.7 |
| 29 | Public_Health | 30 | 80.0 |
| 30 | Sociology | 30 | 60.0 |
|  | **Overall** | **900** | **63.7** |

최종 정답 수는 **573/900 = 63.7%**이다. Rule parser만으로는 217/900(24.1%)이 정답이었으며 596건이 fallback 대상이었다. Strict local judge는 596건 중 594건을 해석하고 정답 356건을 복구하였다. 남은 unresolved는 2건이며 random fallback으로 추가 정답이 발생하지 않았다.

대분야별 정확도는 Business 74.0%, Humanities and Social Science 70.8%, Health and Medicine 65.3%, Art and Design 64.2%, Science 62.7%, Tech and Engineering 51.4%였다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---:|
| [공식 (Qwen3-VL Technical Report)](https://arxiv.org/abs/2511.21631) | 67.4% |
| 우리 재현 결과 | 63.7% |
| 차이 (Δ) | -3.7%p |

## 7. 격차 분석

재현 정확도는 63.7%로 공식 수치 67.4%보다 3.7%p 낮았다. 동일한 checkpoint와 validation split을 사용했지만 judge와 일부 추론 조건은 공식 평가와 다르다. 공식 `gpt-3.5-turbo-0125` 대신 strict-prompt Llama-3.1-8B를 사용했으며, 공개 재현 설정의 긴 context와 생성 예산 대신 단일 GPU에 맞춘 `max_model_length=9048`, `max_new_tokens=2048`을 사용하였다. Sampling seed도 공식 공개값과 다르다. Strict judge와 GPT judge의 label 일치율은 92.5%로 높았지만 완전히 같지는 않았으므로, 남은 격차에는 judge 판정 차이와 생성 예산 및 seed 차이가 함께 반영된 것으로 판단하였다.

## 8. 기타 특이사항 / 한계 (Optional)

- **Runtime:** Qwen vLLM 추론은 00:15:53, 평균 요청 시간은 0.961초/문항이었다. Strict judge의 순수 생성 시간은 79.2초, 평균 0.133초/fallback이었고 모델 로딩을 포함한 실행 시간은 135.8초였다.
- **Peak VRAM:** Qwen vLLM은 KV cache 예약 공간을 포함하여 39.412 GiB, local judge는 16.041 GiB가 측정되었다. 두 모델은 별도 프로세스에서 순차 실행하므로 동시에 합산되는 사용량은 아니다.
- **취약 영역:** 대분야 중 Tech and Engineering이 51.4%로 가장 낮았다. 과목별로는 Materials 33.3%, Diagnostics and Laboratory Medicine 36.7%, Music 36.7%가 가장 낮았다.
- **Modality ablation:** MMMU validation의 동일한 200문항에서 Full 66.5%, Text-only 50.5%, Shuffled-image 48.0%, Blank-image 48.0%였다. 이미지 정보를 제거하거나 무관한 이미지로 바꾸면 16.0~18.5%p 하락하여, 전체적으로는 시각 정보가 성능에 실질적으로 기여하였다.
- **선지 순서 안정성:** 객관식 847문항을 서로 다른 두 순열로 추가 평가했을 때 세 실행 모두 같은 내용의 선지를 고른 비율은 62.7%, 두 실행만 같은 비율은 29.2%, 세 응답이 모두 다른 비율은 8.1%였다. 세 번 모두 정답인 문항은 49.5%였다. 4지선다 606문항에서는 정답 분포를 보정한 뒤에도 오답에서 A를 10.2~12.6%p 과다 선택하고 D를 6.9~12.6%p 과소 선택하는 경향이 반복되었다.
- **해상도 probe:** MMMU-Pro 120문항의 paired 실험에서 Direct 49.2%, Low 47.5%, High 44.2%였다. Direct 대비 High의 차이는 -5.0%p였고 bootstrap 95% CI는 -10.8~+0.8%p로 0을 포함하였다. 따라서 확대된 표본에서는 높은 pixel budget이 정확도를 높인다는 근거를 확인하지 못했다. High 설정은 최대 pixel budget만 높이므로 모든 이미지가 실제로 확대되는 것은 아니다.
- **분석 한계:** Modality ablation은 200문항, 해상도 실험은 MMMU-Pro 120문항의 부분집합 결과이다. 보조 실험은 병목 후보를 찾는 근거이며 전체 benchmark에 그대로 일반화하지 않는다.
