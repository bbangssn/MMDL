# Assignment 01: MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 13조
- **팀원**: 강영선, 김윤나, 민희진, 송민수
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
| 사용 GPU | NVIDIA RTX PRO 5000 Blackwell, 47.23 GiB |
| 실측 peak VRAM | Qwen vLLM: 39.412 GiB (KV cache 예약 공간 포함); local judge: 16.041 GiB |
| 총 소요 시간 | Qwen vLLM 00:15:53 (900문제), local judge 추가 00:16:57; 총 00:32:50 |
| 의존성 | [`code/requirements.txt`](../code/requirements.txt) |
| 실행 커맨드 | `python code/run_mmmu_baseline.py` |

`run_mmmu_baseline.py`는 Qwen vLLM 추론과 local judge를 서로 다른 프로세스로 순서대로 실행한다. 기본값은 지정된 Hugging Face 모델과 데이터를 사용한다. 실행 중에는 로그를 주기적으로 갱신하며, 프로세스가 중단되어도 같은 명령으로 재개할 수 있다.

환경 상세: Python 3.14.7, vLLM 0.30.0, Datasets 5.0.1, Accelerate 1.14.0, qwen-vl-utils 0.0.14, RAM 187.85 GiB, OS Linux-6.8.0-138-generic-x86_64-with-glibc2.35.

추론 백엔드는 vLLM을 사용하였다. 실험 환경에 GPU가 다수개 존재했으므로 `tensor_parallel_size=1`과 `CUDA_VISIBLE_DEVICES=0`으로 단일 GPU만 사용하도록 했다. 모델과 데이터 revision은 코드에 명시되어 있으며, 실행 결과의 `config.json`과 `environment.json`에도 함께 저장된다.

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
- **선택 이유**: 공개 성능과 비교할 때 prompt 차이를 줄이고, 임의의 정답 형식 지시로 점수를 변경하지 않기 위해 공식 공개 구현의 문구를 유지하였다.


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

- **출처**: [Qwen3-VL 공식 MMMU evaluation README](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/README.md)와 [공식 evaluation reproduction 설정](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction)의 Instruct recipe를 사용하였다. `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.0`, `presence_penalty=1.5`는 공식 값과 같다. 측정 실행의 seed는 재실행을 고정하기 위해 42로 설정했다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---:|
| `max_model_length` | 9,048 (입력+출력 합계) |
| `max_new_tokens` | 2,048 |
| `min_pixels` | 1,003,520 = 1280 × 28² |
| `max_pixels` | 4,014,080 = 5120 × 28² |

**선택 근거:** 단일 GPU에서 900문항을 안정적으로 완료하기 위해 context와 생성 길이를 제한하였다. 이미지 pixel budget은 Qwen 공개 평가 설정을 참고하였다.

## 4. 채점(파싱) 방식

- **사용한 로직**: [Qwen3-VL 공식 MMMU evaluation](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)의 rule-based option extractor와 judge prompt를 참고해 구현하였다.
- **rule parser**: 거부 응답을 `Z`로 처리한 뒤, 구두점을 제거한 token에서 유일한 option letter를 찾는다. letter가 모호하면 응답에 유일하게 등장하는 option text를 찾는다. 후보가 없거나 여러 개면 unresolved로 남긴다. Qwen3-VL 공식 평가 코드의 방식을 사용하였다.
- **Short-answer 처리**: Qwen 공개 전처리와 같이 ground-truth answer를 A, `Other Answers`를 B로 놓은 판정 문제로 변환한다.
- **Fallback judge**: rule parser가 처리하지 못한 문항에는 로컬 `meta-llama/Llama-3.1-8B-Instruct`를 greedy decoding으로 1회 적용하였다. Judge도 처리하지 못한 문항은 공식 구현과 같이 random fallback으로 답을 선택하였다.
- **공식 구현과의 차이**: 공식 judge인 `gpt-3.5-turbo-0125` 대신 로컬 `Llama-3.1-8B`를 사용하고, 최대 25회 재질의 대신 1회만 호출하였다.

## 5. 결과

아래 표는 rule parser 결과에 local judge 판정과 최종 seeded random fallback을 합친 결과이다. 모든 subject가 30문항이므로 macro average와 900문항 micro accuracy가 같다.

| No. | Subject | Data Num | Acc |
|---|---|---:|---:|
| 1 | Accounting | 30 | 53.3 |
| 2 | Agriculture | 30 | 43.3 |
| 3 | Architecture_and_Engineering | 30 | 40.0 |
| 4 | Art | 30 | 46.7 |
| 5 | Art_Theory | 30 | 73.3 |
| 6 | Basic_Medical_Science | 30 | 56.7 |
| 7 | Biology | 30 | 36.7 |
| 8 | Chemistry | 30 | 50.0 |
| 9 | Clinical_Medicine | 30 | 50.0 |
| 10 | Computer_Science | 30 | 63.3 |
| 11 | Design | 30 | 46.7 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 33.3 |
| 13 | Economics | 30 | 63.3 |
| 14 | Electronics | 30 | 60.0 |
| 15 | Energy_and_Power | 30 | 36.7 |
| 16 | Finance | 30 | 63.3 |
| 17 | Geography | 30 | 43.3 |
| 18 | History | 30 | 53.3 |
| 19 | Literature | 30 | 53.3 |
| 20 | Manage | 30 | 56.7 |
| 21 | Marketing | 30 | 73.3 |
| 22 | Materials | 30 | 43.3 |
| 23 | Math | 30 | 53.3 |
| 24 | Mechanical_Engineering | 30 | 60.0 |
| 25 | Music | 30 | 26.7 |
| 26 | Pharmacy | 30 | 53.3 |
| 27 | Physics | 30 | 76.7 |
| 28 | Psychology | 30 | 56.7 |
| 29 | Public_Health | 30 | 60.0 |
| 30 | Sociology | 30 | 36.7 |
| | **Overall (macro avg)** | **900** | **52.1** |

계산식: `Overall = mean(30개 과목 accuracy)`. 정답 수 기준으로는 **469/900 = 52.1%**이다.

참고로 외부 judge 없이 rule parser만 적용하면 **217/900 = 24.1%**이며, unresolved가 596개였다. Local judge는 이 중 389개를 해석하고 정답 219개를 복구했다. 남은 207개에는 seeded random fallback을 적용해 33개가 정답이 되었으며, 총 252개를 복구했다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---:|
| [공식 (Qwen3-VL Technical Report)](https://arxiv.org/abs/2511.21631) | 67.4 |
| 우리 재현 결과 | 52.1 |
| 차이 (Δ) | -15.3%p |

우리 재현 결과는 local-judge final이다.

## 7. 격차 분석

측정 정확도는 52.1%로 공식 수치 67.4%보다 15.3%p 낮았다. 모델과 validation split은 같지만 평가 조건이 완전히 동일하지 않기 때문에, 이 차이를 곧바로 모델 성능의 재현 실패로 단정하기는 어렵다.

주요 차이 중 하나는 answer extraction 방식이다. 공식 평가는 `gpt-3.5-turbo-0125` judge를 최대 25회 호출하지만, 본 실험은 Llama-3.1-8B를 greedy decoding으로 1회만 호출하였다. 또한 공식 공개 설정의 `max_model_len=128000`, `max_new_tokens=32768`, seed 3407과 달리 본 실험은 단일 GPU 환경에 맞춰 각각 9,048, 2,048, seed 42를 사용하였다. 짧은 생성 길이는 장문 응답의 최종 답 누락에 영향을 줄 수 있고, sampling seed 차이도 결과 변동의 원인이 될 수 있다. 따라서 15.3%p의 격차는 judge와 생성 예산, seed 등 평가 설정 차이가 함께 반영된 결과로 해석하였다.

## 8. 기타 특이사항 / 한계 (Optional)

- **Runtime:** Qwen vLLM inference는 00:15:53, 평균 요청 시간은 0.961초/문항이었다.
- **취약 영역:** 최종 category 기준 Art and Design 48.3%, Tech and Engineering 49.5%가 낮았고, subject 중에서는 Music 26.7%, Diagnostics and Laboratory Medicine 33.3%가 낮았다. 실제 오답에는 병리 영상의 단정적 오인, resource-allocation graph 관계 반전, 항공사진 계산의 단위 변환 실패, 장문 계산 후 최종 답 누락이 있었다.
- **다음 단계:** 낮은 성능을 보인 분야의 오류 사례를 바탕으로 fine-tuning 데이터를 구성한다.
- **Peak VRAM 측정:** vLLM 실행에서 KV cache 예약 공간을 포함해 39.412 GiB가 측정되었다.
