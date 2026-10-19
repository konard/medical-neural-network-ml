# Review of the escalation ladder against current best practice

Issue #23 asks us to check that every step of the research ladder follows
industry best practice, and to fix what is wrong. This page lists every
claim we checked, the primary source, and the verdict. All sources were
read on 2026-10-19.

Files checked:

* [if-models-fail.md](../issue-1/if-models-fail.md) §2–§7 (the ladder);
* [research-notes/training-and-data.md](../issue-1/research-notes/training-and-data.md)
  (its source notes);
* [research-plan.md](../issue-1/research-plan.md) and
  [ci-and-gpu-runner.md](../issue-1/ci-and-gpu-runner.md) (the test
  escalation).

Verdicts: **correct**, **wrong**, **outdated**, **imprecise**, **missing**.
Every correction marked "fixed" is now in the file.

## 1. Claims checked

### Step 1: prompting, RAG, tools

| Location | The doc says | The primary source says | Verdict | Correction |
| --- | --- | --- | --- | --- |
| if-models-fail §2, step 1 | Medprompt took GPT-4 to 90.2% on MedQA without training | 90.2% with 5 few-shot examples and a 5-run ensemble; 90.6% with 20 examples and 11 runs. [arXiv:2311.16452](https://arxiv.org/html/2311.16452) | correct | – |
| if-models-fail §2, step 1 | MedRAG gave up to +18% | "up to 18%" over chain-of-thought, 7,663 questions, 41 combinations. [arXiv:2402.13178](https://arxiv.org/abs/2402.13178) | correct | – |

### Step 2: LoRA / QLoRA

| Location | The doc says | The primary source says | Verdict | Correction |
| --- | --- | --- | --- | --- |
| if-models-fail §2, step 2 | Adapters are a few MB | Rank 16 on attention and MLP trains 29.2M (4B) and 112M (27B) weights ([measurements.md §7](../issue-1/measurements.md#7-fine-tuning-memory-medbench-estimate---finetune)); in bf16 that is 58 MB and 224 MB | wrong | Fixed: "rank-16 adapters are about 58 MB (4B) to 224 MB (27B) in bf16" |
| if-models-fail §2, step 2 | Google's MedGemma notebook: QLoRA, 1 epoch, one 40 GB GPU | The notebook is for `google/medgemma-4b-it`; NF4, r=16, alpha=16, all linear layers, 1 epoch, lr 2e-4; "at least 40 GB"; it also sets `modules_to_save=["lm_head","embed_tokens"]`. With hidden size 2560 and a 262,208-token vocabulary each of these is about 0.67B weights. [notebook](https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb) | imprecise | Fixed: say it is the 4B notebook, and that it trains the embedding and output layers, so its adapter is GB-sized |
| training-and-data §3.4 item 8 | The recipe, without `modules_to_save` | As above | missing | Fixed: added `modules_to_save` and why it matters |
| if-models-fail §7 | QLoRA 4B "about 5 GB"; QLoRA 27B "about 20 GB, so one L4 or RTX 4090"; LoRA 27B "about 60 GB" | The repository's own formula: QLoRA 4B 6.5–13.5 GB, QLoRA 27B 20.1–28.2 GB, LoRA 27B 60.8–68.9 GB at 512–2 048 tokens; 24 GB for 27B is only a hypothesis ([measurements.md §7](../issue-1/measurements.md#7-fine-tuning-memory-medbench-estimate---finetune)) | outdated | Fixed: the formula ranges, and "24 GB only with short sequences or a chunked loss; a hypothesis to measure first" |
| if-models-fail §7 | LoRA of 27B on 10M tokens is about 1.1 H100-hours | 6 × 27e9 × 1e7 = 1.6e18 FLOP; at 40% of the 989 TFLOPS bf16 peak that is 1.1 hours | correct | – |
| training-and-data §5.1 | Unsloth: Apache-2.0, "2x faster, 70% less VRAM" | Same on the README; the Studio UI is AGPL-3.0. [unslothai/unsloth](https://github.com/unslothai/unsloth) | correct | – |
| training-and-data §5.1 | torchtune is no longer maintained | "no longer actively maintained". [pytorch/torchtune](https://github.com/pytorch/torchtune) | correct | – |
| measurements §7 (context) | A chunked loss (Liger Kernel, Unsloth) may let 27B QLoRA fit 24 GB | Liger has a fused linear cross-entropy, claims +20% throughput and -60% memory, BSD-2, works with TRL. [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | correct | – |

### Step 3: full fine-tuning, DPO, GRPO, distillation

| Location | The doc says | The primary source says | Verdict | Correction |
| --- | --- | --- | --- | --- |
| if-models-fail §2, step 3a | 16 bytes per weight: about 80 GB for 4B, 450 GB for 27B | bf16 weights and gradients plus fp32 master weights and two AdamW moments = 16 bytes; matches the formula in measurements.md §7 | correct | – |
| training-and-data §3.1 | Full FT needs "roughly 4x model size or more" | 16 bytes per weight is 8x the 2-byte bf16 model size, before activations | imprecise | Fixed: "about 16 bytes per weight ... about 8x the bf16 model size, plus activations" |
| training-and-data §3.2, DPO | "can over-optimize" | DPO tends to make answers longer rather than better. [Park et al. 2024, arXiv:2403.19159](https://arxiv.org/abs/2403.19159) | missing | Fixed: added length exploitation and "track answer length" [S60] |
| training-and-data §3.2, KTO | Matched or beat preference methods at 1B–30B "(unverified, secondary source)" | Abstract: "matches or exceeds the performance of preference-based methods at scales from 1B to 30B". [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) | correct | Fixed: removed "unverified" |
| training-and-data §3.2, GRPO | Pitfalls: checker, reward hacking, slow | The original GRPO loss inflates answer length, most for wrong answers ([Dr. GRPO, arXiv:2503.20783](https://arxiv.org/abs/2503.20783)). TRL `GRPOConfig` now defaults to `loss_type="dapo"` and `beta=0.0` ([TRL docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)) | missing | Fixed: added the length bias and the Dr. GRPO / DAPO loss [S61] |
| if-models-fail §2, step 3c | MedVLM-R1: 600 samples, 55% → 78% | 600 VQA samples, 2B model, 55.11% → 78.22%. [arXiv:2502.19634](https://arxiv.org/abs/2502.19634) | correct | – |
| if-models-fail §2, step 3d | MedGemma's text skills came from distillation | The report: pre-training distilled from a large instruction-tuned teacher; RL then generalized better than SFT. [arXiv:2507.05201](https://arxiv.org/html/2507.05201v4) | correct | – |
| training-and-data §3.4 | MedGemma SFT: lr 1e-7 to 1e-6, one epoch; EHRQA 86.3 → 93.6; SIIM 59.7 → 71.5; CRC100k 32.8 → 94.5 | Same numbers in the report | correct | – |
| training-and-data §5.1, TRL | SFT, DPO, KTO, GRPO, RLOO, reward, distillation; ORPO and CPO experimental; no PPO | TRL v1.16 docs: stable SFT, DPO, KTO, GRPO, RLOO, Reward, Distillation; ORPO, CPO, Online DPO, GKD experimental; no PPO. [TRL](https://huggingface.co/docs/trl/index) | correct | – |

### Step 4: continued pre-training

| Location | The doc says | The primary source says | Verdict | Correction |
| --- | --- | --- | --- | --- |
| if-models-fail §2, step 4 | MEDITRON: 48.1B tokens on Llama 2 | 48.1B tokens including 400M of replay data. [epfl-nlp/meditron](https://github.com/epfl-nlp/meditron), [arXiv:2311.16079](https://arxiv.org/abs/2311.16079) | correct | – |
| training-and-data §3.4 | MedGemma mixed medical data at a small weight | Medical data at 2% (encoder) and 10% (decoder) weight, about 5 epochs; this is the replay practice that limits forgetting | correct | – |

### Step 5: from scratch

| Location | The doc says | The primary source says | Verdict | Correction |
| --- | --- | --- | --- | --- |
| if-models-fail §3 | About 20 tokens per parameter (Chinchilla); 7B needs 140B tokens | 70B on 1.4T tokens, data and size scaled equally. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) | correct, but missing context | Fixed: added that this is a minimum; Llama 3.1 8B used about 15T tokens and Gemma 3 4B 4T |
| if-models-fail §2, step 5 | About 61 000 H100-hours ($150–250k) for a 27B LLM | 27B × 20 = 540B tokens; 6ND = 8.7e22 FLOP; at 40% of peak that is about 61k hours. Gemma 3 27B was trained on 14T tokens, about 26x more. [arXiv:2503.19786](https://arxiv.org/html/2503.19786) | imprecise | Fixed: labelled as the Chinchilla minimum and added the Gemma 3 figure |
| training-and-data §4.1 | Llama 3.1: 39.3M H100 hours, 8B 1.46M; source is a mirror | The official card gives the same numbers and about 15T tokens. [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | correct, weak source | Fixed: S21 now points to the official card; removed "secondary mirrors" |

### Data, augmentation, privacy

| Location | The doc says | The primary source says | Verdict | Correction |
| --- | --- | --- | --- | --- |
| training-and-data §6.3, if-models-fail §4 | 18 HIPAA Safe Harbor identifiers, 3-digit ZIP rule, ages over 89 | Same list in 45 CFR 164.514(b)(2); it also covers identifiers of relatives, employers and household members. [eCFR](https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514) | correct | – |
| if-models-fail §5 | RoentGen added about 5% | 5% with synthetic plus real data, 3% with synthetic only. [arXiv:2211.12737](https://arxiv.org/abs/2211.12737) | correct | – |
| if-models-fail §5 | Up to 4 epochs nearly as good as new data | "up to 4 epochs ... negligible changes to loss". [arXiv:2305.16264](https://arxiv.org/abs/2305.16264) | correct | – |
| if-models-fail §4 | Patient-level splits, external hospital test, fairness checks | The IMDRF GMLP principles ask for the same: representative data and test sets independent of training data. [IMDRF N88](https://imdrf.org/sites/default/files/2025-02/IMDRF_AIML%20WG_GMLP_N88%20Final.pdf) | correct | – |

### Evaluation, safety, regulation

| Location | The doc says | The primary source says | Verdict | Correction |
| --- | --- | --- | --- | --- |
| if-models-fail §6 | PCCP final guidance December 2024 | Final; first issued December 2024; the current page is dated August 2025 (docket FDA-2022-D-2628). [FDA](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence) | imprecise | Fixed: "final guidance December 2024, current version August 2025" (also in training-and-data §8.3 and S57) |
| training-and-data §8.3 | AI-enabled device lifecycle guidance is still a draft (January 2025) | Still marked Draft on the FDA page | correct | – |
| if-models-fail §6, training-and-data §8.3 | AI Act high-risk rules for medical devices apply from 2 August 2028 (2026 Digital Omnibus); "secondary law-firm sources" | Regulation (EU) 2026/1744 of 8 July 2026, in force 27 July 2026; new Art. 113: Annex III from 2 Dec 2027, Annex I (medical devices) from 2 Aug 2028; Art. 50(2) marking of generated content until 2 Dec 2026 for systems already on the market. [Official Journal](https://eur-lex.europa.eu/eli/reg/2026/1744/oj) | correct, weak source | Fixed: linked the Official Journal, added the Art. 50(2) grace period, removed the caveat |
| training-and-data §5.5 | Evaluation tools: lm-eval-harness, VLMEvalKit, MedEvalKit, MultiMedEval | HealthBench: 5,000 conversations, 262 physicians, 48,562 rubric criteria ([arXiv:2505.08775](https://arxiv.org/abs/2505.08775)). MedHELM: 121 tasks, LLM jury ([arXiv:2505.23802](https://arxiv.org/abs/2505.23802)) | missing | Fixed: added both rows [S62] |
| training-and-data §8.1 | LLM-as-judge "with checks" | Judges show position, verbosity and self-preference bias; over 80% agreement with humans. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) | imprecise | Added the concrete checks to if-models-fail §6 |

### Test escalation

| Location | The doc says | The primary source says | Verdict | Correction |
| --- | --- | --- | --- | --- |
| ci-and-gpu-runner §1 | CPU smoke test runs on hosted `ubuntu-24.04` | `.github/workflows/model-tests.yml` uses `ubuntu-26.04` for every hosted job; GitHub lists `ubuntu-26.04` with 4 vCPU / 16 GB for public repos. [GitHub docs](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) | wrong | Fixed: `ubuntu-26.04` |
| ci-and-gpu-runner §1 | Four layers of tests | The workflow also has `ocr-baseline` (Tesseract) and `unimedvl-cpu` (upstream code, Python 3.14, no weights) | outdated | Fixed: six layers, two new table rows, two nodes in the diagram |
| ci-and-gpu-runner §1 | GPU tests: MedGemma 27B or 4B | `gpu-model` also accepts `unimedvl` | outdated | Fixed: "or UniMedVL" |
| research-plan Phase 1, step 4 | UniMedVL runs in a Python 3.10 environment | `scripts/medbench-model-run.sh` (`UNIMEDVL_PYTHON=python3.14`), `scripts/unimedvl-requirements.txt` (`--python-version 3.14`) and the workflow all use 3.14 | wrong | Fixed: "Python 3.14 environment" |
| ci-and-gpu-runner §4 | Hosted T4 GPU runner at $0.052/min | `linux_4_core_gpu` at $0.052/min. [GitHub pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing) | correct | – |
| ci-and-gpu-runner §5 | actions/runner v2.338.0 | Latest release is v2.338.0. [actions/runner](https://github.com/actions/runner/releases) | correct | – |

Out of scope, not changed: [docs/hardware-workflows.md](../../hardware-workflows.md)
still says "Python 3.13 isolated environment" for UniMedVL; the scripts use
3.14.

## 2. Missing practices, now added

Added as one short table at the end of
[if-models-fail.md §6](../issue-1/if-models-fail.md#6-evaluation-safety-and-rules-8):

| Practice | Why | Source |
| --- | --- | --- |
| Contamination check before trusting public scores | Models may have seen MedQA-style test items | [Golchin and Surdeanu 2023](https://arxiv.org/abs/2308.08493) |
| LLM-as-judge with order swaps, length caps and a doctor-checked sample | Position, verbosity and self-preference bias | [Zheng et al. 2023](https://arxiv.org/abs/2306.05685) |
| Track answer length in DPO and GRPO; use the Dr. GRPO or DAPO loss | Both reward longer answers | [Park et al. 2024](https://arxiv.org/abs/2403.19159), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer) |
| HealthBench and MedHELM next to MedQA | Open-ended, rubric-graded clinical tasks | [HealthBench](https://arxiv.org/abs/2505.08775), [MedHELM](https://arxiv.org/abs/2505.23802) |
| Pinned revisions, seeds, MLflow, DVC | Reproducible runs | training-and-data §5.3 |
| Differential privacy (Opacus) or federated learning for clinic data | Language models can repeat rare training records | [Carlini et al. 2021](https://arxiv.org/abs/2012.07805), [Opacus](https://github.com/meta-pytorch/opacus) |
| Model card; TRIPOD+AI, TRIPOD-LLM, CLAIM 2024, DECIDE-AI | Standard reporting for clinical AI | [Model cards](https://arxiv.org/abs/1810.03993), [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378), [TRIPOD-LLM](https://pubmed.ncbi.nlm.nih.gov/39779929/), [CLAIM](https://pubs.rsna.org/page/ai/claim), [DECIDE-AI](https://pubmed.ncbi.nlm.nih.gov/35585198/) |
| IMDRF Good Machine Learning Practice (10 principles, January 2025) | Regulators' shared baseline | [IMDRF N88](https://imdrf.org/sites/default/files/2025-02/IMDRF_AIML%20WG_GMLP_N88%20Final.pdf) |

Also worth adding later, not added now:

* CONSORT-AI and SPIRIT-AI (Nature Medicine, 2020) if a randomized trial is
  ever run.
* The NIST Generative AI Profile, [NIST AI 600-1](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
  (July 2024), as a voluntary risk checklist.
