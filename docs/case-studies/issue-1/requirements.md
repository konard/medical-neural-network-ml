# Requirements of issue #1 and how each one is met

The issue text is saved in [data/issue-1.txt](data/issue-1.txt). Below,
every sentence that asks for something is one requirement (R1–R16). Each
row says what was done and where.

| # | Requirement (shortened) | What was done | Where |
| --- | --- | --- | --- |
| R1 | A detailed plan for research on teaching a medical neural network | A 6-phase plan with goals, steps, resources, alternatives and "done when" for each phase | [research-plan.md](research-plan.md) |
| R2 | First try existing networks on **synthetic** typical tasks | `medbench` creates synthetic cases, documents and phantom images with exact answers. It runs any model through the same scoring code. MedGemma 4B runs in CI on these tasks | [`src/medbench`](../../../src/medbench), [`.github/workflows/model-tests.yml`](../../../.github/workflows/model-tests.yml), [research-plan.md, Phases 0–1](research-plan.md#phase-0-a-harness-that-tests-any-model-the-same-way-done) |
| R3 | Public-domain or permissive datasets for typical documents and images | More than 40 datasets checked, with licenses, sizes and traps. A permissive starter set per task | [research-notes/datasets.md](research-notes/datasets.md), [research-plan.md, Phase 2](research-plan.md#phase-2-ready-models-on-public-data) |
| R4 | OCR of medical histories | Task `ocr-history`: rendered printed history, scored by CER/WER. OCR models compared | [architectures.md §4](architectures.md#4-other-strong-options), `synthetic_documents.py` |
| R5 | Handwritten prescriptions | Task `ocr-prescription`: handwriting-style rendering and drug + strength recall. RxNorm check plus a human in the loop | same, [images/prescription-handwritten.png](images/prescription-handwritten.png) |
| R6 | X-ray, MRI, ultrasound | Tasks `imaging-xray`, `imaging-mri`, `imaging-ultrasound` with phantoms; sensitivity and specificity | `synthetic_images.py`, [images/](images/) |
| R7 | Initial diagnosis with several likely options | Task `diagnosis`: the model must return a ranked list of 5; scored by top-1/3/5 and MRR. The naive Bayes reference shows the best possible score | `synthetic_cases.py`, [measurements.md §5](measurements.md#5-baseline-scores-without-any-model) |
| R8 | Check `google/medgemma-27b-it` | Architecture, scores, license and memory studied. Runs in the GPU job (`gpu-model: medgemma-27b-it`); needs 18.3 GB in 4-bit | [architectures.md §2](architectures.md#2-medgemma-google-gemma-3-plus-a-medical-eye), [research-notes/models.md §1](research-notes/models.md#1-medgemma-family-google-and-medsiglip) |
| R9 | Check `General-Medical-AI/UniMedVL` | Architecture (Mixture-of-Transformers, image generation), scores and memory studied. No GGUF build, so it needs PyTorch on a 40 GB+ GPU; documented as the next step | [architectures.md §3](architectures.md#3-unimedvl-shanghai-ai-lab-one-model-that-reads-and-draws), [research-notes/models.md §2](research-notes/models.md#2-unimedvl-shanghai-ai-lab-general-medical-ai-group) |
| R10 | "And other similar options" | Lingshu, Hulu-Med, HuatuoGPT-Vision, MedVLM-R1, CheXagent, MAIRA-2, text LLMs, OCR VLMs and non-LLM imaging tools | [architectures.md §4](architectures.md#4-other-strong-options), [research-notes/models.md §3–6](research-notes/models.md) |
| R11 | Measure CPU inference speed | Measured with `llama-bench` and `llama-server` on the dev VM: 4.21 tokens/s answer speed when weights are in RAM, 0.40 with mmap under the 3 GiB limit. Measured end to end on GitHub-hosted runners (two runs): 7.8–11.6 tokens/s, 66–84 s per image, about 6 s per text question. The calculator was tuned to both | [measurements.md](measurements.md), [data/](data/) |
| R12 | If CPU is too slow for CI, GPU versions of **all** tests for a local (self-hosted) Actions runner | CPU CI runs a small smoke set (3 tasks × 2 items). The `gpu` job runs all 6 tasks × 20 items on `[self-hosted, linux, x64, gpu]`, started by hand only. Runner setup and security are documented | [ci-and-gpu-runner.md](ci-and-gpu-runner.md), [`.github/workflows/model-tests.yml`](../../../.github/workflows/model-tests.yml) |
| R13 | All instructions and resources calculated | `medbench estimate` gives memory, speed and time per request for every model and hardware; `--finetune` gives training memory. Results, CI cost and GPU prices are saved | [measurements.md §6–7](measurements.md#6-calculated-speed-and-memory-medbench-estimate), [data/estimates.txt](data/estimates.txt), [data/finetune-estimates.txt](data/finetune-estimates.txt), [research-notes/hardware-and-ci.md](research-notes/hardware-and-ci.md) |
| R14 | A plan for when ready models fail: fine-tuning, training from scratch or from the best base, with a big dataset | A ladder from prompting to training from scratch, with costs, pros and cons, and a concrete plan | [if-models-fail.md](if-models-fail.md), [research-notes/training-and-data.md](research-notes/training-and-data.md) |
| R15 | How to collect a dataset, multiply it and reuse it for more effect | Collection pipeline, legal steps, labeling, splits, augmentation, synthetic data, repeated epochs and balancing. Safe augmentations are implemented and tested | [if-models-fail.md §4–5](if-models-fail.md#4-collecting-a-dataset), [`src/medbench/augment.py`](../../../src/medbench/augment.py) |
| R16 | Explain the architectures of the best solutions, their differences and their effect on results | Diagrams of a generic VLM, MedGemma, UniMedVL and OCR families, and a "design choice → effect" table | [architectures.md](architectures.md) |

Requirements about **how** the work is written and done:

| # | Requirement | How it is met |
| --- | --- | --- |
| H1 | Every detail explained; all alternatives with pros and cons at every stage | Each phase and each step has an "Option / Pros / Cons" table ([research-plan.md](research-plan.md), [if-models-fail.md](if-models-fail.md), [ci-and-gpu-runner.md](ci-and-gpu-runner.md)) |
| H2 | Simplest possible English, readable without background | Short sentences, and terms explained in the [glossary](glossary.md) |
| H3 | Mermaid diagrams where they fit | Diagrams in [README.md](README.md), [research-plan.md](research-plan.md), [architectures.md](architectures.md), [if-models-fail.md](if-models-fail.md), [measurements.md](measurements.md) and [ci-and-gpu-runner.md](ci-and-gpu-runner.md) |
| H4 | Do not overload the development machine | Runs use `nice`, fewer threads than CPUs, time budgets and a memory preflight. The slow local vision run was stopped, not left running ([research-plan.md](research-plan.md#protecting-the-development-machine), [measurements.md §4](measurements.md#4-a-real-end-to-end-run-on-this-machine-stopped)) |
| H5 | Collect the data in `./docs/case-studies/issue-1` and do a deep case study, with online facts | This folder: research notes with sources, raw logs and measurements |
| H6 | List existing components and libraries that solve similar problems | Table below, and details in the research notes |
| H7 | Everything in one pull request | Pull request #2 |

## Existing components and libraries

| Need | Ready components | Used here? |
| --- | --- | --- |
| Run models on CPU/GPU | llama.cpp, Ollama, vLLM, SGLang, Hugging Face Transformers | llama.cpp (`scripts/medbench-model-run.sh`); Transformers backend in `medbench` |
| Evaluate medical models | MedEvalKit, VLMEvalKit, lmms-eval, MultiMedEval, HealthBench | Planned for Phase 2 |
| OCR | Tesseract, PaddleOCR, TrOCR, olmOCR-2, Chandra, dots.ocr, Marker | Tesseract backend; VLM OCR via the OpenAI-compatible backend |
| Drug names | RxNorm (NLM), RapidFuzz for fuzzy matching | Planned for Phase 3 |
| Medical imaging | MONAI, nnU-Net, TotalSegmentator, TorchXRayVision, MedSAM2, TorchIO | Planned as baselines and augmentation |
| Synthetic data | Synthea, RoentGen, UniMedVL generation, our own generators | Own generators in `medbench` |
| Fine-tuning | Hugging Face TRL + PEFT, Unsloth, Axolotl, LLaMA-Factory, ms-swift | Planned for Phase 4 |
| Labeling | Label Studio, CVAT, MONAI Label | Planned for dataset collection |
| Privacy | Presidio, `deid` tools, DICOM anonymizers; Flower and NVIDIA FLARE for federated learning | Planned for dataset collection |
| Data versioning | DVC, Hugging Face datasets | Planned |

## Assumptions

* **"Synthetic data" for the first tests** means data generated by code
  with an exact known answer. Real public datasets come in Phase 2.
* **"Local GitHub Actions runner"** means a self-hosted runner with an
  NVIDIA GPU. Because the repository is public, the GPU job only starts by
  hand, never from pull requests.
* **MedGemma 4B** is the CPU model in CI, because 27B does not fit the
  speed or memory of a hosted runner (18.3 GB in 4-bit vs 16 GB RAM).
* **UniMedVL** was studied and its resources calculated, but not run,
  because it needs a 40 GB+ GPU that is not available here.
* Nothing here is medical advice, and none of the models is approved for
  clinical use.
