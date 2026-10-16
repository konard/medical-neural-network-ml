# Research notes: models for medical data processing

Issue: https://github.com/konard/medical-neural-network-ml/issues/1
Compiled: 2026-10-14. These notes are raw material for the case study, not the final plan.

## Introduction

The issue asks which neural networks can:

1. OCR medical histories and handwritten prescriptions.
2. Read X-ray, MRI, CT and ultrasound images.
3. Suggest an initial diagnosis from symptoms, giving several most-likely options.

It asks us to start from `google/medgemma-27b-it` and `General-Medical-AI/UniMedVL`.

No single open model does all three tasks well. The current landscape splits into six groups:

| Group | What it does | Typical members |
|---|---|---|
| Medical vision-language models (VLMs) | image + text in, text out | MedGemma, Lingshu, Hulu-Med, HuatuoGPT-Vision |
| Unified understand + generate models | text/image in, text/image out | UniMedVL, HealthGPT |
| Medical text LLMs | reasoning and differential diagnosis from text | MedGemma-27B-text, HuatuoGPT-o1, Baichuan-M2/M3, Med42 |
| Document OCR / document VLMs | scans to text, including some handwriting | PaddleOCR-VL, dots.ocr, olmOCR-2, DeepSeek-OCR, Chandra, TrOCR |
| Task-specific imaging models | classification and segmentation with measurable accuracy | TorchXRayVision, nnU-Net, TotalSegmentator, MedSAM2 |
| Contrastive encoders | embeddings and zero-shot classification | MedSIGLIP, BiomedCLIP |

### How the facts were checked

- Parameter counts come from the Hugging Face (HF) API `safetensors` metadata unless noted. The API was checked on 2026-10-13/14.
- Dates are HF repo creation dates or arXiv v1 dates.
- Benchmarks are copied from model cards or papers. They are self-reported by the authors unless noted.
- No model weights were downloaded.

### How memory was estimated

These are rules of thumb for the weights only:

| Precision | Bytes per parameter |
|---|---|
| bf16 | 2 |
| int8 | 1 |
| 4-bit (NF4/GPTQ/AWQ/GGUF Q4) | about 0.55–0.6 |

Real use needs extra memory:

- Add about 10–30% for activations and KV cache at short context.
- Long contexts (32K–128K) can need many more GB.

All memory figures in this file are estimates unless a source is given.

---

## 1. MedGemma family (Google) and MedSigLIP

### 1.1 Overview

| Item | Value |
|---|---|
| Owner | Google Health AI / Google DeepMind (repo `google-health/medgemma`, part of Health AI Developer Foundations, HAI-DEF) |
| First release | MedGemma 4B and 27B-text: 2025-05-20 (Google I/O). MedGemma 27B multimodal and MedSigLIP: 2025-07-09. MedGemma 1.5 4B: 2026-01-13 |
| Tech report | arXiv:2507.05201 (Sellergren et al., v1 2025-07-07, v4 2026-04-06). MedGemma 1.5: arXiv:2604.05081 (v1 2026-04-06) |
| License | "Health AI Developer Foundations Terms of Use" (HF license field `other`). Commercial use is allowed under these terms, but they include use restrictions and are not OSI-open |
| Gating | All HF repos are gated (`gated: auto`). You must log in and accept the terms, then access is granted automatically |
| Training stack | JAX on TPUv4, TPUv5e and TPUv5p |

The Gemma-3 based variants:

| Model ID | HF created | Params (safetensors) | Base | Modalities |
|---|---|---|---|---|
| google/medgemma-4b-pt | 2025-05-19 | 4.30B | gemma-3-4b-pt | image + text → text |
| google/medgemma-4b-it | 2025-05-19 | 4.30B | medgemma-4b-pt | image + text → text |
| google/medgemma-27b-text-it | 2025-05-19 | 27.0B | gemma-3-27b-pt | text → text |
| google/medgemma-27b-it | 2025-07-09 | 27,432,406,640 (BF16) | gemma-3-27b-pt | image + text → text |
| google/medgemma-1.5-4b-it | 2026-01-07 (public 2026-01-13) | 4.30B | Gemma 3 4B | image (2D, 3D volumes, WSI) + text → text |
| google/medsiglip-448 | 2025-07-09 | 878M (F32) | SigLIP-400M | image + text → embeddings |

Notes on the table:

- The HF API reports a "total" of 28.84B for `medgemma-27b-it`. That figure is probably double counting. The BF16 tensor count (27.43B) is the same as `gemma-3-27b-it`.
- MedGemma 1.5 exists only at 4B. There is no MedGemma 1.5 27B.
- There is no MedGemma based on Gemma 4 as of 2026-10-14. Gemma 4 itself is Apache-2.0 and came out in spring 2026 (tech report arXiv:2607.02770).
- A secondary source says a MedGemma paper appeared in Nature Medicine in October 2026 (unverified).

### 1.2 Architecture: Gemma 3 plus a medical SigLIP encoder

MedGemma keeps the Gemma 3 architecture unchanged. The medical skill comes from changes to the data and the training stages.

**Vision encoder**

- SigLIP-400M ("So400m": 27 layers, hidden size 1152, patch size 14) at 896×896 input.
- Each image is compressed to exactly 256 soft tokens. A 4×4 average pool over the 64×64 patch grid does this.
- In MedGemma the encoder is replaced by MedSigLIP weights. These are the SigLIP encoder further trained on medical image-text pairs.
- The vision tokens go through a linear projection into the LLM embedding space. This is the Gemma 3 "multimodal projector" with RMSNorm.

**LLM backbone (27B)**

| Setting | Value |
|---|---|
| Model class | `Gemma3ForConditionalGeneration` |
| Decoder layers | 62 |
| Hidden size | 5376 |
| MLP intermediate size | 21504 |
| Attention heads | 32 query heads, 16 KV heads (grouped-query attention, GQA) |
| Head dim | 128 |
| Vocabulary | 262,208 tokens (SentencePiece, shared with Gemini) |

**Attention and context**

- Layers alternate 5 local sliding-window layers (window 1024 tokens) to 1 global layer.
- This cuts KV-cache memory a lot and is why 128K context is practical.
- RoPE base is 1e6 on global layers and 1e4 on local layers, with linear scaling factor 8.
- `max_position_embeddings` is 131,072. The model card states 128K input tokens and 8,192 output tokens.
- Inside an image, image tokens use bidirectional attention. Text uses causal attention.

**Other Gemma 3 features**

- QK-norm replaces Gemma 2 soft-capping.
- Gemma 3 has "Pan & Scan" (crop and tile high-res or non-square images). The MedGemma report does not say it is used. Medical images are just resized to 896×896.

**Output:** text only. MedGemma cannot produce images or segmentation masks. MedGemma 1.5 can output bounding-box coordinates as text.

**How the medical variants were trained**

1. **MedSigLIP.** SigLIP-400M was further trained on about 33M medical image-text pairs: 635k non-pathology images and 32.6M histopathology patches. Original SigLIP data was mixed in to keep general ability. The model was trained at 896px. The 448px release down-samples the positional embeddings.
2. **Multimodal pretraining.** Gemma 3 pretraining continued with medical image-text data mixed in. The listed data includes:
   - MIMIC-CXR (about 231k)
   - EyePACS fundus images (about 199k)
   - dermatology (about 51k)
   - CT slices (about 60k) and MRI slices (about 47.6k)
   - PMC-OA figures (about 41.8k)
   - histopathology
3. **Post-training.** Distillation plus reinforcement learning (RL) on medical text and multimodal data:
   - MedMCQA (about 182k) and MedQA (about 9.3k)
   - about 200k synthetic questions
   - EHR / FHIR data for the 27B multimodal model
4. **27B-text** was trained only on medical text post-training data. It is the best MedGemma for pure text diagnosis.
5. Both 27B models use **test-time scaling** for some reported numbers, such as best-of-5 on MedQA.

### 1.3 Benchmarks (self-reported, model card and tech report)

**Text QA**

| Benchmark | Gemma3-4B | MedGemma-4B | Gemma3-27B | MedGemma-27B-text | MedGemma-27B-mm |
|---|---|---|---|---|---|
| MedQA (4-opt) | 50.7 | 64.4 | 74.9 | 87.7 (89.8 best-of-5) | 85.3 (87.0 best-of-5) |
| MedMCQA | 45.4 | 55.7 | 62.6 | 74.2 | 70.2 |
| PubMedQA | 68.4 | 73.4 | 73.4 | 76.8 | 77.2 |
| MMLU Med | 67.2 | 70.0 | 83.3 | 87.0 | 86.2 |
| MedXpertQA (text) | 11.6 | 14.2 | 15.7 | 25.7 | 23.7 |
| AfriMed-QA | 48 | 52 | 72 | 84 | 72 |
| EHRQA | 70.9 | 67.6 | 84.2 | 86.3 | 90.5 |

**Imaging**

| Benchmark | Gemma3-4B | MedGemma-4B | Gemma3-27B | MedGemma-27B-mm |
|---|---|---|---|---|
| MIMIC-CXR top-5 conditions, macro F1 | 81.2 | 88.9 | 71.7 | 90.0 |
| CheXpert CXR, macro F1 | 32.6 | 48.1 | 26.2 | 49.9 |
| CXR14 (3 conditions), macro F1 | 32.0 | 50.1 | 31.4 | 45.3 |
| PathMCQA (histopathology) | 37.1 | 69.8 | 42.2 | 71.6 |
| US-DermMCQA | 52.5 | 71.8 | 66.9 | 71.7 |
| EyePACS (fundus) | 14.4 | 64.9 | 20.3 | 75.3 |
| SLAKE (tokenized F1) | 40.2 | 72.3 | 42.5 | 70.0 |
| VQA-RAD (tokenized F1) | 33.6 | 49.9 | 42.7 | 46.7 |
| MedXpertQA (multimodal) | 16.4 | 18.8 | 22.0 | 26.8 |

**CXR report generation (MIMIC-CXR, RadGraph F1)**

| Model | RadGraph F1 |
|---|---|
| MedGemma-4B-pt | 29.5 |
| MedGemma-4B tuned for CXR reports | 30.3 |
| MedGemma-27B-mm pt (not released) | 27.0 |
| PaliGemma 2 3B | 28.8 |
| PaliGemma 2 10B | 29.5 |
| MedGemma-4B-it | 21.9 |
| MedGemma-27B-it | 21.3 |

The instruction-tuned models score lower because their report style differs from MIMIC references.

**Human and agent evaluation**

- **Radiologist review.** A US board-certified radiologist judged 81% of MedGemma-4B CXR reports as leading to the same or better clinical decisions than the original reports. This was 306 cases with a single reader.
- **AgentClinic** (simulated patient dialogue with diagnosis):

| Setting | MedGemma 27B | Gemma 3 27B |
|---|---|---|
| AgentClinic-MedQA | 56.2 | 50.7 |
| AgentClinic-MIMIC | 46.0 | 35.2 |

- **Summary claims** in the tech report:
  - +2.6–10% on medical multimodal QA versus base Gemma 3.
  - +15.5–18.1% on CXR classification.
  - +10.8% on agentic tasks.
  - Fine-tuning cut EHR retrieval errors by about 50%.

**MedGemma 1.5 4B versus MedGemma 1** (arXiv:2604.05081)

| Task | MedGemma 1 4B | MedGemma 1.5 4B | MedGemma 1 27B |
|---|---|---|---|
| MedQA | 64.4 | 69.1 | — |
| MedMCQA | 55.7 | 59.8 | — |
| CT, 7 conditions (3D volume input) | 58.2 | 61.1 | 57.8 |
| MRI classification | 51.3 | 64.7 | 57.4 |
| Whole-slide pathology report (ROUGE) | 2.2 | 49.4 | — |
| EyePACS | 64.9 | 76.8 | — |
| Chest ImaGenome bbox IoU (localization) | 3.1 | 38.0 | — |
| EHRQA | 67.6 | 89.6 | 90.5 |
| SLAKE (tokenized F1) | 72.3 | 59.7 (regression) | — |
| Lab-report extraction, macro F1 | — | 85 | — |
| EHR PDF extraction (dataset 2), macro F1 | — | 91 | — |
| CXR report RadGraph F1 | — | 27.2 | — |

MedGemma 1.5 adds features that matter for this project:

- 3D CT/MRI volumes (passed as many slices)
- whole-slide pathology
- longitudinal CXR comparison
- anatomy localization (bounding boxes)
- medical document understanding, such as lab reports and PDFs, which overlaps with the OCR task

### 1.4 Memory (estimates)

Arithmetic only: parameters × bytes per weight. The KV cache, runtime buffers, the vision encoder's activations and, for training, the logits are not included, and none of these sizes was measured on the GPUs named. "Fits" is therefore a hypothesis.

| Model | bf16 | int8 | 4-bit |
|---|---|---|---|
| MedGemma 4B | about 8.6 GB (fits a 12–16 GB GPU) | about 4.5 GB | about 3 GB |
| MedGemma 27B | about 55 GB (one 80 GB A100/H100, or 2×48 GB) | about 28 GB | about 16–17 GB (expected to fit one 24 GB GPU at short context; not measured) |
| MedSigLIP | under 2 GB in fp16 | — | — |

Community quantizations exist, such as the unsloth GGUF and bnb-4bit builds of `medgemma-27b-it`. 128K context on the 27B adds many GB of KV cache, even with the sliding-window design.

### 1.5 Limitations (from model card and report)

- Not validated for direct clinical use. Outputs need independent verification and clinical correlation.
- Evaluated mostly on single-image, single-turn tasks. Multi-image and multi-turn use is less tested. MedGemma 1.5 adds some longitudinal support.
- Sensitive to prompt wording. Mainly English.
- Possible benchmark contamination. Google says it checked, but the risk remains for public datasets.
- Free-text CXR reports from the IT models score lower than from the PT models on RadGraph. Report generation usually needs task fine-tuning.
- No image or mask output, so it does not segment.

### 1.6 MedSigLIP

| Item | Value |
|---|---|
| Model ID | `google/medsiglip-448` (2025-07-09) |
| Parameters | 878M total: about 400M vision tower + about 400M text tower, stored in F32 |
| Input | 448×448 images, text up to 64 tokens |
| Output | shared embedding space |
| Uses | zero-shot classification, retrieval, linear probes; it is also the vision tower inside MedGemma |
| License | HAI-DEF terms, gated |
| Paper claim | competitive with or better than task-specific encoders on CXR, dermatology, ophthalmology and pathology linear-probe and zero-shot tasks (see arXiv:2507.05201) |

---

## 2. UniMedVL (Shanghai AI Lab, General Medical AI group)

### 2.1 Overview

| Item | Value |
|---|---|
| Model ID | `General-Medical-AI/UniMedVL` (HF, created 2025-10-15, not gated, Apache-2.0, tags `qwen2`, `custom_code`, `any-to-any`, about 43 downloads) |
| Code | https://github.com/uni-medical/UniMedVL (created 2025-10-11, Apache-2.0, about 111 stars). The HF org and GitHub org names differ |
| Paper | arXiv:2510.15710, "UniMedVL: Unifying Medical Multimodal Understanding and Generation through Observation-Knowledge-Analysis" (Ning et al., Junjun He's group). v3 revised 2026-05-28 |
| Weights | `ema_bf16.safetensors` 29.2 GB (about 14.6B params) + `ae.safetensors` 335 MB (FLUX VAE) |
| What it does | image + text → text (VQA, report generation); text (+ image) → image (text-to-image, virtual staining, super-resolution, modality translation, counterfactual CXR); interleaved tasks |
| Training code / data | not released as of the README. The UniMed-5M dataset is "coming soon" |

### 2.2 Architecture

UniMedVL is a medical fine-tune of ByteDance **BAGEL-7B-MoT**. The code tree has `modeling/unimedvl/bagel.py`, `qwen2_navit.py` and `siglip_navit.py`, with `layer_module: Qwen2MoTDecoderLayer`. The README credits Bagel and HealthGPT.

**LLM, as a Mixture-of-Transformers (MoT)**

- Starts from Qwen2.5-7B-Instruct.
  - Config: hidden size 3584, 28 layers, 28 attention heads, 4 KV heads.
  - MLP size 18944, vocabulary 152,064, max positions 32,768, RoPE θ 1e6, QK-norm on.
- MoT keeps two full copies of the transformer weights:
  - an "understanding" expert for text and ViT tokens
  - a "generation" expert for VAE latent tokens
- Each token is routed by its type, not by a learned router.
- All tokens share one self-attention over the whole sequence. Projections and FFNs are separate per expert.
- So about 7B parameters are active per token, out of about 14B total. This matches Bagel's "7B active / 14B total".

**Understanding encoder**

- A SigLIP2-so400m-style ViT: hidden 1152, 26 layers, patch 14.
- Uses NaViT-style native resolution packing at 378–980 px (`vit_max_num_patch_per_side` = 70).
- An MLP connector maps ViT features into the LLM.

**Generation path**

- Images are encoded with the FLUX VAE (8× down-sampling, kept frozen, not medically tuned).
- The model predicts velocity for rectified-flow / flow-matching denoising of the latents, at 512–1024 px.
- Training loss is next-token cross-entropy + flow-matching MSE, weighted 0.25 : 1.

**Context:** packed sequences up to the Qwen2 limit of 32K positions. The practical limits in the inference scripts are smaller (unverified exact value).

**"Observation-Knowledge-Analysis" (OKA)** is the paper's framing:

- Observation is the UniMed-5M data.
- Knowledge is curriculum training.
- Analysis is the unified model.

**Training**

The three-stage curriculum used LR 5e-5, then 2.5e-5, then 1e-5, with AdamW and EMA 0.995.

| Stage | Steps | Image-to-text (I2T) | Text-to-image (T2I) | Interleaved |
|---|---|---|---|---|
| 1 | 85K | 75% | 25% | — |
| 2 | 120K | 40% | 45% | 10% |
| 3 | 70K | 37% | 35% | 25% |

Training hardware was 8×A800 80 GB. The authors recommend at least 16×A800 for training.

**Data (UniMed-5M)**

- 5.6M samples: 4.0M for understanding and 1.6M for generation.
- Covers 8–9 modalities: CXR, CT, MRI, ultrasound, pathology, fundus, dermoscopy, endoscopy and more.
- Built from PMC-OA (1.0M), Quilt-1M (644K), HealthGPT data (638K), PubMedVision (385K), GMAI-VL (288K) and others.
- Stage 2 uses 1.9M samples. Stage 3 uses 330K interleaved samples.

### 2.3 Benchmarks (self-reported)

**Medical VQA (accuracy %)**

| Model | VQA-RAD | SLAKE | PathVQA | OmniMedVQA | GMAI-MMBench | Avg |
|---|---|---|---|---|---|---|
| UniMedVL (14B total) | 61.9 | 75.4 | 53.5 | 85.8 | 60.75 | 67.47 |
| GMAI-VL-7B (understanding only) | 66.3 | 72.9 | 39.8 | 88.5 | 61.74 | — |
| HealthGPT-L14 (unified) | 58.3 | 64.5 | 44.4 | 74.4 | 43.1 | — |
| Bagel-7B (base, general) | 60.09 | 58.91 | 39.05 | 71.13 | 48.11 | — |

**Generation and image-to-image tasks**

| Task | Metric | UniMedVL | Comparison |
|---|---|---|---|
| Text-to-image (8 modalities, average) | gFID ↓ | 96.29 | Bagel 215.49 |
| Text-to-image (8 modalities, average) | BiomedCLIP score ↑ | 0.706 | Bagel 0.660 |
| Virtual IHC staining | PSNR / SSIM | 20.27 / 0.456 | |
| MRI super-resolution | PSNR / SSIM | 27.29 / 0.890 | |
| T2↔FLAIR translation | PSNR / SSIM | 25.07 / 0.882 | |
| Counterfactual CXR | gFID | 27.17 | |

**MIMIC-CXR report generation**

| METEOR | BLEU-4 | ROUGE-L | RaTE |
|---|---|---|---|
| 0.2193 | 0.0235 | 0.2727 | 0.5138 |

**Segmentation (v3 of the paper):** Dice 33.51, against 51.14 for a dedicated Medical SAM3 baseline. It is far from specialist segmenters.

### 2.4 Memory and running it

**Software:** custom code with no plain `transformers` pipeline.

- The README uses a conda env (`codes/environment.yaml`) with Python 3.10, torch 2.5.1, transformers 4.49.0 and flash-attn.
- Inference scripts are `codes/interactive_vqa_inferencer.py` and `codes/interactive_image_generator.py`.

**Memory**

- The bf16 weights alone take about 29 GB.
- The inferencer splits layers across GPUs with accelerate `infer_auto_device_map`, with default `max_mem_per_gpu="40GiB"`. A 48 GB or 80 GB card works. 2×24 GB may work with the device map (unverified).

**Quantization**

- There are no official 4-bit or 8-bit builds.
- Back-of-envelope sizes: int8 about 15 GB, 4-bit about 8–9 GB.
- Quantizing the generation expert may hurt image quality (unverified).
- Only one expert is active per token, so compute per token is like a 7B model. Memory is like a 14B model.

### 2.5 Limitations (paper and README)

- 2D only, with no native 3D volumes.
- Generated images can have artifacts. The FLUX VAE is not medically tuned.
- Weak on surgical scenes, counting and grading tasks.
- Higher peak memory than understanding-only 7B VLMs.
- Research use only, not for clinical use.
- Few downloads and a small community. Training code and data are not yet public.
- For the issue's tasks (OCR, reading images, differential diagnosis), the image-generation ability is mostly irrelevant. Its VQA scores are similar to Lingshu-7B or Hulu-Med-7B, which are simpler, understanding-only models.

---

## 3. Medical vision-language alternatives and general baselines

### 3.1 Short profiles

**LLaVA-Med v1.5** (Microsoft, `microsoft/llava-med-v1.5-mistral-7b`, 2024-05-14, 7.57B, arXiv:2306.00890): CLIP ViT-L/14-336 + 2-layer MLP projector (`mlp2x_gelu`) + Mistral-7B-Instruct-v0.2. Trained on PMC-15M figure-caption pairs (concept alignment, then GPT-4-generated instructions). The HF field says Apache-2.0, but the repo states research use only. It is a historic baseline and now scores weakly (37.8 average on Lingshu's suite).

**HuatuoGPT-Vision** (CUHK-Shenzhen / FreedomIntelligence)

| Item | Value |
|---|---|
| Released | 2024-06-26 |
| 7B | 7.94B params: Qwen2-7B + CLIP-L/336 + MLP |
| 34B | 34.75B params: Yi-34B style, 60 layers, LLaVA layout |
| Training data | PubMedVision, 1.3M VQA pairs: PubMed images re-captioned by GPT-4V, then cleaned |
| License | Apache-2.0 |
| Paper | arXiv:2406.19280 |
| Main lesson | data quality (denoised PubMed figures) mattered more than architecture |

**Lingshu** (Alibaba DAMO Academy)

| Item | Value |
|---|---|
| Released | 2025-06-05 |
| Sizes | Lingshu-7B (8.29B) and Lingshu-32B (33.45B), both on Qwen2.5-VL. Lingshu-I-8B (2026-01-26, 7.94B) is on InternVL3 |
| Architecture | Qwen2.5-VL: native-resolution ViT with window attention and 2D-RoPE, plus a 2×2 patch-merger MLP into a Qwen2.5 LLM |
| Training | multi-stage: medical captions, instruction data, medical reasoning; an RL variant was also explored. Covers 12+ modalities |
| License | MIT |
| Paper | arXiv:2506.07044 |
| Tooling | MedEvalKit evaluation toolkit (alibaba-damo-academy/MedEvalKit) |
| Status | Among the strongest open medical VLMs in mid-2025 |

**Hulu-Med** (Zhejiang University, ZJUI-AI4H)

| Item | Value |
|---|---|
| Released | 2025-10-09 |
| Sizes | 4B; 7B (8.04B, Qwen2.5-7B); 14B (Qwen3-14B); 32B (33.2B, Qwen2.5-32B) |
| Inputs | one model for 2D images, 3D volumes and video |
| Efficiency trick | "medical-aware token reduction" cuts up to about 55% of visual tokens |
| Training data | 16.7M public or synthetic samples |
| Training cost | about 4k–40k GPU-hours |
| License | Apache-2.0 |
| Paper | arXiv:2510.08668 |
| 7B scores | OmniMedVQA 84.2, PMC-VQA 66.8, VQA-RAD 78.0, SLAKE 86.8, PathVQA 65.6, MedXpertQA 29.0, MedQA 73.5 |
| 32B scores | VQA-RAD 81.4, SLAKE 85.7, MedQA 80.4 |

**MedVLM-R1**

| Item | Value |
|---|---|
| Model ID | `JZPeterPan/MedVLM-R1` |
| Params / base | 2.21B, Qwen2-VL-2B |
| Training | GRPO reinforcement learning on only 600 VQA samples, rewarding format and answer correctness |
| License | Apache-2.0 |
| Paper | arXiv:2502.19634 |
| Result | accuracy on the MRI → CT/X-ray generalization test rose from 55.11% to 78.22% |
| Main lesson | RL with verifiable rewards generalizes better than supervised fine-tuning on small data |

**CheXagent** (Stanford AIMI)

| Item | Value |
|---|---|
| Versions | CheXagent-8b (2023-12-15, 8.36B) and CheXagent-2-3b (2024-04-29, 3.14B) |
| CheXagent-2-3b LLM | phi-style, hidden 2560, 32 layers, 2048 context |
| Scope | chest X-ray only: classification, findings, report generation, grounding |
| Benchmark | CheXbench |
| License | MIT on CheXagent-2-3b; check each repo |
| Paper | arXiv:2401.12208 |
| Reader study | drafts saved residents about 36% of report-writing time |

**Older and narrower models (short notes)**

- **RadFM** (2023-08, arXiv:2308.02463, MIT code at chaoyi-wu/RadFM): trained on MedMD, about 16M 2D and 3D scans covering about 5000 diseases. Architecture is a 3D ViT + perceiver resampler + MedLLaMA-13B (unverified details). One of the first models with native 3D input; now superseded on benchmarks.
- **Med-Flamingo** (2023-07, arXiv:2307.15189): OpenFlamingo-9B (CLIP ViT-L/14 + LLaMA-7B with gated cross-attention). Built for few-shot in-context learning. Historic; LLaMA-1 license limits apply.
- **BiomedCLIP** (Microsoft, 2023-04, arXiv:2303.00915, MIT): contrastive ViT-B/16 (224 px) + PubMedBERT text encoder, trained on PMC-15M. Used for zero-shot classification, retrieval, and as a metric (UniMedVL reports a "BiomedCLIP score").

**MAIRA-2** (Microsoft)

| Item | Value |
|---|---|
| Model ID | `microsoft/maira-2` (2024-07-29) |
| Params | 6.88B |
| Architecture | RAD-DINO chest X-ray encoder + Vicuna-7B v1.5 (unverified exact LLM) |
| Task | grounded radiology reports, where each finding sentence gets a bounding box |
| Metric | RadFact |
| License | HF field `other`; research-only MSRLA (unverified) |
| Gating | gated |
| Paper | arXiv:2406.04449 |

**BiomedParse** (Microsoft)

| Item | Value |
|---|---|
| Released | 2024-11-04 |
| Tasks | text-prompted segmentation, detection and recognition: 82 object types across 9 modalities |
| Training data | about 6M image-mask-text triples |
| License | weights cc-by-nc-sa-4.0 (non-commercial), gated; code Apache-2.0 |
| Paper | arXiv:2405.12971 (Nature Methods) |

### 3.2 General-purpose VLM baselines

Medical fine-tunes usually build on these models. Recent general models often match older medical fine-tunes.

| Model | Released | Params | License | Notes |
|---|---|---|---|---|
| Qwen2.5-VL-7B / 32B / 72B | 2025-01 | 8.29B / about 33.5B / 73.4B | Apache-2.0 (7B, 32B); Qwen license (72B) | Base of Lingshu, MedVLM-R1 (Qwen2-VL), olmOCR-2 and others |
| Qwen3-VL-8B / 235B-A22B | 2025-09/10 | 8.77B / 235B MoE (22B active) | Apache-2.0 | Base of Chandra OCR |
| Qwen3.5-27B | 2026-02-24 | about 27B | Apache-2.0 | Newer general baseline (medical scores unverified) |
| InternVL3-8B / 38B / 78B | 2025-04 | 7.94B / about 38B / about 78B | Apache-2.0 (8B); `other` (78B) | InternViT + Qwen2.5 LLM, MLP projector. Base of Lingshu-I. InternVL3.5 followed in 2025-08 |
| Llama-3.2-11B / 90B-Vision | 2024-09 | 10.67B / 88.6B | Llama 3.2 Community License, gated (manual) | Cross-attention adapter into a frozen Llama 3.1. Weak on medical VQA (unverified exact scores) |
| Gemma-4-31B-it | 2026 | about 31B | Apache-2.0 | No medical variant yet |

### 3.3 Comparison on one shared benchmark suite

These numbers come from the Lingshu paper and model card (accuracy %). The suite is the same for every row, so this is the fairest public cross-model table found. It is still run by one of the competitors (Lingshu's authors).

| Model | MMMU-Med | VQA-RAD | SLAKE | PathVQA | PMC-VQA | OmniMedVQA | MedXpertQA | Avg |
|---|---|---|---|---|---|---|---|---|
| GPT-4.1 | 75.2 | 65.0 | 72.2 | 55.5 | 55.2 | 75.5 | 45.2 | 63.4 |
| Claude Sonnet 4 | 74.6 | 67.6 | 70.6 | 54.2 | 54.4 | 65.5 | 43.3 | 61.5 |
| Gemini-2.5-Flash | 76.9 | 68.5 | 75.8 | 55.4 | 55.4 | 71.0 | 52.8 | 65.1 |
| MedVLM-R1-2B | 35.2 | 48.6 | 56.0 | 32.5 | 47.6 | 77.7 | 20.4 | 45.4 |
| MedGemma-4B-IT | 43.7 | 72.5 | 76.4 | 48.8 | 49.9 | 69.8 | 22.3 | 54.8 |
| LLaVA-Med-7B | 29.3 | 53.7 | 48.0 | 38.8 | 30.5 | 44.3 | 20.3 | 37.8 |
| HuatuoGPT-V-7B | 47.3 | 67.0 | 67.8 | 48.0 | 53.3 | 74.2 | 21.6 | 54.2 |
| Qwen2.5-VL-7B | 50.6 | 64.5 | 67.2 | 44.1 | 51.9 | 63.6 | 22.3 | 52.0 |
| InternVL3-8B | 59.2 | 65.4 | 72.8 | 48.6 | 53.8 | 79.1 | 22.4 | 57.3 |
| Lingshu-7B | 54.0 | 67.9 | 83.1 | 61.9 | 56.3 | 82.9 | 26.7 | 61.8 |
| HealthGPT-14B | — | — | — | — | — | — | — | 56.2 |
| HuatuoGPT-V-34B | — | — | — | — | — | — | — | 54.3 |
| Qwen2.5-VL-32B | — | — | — | — | — | — | — | 56.1 |
| InternVL3-38B | — | — | — | — | — | — | — | 59.4 |
| Lingshu-32B | 62.3 | 76.5 | 89.2 | 65.9 | 57.9 | 83.4 | 30.9 | 66.6 |

MedGemma-27B and UniMedVL are not in this table. Their own numbers use different metrics, such as tokenized F1 instead of accuracy, so they cannot be compared directly.

From the same source:

**MedQA (text)**

| Model | MedQA |
|---|---|
| Claude Sonnet 4 | 92.1 |
| GPT-4.1 | 89.1 |
| Lingshu-32B | 74.7 |
| Lingshu-7B | 63.3 |
| InternVL3-8B | 62.1 |
| Qwen2.5-VL-7B | 57.3 |
| MedGemma-4B | 56.2 |

**Report generation** (ROUGE-L / CIDEr / RaTE / SembScore / RadCliQ⁻¹)

| Model | MIMIC-CXR |
|---|---|
| Lingshu-7B | 30.8 / 109.4 / 52.1 / 30.0 / 69.2 |
| MedGemma-4B | 25.6 / 81.0 / 52.4 / 29.2 / 62.9 |
| GPT-4.1 | 9.0 / 82.8 / 51.3 / 23.9 / 57.1 |

### 3.4 Summary table: medical VLMs

Memory figures are estimates (see the Introduction).

| Model | Params | Architecture | Modalities | License | Gated | Min VRAM 4-bit / bf16 | Key benchmark |
|---|---|---|---|---|---|---|---|
| MedGemma-4B-it | 4.30B | MedSigLIP-400M (896px, 256 tok) + Gemma 3 4B | img+txt → txt | HAI-DEF | yes (auto) | about 3 / 9 GB | MedQA 64.4; MIMIC-CXR F1 88.9 |
| MedGemma-1.5-4B-it | 4.30B | same, plus 3D/WSI/bbox training | 2D, 3D, WSI, docs → txt | HAI-DEF | yes (auto) | about 3 / 9 GB | MedQA 69.1; EHRQA 89.6 |
| MedGemma-27B-it | 27.43B | MedSigLIP + Gemma 3 27B (62 L, GQA, 5:1 local/global, 128K) | img+txt → txt | HAI-DEF | yes (auto) | about 17 / 55 GB | MedQA 87.0 (best-of-5); EyePACS 75.3 |
| UniMedVL | about 14.6B (7B active) | Bagel MoT: 2× Qwen2.5-7B experts + SigLIP2 NaViT + FLUX VAE, flow matching | img+txt → txt / img | Apache-2.0 | no | about 9 (unofficial) / 29+ GB | SLAKE 75.4; OmniMedVQA 85.8 |
| Lingshu-7B | 8.29B | Qwen2.5-VL-7B | img+txt → txt | MIT | no | about 6 / 17 GB | 61.8 avg (7 VQA sets) |
| Lingshu-32B | 33.45B | Qwen2.5-VL-32B | img+txt → txt | MIT | no | about 20 / 67 GB | 66.6 avg; SLAKE 89.2 |
| Hulu-Med-7B | 8.04B | Qwen2.5-7B + ViT, token reduction | 2D, 3D, video → txt | Apache-2.0 | no | about 6 / 16 GB | SLAKE 86.8; VQA-RAD 78.0 |
| Hulu-Med-32B | 33.2B | Qwen2.5-32B + ViT | 2D, 3D, video → txt | Apache-2.0 | no | about 20 / 66 GB | VQA-RAD 81.4; MedQA 80.4 |
| HuatuoGPT-Vision-7B | 7.94B | CLIP-L/336 + MLP + Qwen2-7B | img+txt → txt | Apache-2.0 | no | about 6 / 16 GB | 54.2 avg (Lingshu suite) |
| LLaVA-Med v1.5 | 7.57B | CLIP-L/336 + MLP + Mistral-7B | img+txt → txt | research only | no | about 5 / 15 GB | 37.8 avg (Lingshu suite) |
| MedVLM-R1 | 2.21B | Qwen2-VL-2B + GRPO | img+txt → txt | Apache-2.0 | no | about 2 / 5 GB | 78.22% OOD (out-of-distribution) VQA |
| CheXagent-2-3b | 3.14B | vision encoder + phi-style 3B LLM | CXR → txt | MIT | no (unverified) | about 2.5 / 7 GB | CheXbench |
| MAIRA-2 | 6.88B | RAD-DINO + Vicuna-7B | CXR (+prior) → grounded report | research-only (unverified) | yes | about 5 / 14 GB | RadFact |
| RadFM | about 14B (unverified) | 3D ViT + perceiver + MedLLaMA-13B (unverified) | 2D/3D → txt | MIT (code) | no | about 9 / 28 GB (unverified) | RadBench |
| BiomedCLIP | about 0.2B (unverified) | ViT-B/16 + PubMedBERT, contrastive | img/txt → embedding | MIT | no | under 1 GB | zero-shot classification |
| BiomedParse | about 0.9B (unverified) | SEEM-style text-prompted segmenter | img+txt → masks | CC-BY-NC-SA-4.0 | yes | about 4 GB fp16 (unverified) | Dice on 102k test triples |

---

## 4. Text LLMs for symptom-based diagnosis

### 4.1 Benchmarks used for diagnosis

**MedQA (USMLE)**

- Multiple-choice questions in the style of the US medical licensing exam.
- The common split is the English 4-option test set of 1,273 questions. The source paper is Jin et al. 2020, arXiv:2009.13081 (counts from memory, unverified).
- Strong 2026 open models score 85–92%, so it is close to saturated.

**MedMCQA**

- About 194k Indian medical entrance exam questions, with a 4.18k-question dev set used for evaluation (unverified counts).
- Paper: arXiv:2203.14371.

**PubMedQA**

- Yes/no/maybe questions about PubMed abstracts. 1k expert-labelled examples.
- Paper: arXiv:1909.06146.

**MMLU (medical subsets)**

- Clinical knowledge, medical genetics, anatomy, professional medicine, college biology and college medicine.

**MedXpertQA**

- Hard expert-level questions, with text and multimodal subsets. Paper: arXiv:2501.18362.
- Scores are still low: MedGemma-27B scores 25.7 (text) and frontier APIs score 45–53. It still separates models well.

**HealthBench** (OpenAI, 2025)

- 5,000 multi-turn health conversations, graded against 48k physician-written rubric criteria (unverified counts).
- Has a "Hard" subset.
- Closer to real chat-based triage than multiple-choice exams.

**DDXPlus** (Mila and Dialogue Health, NeurIPS 2022 Datasets track)

- Roughly 1.3M synthetic patients, each with a ground-truth pathology, symptoms, antecedents and a differential diagnosis (arXiv:2205.09148 abstract). About 49 pathologies (unverified count).
- It fits the issue's "several most-likely options" request directly. Metrics like top-k accuracy and differential F1 can be measured.
- The patients are synthetic and the pathology list is limited, mostly respiratory and ENT (unverified).

**AgentClinic** (arXiv:2405.07960)

- A simulated doctor-patient dialogue: the model must ask questions, order tests and then diagnose.
- Used in the MedGemma report.

### 4.2 Models

**MedGemma-27B-text-it** (see section 1)

- MedQA 87.7, or 89.8 with best-of-5.
- Best Google option for text-only differential diagnosis.
- HAI-DEF license, gated.

**Meditron** (EPFL)

- meditron-7b and meditron-70b (68.98B), released 2023-11-08.
- Llama-2 further pretrained on curated PubMed papers, abstracts and clinical guidelines (GAP-Replay, about 48B tokens; unverified token count).
- Llama-2 license, gated. Paper arXiv:2311.16079.
- Claimed +6% over the base Llama-2. MedQA about 70% for the 70B after fine-tuning (unverified).
- Meditron3-70B (2024-07) is built on Llama-3.1-70B.
- Now outdated.

**BioMistral-7B**

- Released 2024-02-14. Mistral-7B further pretrained on PubMed Central. Apache-2.0. Paper arXiv:2402.10373.
- Scores after supervised fine-tuning (SFT), 3-shot:

| MedQA | MedQA 5-option | PubMedQA | MedMCQA | Average |
|---|---|---|---|---|
| 50.6 | 42.8 | 77.5 | 48.1 | 57.3 |

- Weak by 2026 standards.

**OpenBioLLM** (Saama)

- Llama3-OpenBioLLM-70B and 8B, released 2024-04. Llama 3 license.
- Fine-tuned from Llama-3 Instruct with DPO on medical data.

| Model | Average (9 benchmarks) | MedQA (4-opt) | PubMedQA | MedMCQA |
|---|---|---|---|---|
| 70B | 86.06 | 78.16 | 78.97 | 74.01 |
| 8B | 72.5 | 58.99 | — | — |

**Med42-v2** (M42, Abu Dhabi)

- Llama3-Med42-70B (70.55B) and 8B, released 2024-06/07. Llama 3 license. Paper arXiv:2408.06142.

| Model | MMLU Pro | MMLU | MedMCQA | MedQA | USMLE |
|---|---|---|---|---|---|
| 70B | 64.36 | 87.12 | 73.20 | 79.10 | 83.80 |
| 8B | — | — | — | 62.84 | — |

**HuatuoGPT-o1** (CUHK-Shenzhen)

- Sizes: 8B (Llama-3.1-8B), 70B (Llama-3.1-70B), 7B (Qwen2.5-7B), 72B (Qwen2.5-72B). Released 2024-12-27. Apache-2.0. Paper arXiv:2412.18925.
- Training has two stages:
  1. A GPT-4o verifier checks answers to verifiable medical problems. A search finds long chain-of-thought paths that reach the right answer, and the model is fine-tuned on them.
  2. PPO RL with rewards from the verifier.
- MedQA: 72B 88.85, 8B 74.78. These are third-party numbers from the II-Medical model card.

**m1** (UCSC VLAA)

- m1-7B-23K (7.6B) and m1-32B-1K (32.8B). Qwen2.5 base. MIT. Released 2025-03-31. Paper arXiv:2504.00869.
- Shows that test-time scaling works for medicine: fine-tune on a small set of reasoning traces (1K–23K), then limit the "thinking" budget.
- The best budget is about 4K reasoning tokens. Longer thinking does not help and can hurt, because the limit is missing medical knowledge, not reasoning.
- m1-7B MedQA 75.81.

**Baichuan-M2-32B** (Baichuan)

- Released 2025-08-10. Qwen2.5-32B base. Apache-2.0. Paper arXiv:2509.02208.
- RL against a patient simulator plus a clinical rubric generator, which acts as a "verifier system".
- HealthBench scores, against 57.6 for gpt-oss-120b:

| Overall | Hard | Consensus |
|---|---|---|
| 60.1 | 34.7 | 91.5 |

- Fits a single high-end GPU in 4-bit.

**Baichuan-M3-235B**

- Released 2026-01-13. Qwen3-235B-A22B MoE base (22B active). Apache-2.0. Paper arXiv:2602.06570.
- HealthBench-Hard 44.4. The authors claim it beats GPT-5.2 on their medical evaluations (self-reported).
- FP8 and GPTQ-INT4 builds exist. The authors say W4 cuts memory to about 26% of bf16.
- Uses gated Eagle3 speculative decoding.
- Needs multi-GPU hardware even at 4-bit (about 130 GB, estimate).

**II-Medical-8B** (Intelligent Internet)

- Released 2025-05-12. Qwen3-8B base. Apache-2.0.
- HealthBench 40%, MedQA 87.82 (self-reported). Average 70.49 over 10 benchmarks.
- Very strong for its size.

### 4.3 Summary table: text LLMs

| Model | Params | Architecture | Modalities | License | Gated | Min VRAM 4-bit / bf16 | Key benchmark |
|---|---|---|---|---|---|---|---|
| MedGemma-27B-text-it | 27.0B | Gemma 3 decoder, 128K | txt | HAI-DEF | yes | about 17 / 54 GB | MedQA 87.7 (89.8 best-of-5) |
| Meditron-70B | 68.98B | Llama-2 + continued pretraining | txt | Llama 2 | yes | about 40 / 138 GB | MedQA about 70 (unverified) |
| BioMistral-7B | 7.24B | Mistral + PMC pretraining | txt | Apache-2.0 | no | about 5 / 15 GB | MedQA 50.6 |
| OpenBioLLM-70B | 70.6B | Llama-3 + DPO | txt | Llama 3 | no | about 40 / 141 GB | MedQA 78.16 |
| Med42-v2-70B | 70.55B | Llama-3 + SFT/DPO | txt | Llama 3 | yes (unverified) | about 40 / 141 GB | MedQA 79.10 |
| HuatuoGPT-o1-72B | about 72.7B | Qwen2.5-72B + verifier-guided CoT + RL | txt | Apache-2.0 (Qwen license for 72B base, unverified) | no | about 42 / 145 GB | MedQA 88.85 |
| HuatuoGPT-o1-8B | 8.03B | Llama-3.1-8B + CoT + RL | txt | Apache-2.0 / Llama 3.1 | no | about 6 / 16 GB | MedQA 74.78 |
| m1-7B-23K | 7.6B | Qwen2.5-7B + reasoning SFT + budget control | txt | MIT | no | about 5 / 15 GB | MedQA 75.81 |
| II-Medical-8B | 8.2B | Qwen3-8B + SFT/RL | txt | Apache-2.0 | no | about 6 / 16 GB | MedQA 87.82 |
| Baichuan-M2-32B | 32.8B | Qwen2.5-32B + patient-simulator RL | txt | Apache-2.0 | no | about 20 / 66 GB | HealthBench 60.1 |
| Baichuan-M3-235B | 235B MoE (22B active) | Qwen3-235B-A22B + RL | txt | Apache-2.0 | no | about 130 / 470 GB | HealthBench-Hard 44.4 |

---

## 5. OCR for medical histories and handwritten prescriptions

### 5.1 Approaches

| Approach | Examples | Strengths | Weaknesses |
|---|---|---|---|
| Classic pipeline (detect boxes, recognize lines) | Tesseract, PaddleOCR PP-OCR, docTR, Surya | Fast, small, run on CPU, give character boxes | Weak on cursive handwriting, complex layouts and tables |
| Line-level transformer recognizer | TrOCR | Very good on cropped handwritten lines | Needs a separate line detector; English-centric checkpoints |
| OCR-free document VLM, small (0.3–3B) | Donut, Nougat, GOT-OCR2, PaddleOCR-VL, GLM-OCR, dots.ocr, DeepSeek-OCR, granite-docling | Whole page to Markdown/HTML/JSON in one pass, including tables and reading order | Can hallucinate plausible text. Handwriting quality varies |
| OCR-tuned general VLM (7–9B) | olmOCR-2, Chandra, Nanonets-OCR2 | Best on hard pages and handwriting | Need a GPU. Hallucination risk on illegible text |
| Domain-aware general VLM | MedGemma-1.5-4B (lab reports and EHR PDFs), Qwen3-VL | Can extract structured medical fields directly | Not designed as an OCR engine. Few published handwriting numbers |

Handwritten prescriptions are the hardest case:

- Abbreviations (bid, tds, q8h), drug names and doses are often written in cursive.
- None of the benchmarks found focus on handwritten medical prescriptions. This needs our own evaluation set.
- Practical pattern: run an OCR VLM, then validate against a drug dictionary (RxNorm or a national formulary) with fuzzy matching, and flag low-confidence fields for human review.

### 5.2 Models and tools

| Tool | Owner / date | Params | Architecture | Handwriting | License |
|---|---|---|---|---|---|
| Tesseract 5 | Google / community (GitHub since 2014) | — | LSTM line recognizer | poor | Apache-2.0 |
| PaddleOCR PP-OCRv5 / v6 | Baidu (repo since 2020; PP-OCRv6 2026-06, arXiv:2606.13108) | 1.5M–34.5M (three tiers) | detection (DB) + recognition (SVTR-style) | some (v5 added handwriting data; unverified quality) | Apache-2.0 |
| PaddleOCR-VL / VL-1.5 | Baidu, 2025-10-16 / 2026-01-28 | 0.96B | NaViT dynamic-resolution encoder + ERNIE-4.5-0.3B LM, plus a layout model | yes, claimed (unverified) | Apache-2.0 |
| docTR | Mindee | small (unverified) | DBNet/LinkNet detection + CRNN/ViTSTR/PARSeq recognition | limited | Apache-2.0 |
| Surya | Datalab | about 0.3–1B (unverified) | line detection + transformer recognizer, layout and reading order | partial | code Apache-2.0 (GPL earlier, unverified); weights cc-by-nc-sa-4.0 or a modified OpenRAIL with revenue limits (unverified) |
| TrOCR | Microsoft, 2021 (arXiv:2109.10282) | base-handwritten 333M; large 558M (unverified) | ViT/BEiT encoder + RoBERTa-style decoder | yes, IAM English cursive lines | MIT |
| Donut | NAVER Clova, 2022 (arXiv:2111.15664) | about 200M (unverified) | Swin encoder + BART decoder, OCR-free | limited | MIT |
| Nougat | Meta, 2023 | 349M | Swin + mBART, academic PDFs to Markdown | no | CC-BY-NC-4.0 (weights) |
| GOT-OCR2.0 | StepFun / UCAS, 2024-09 (arXiv:2409.01704) | 580M (paper) / 716M (HF count) | high-compression encoder + Qwen-0.5B decoder | some | Apache-2.0 |
| Florence-2-large | Microsoft | 777M | DaViT + seq2seq, multi-task including OCR | limited | MIT |
| granite-docling-258M | IBM | 258M | SigLIP2 + small LM, outputs DocTags | no (printed docs) | Apache-2.0 |
| LightOnOCR-1B | LightOn | 1.16B | VLM OCR (Mistral-style; unverified) | unverified | Apache-2.0 |
| GLM-OCR | Zhipu, 2026-01-30 (arXiv:2603.10910) | about 0.9B (1.33B HF count) | 0.4B CogViT + 0.5B GLM decoder with multi-token prediction | claimed (unverified) | MIT |
| dots.ocr | rednote hi lab, 2025-07-30 | 3.04B (1.7B LLM) | NaViT-style encoder + small LLM; layout and content in one pass | some (unverified) | MIT |
| DeepSeek-OCR | DeepSeek, 2025-10-17 | 3.34B | "DeepEncoder" (SAM-base + CLIP-L with 16× token compression) + DeepSeek-3B-MoE (about 570M active) | some | MIT |
| DeepSeek-OCR 2 | DeepSeek, 2026-01-27 | 3.39B | DeepEncoder V2 reorders visual tokens causally; (0–6)×144 + 256 visual tokens per page | some | Apache-2.0 |
| Nanonets-OCR2-3B | Nanonets | 3.75B | Qwen2.5-VL-3B fine-tune; outputs Markdown, tables, checkboxes, signatures | yes, claimed | not stated in HF metadata (unverified) |
| olmOCR / olmOCR-2-7B-1025 | Allen AI, 2025-01 / 2025-10 | about 8.3B (7B LLM) | Qwen2-VL-7B / Qwen2.5-VL-7B fine-tune; v2 trained with RL on unit-test rewards; FP8 build recommended | some (old scans) | Apache-2.0 |
| Chandra | Datalab, 2025-10-21 | 8.77B | Qwen3-VL-8B fine-tune; layout-aware HTML/Markdown/JSON | yes: "good handwriting support", 40+ languages | code Apache-2.0; weights modified OpenRAIL-M (commercial limits, unverified) |

Benchmark: olmOCR-bench overall score (from the Chandra and olmOCR model cards, self-reported):

| System | olmOCR-bench overall |
|---|---|
| Chandra | 83.1 |
| olmOCR-2 pipeline v0.4.0 | 82.3 ± 1.1 (FP8: 82.4) |
| olmOCR v0.3.0 | 78.5 |
| Marker | 76.5 |
| DeepSeek-OCR | 75.4 |
| Mistral OCR API | 72.0 |
| GPT-4o | 69.9 |
| Qwen3-VL | 64.6 |

Other claims:

- On OmniDocBench, dots.ocr, PaddleOCR-VL-1.5 (94.5 on v1.5) and GLM-OCR each claim state of the art for their release dates.
- DeepSeek-OCR claims about 97% decoding precision at under 10× vision-token compression.
- PP-OCRv6 medium reports 86.2% detection Hmean and 83.2% recognition accuracy across 50 languages.

None of these benchmarks is medical, and handwriting is at most a small subset. For prescriptions we should build a private test set, for example 200–500 anonymized prescriptions, and measure:

- character error rate (CER)
- field-level accuracy for drug name, dose and frequency

### 5.3 Memory (estimates)

| Tool size | Example | Hardware |
|---|---|---|
| Classic OCR | Tesseract, PP-OCR | CPU is enough |
| 0.3–1B VLMs | PaddleOCR-VL, GLM-OCR, granite-docling | 2–4 GB GPU in bf16 |
| About 3B | dots.ocr, DeepSeek-OCR | about 7–8 GB bf16 |
| 7–9B | olmOCR-2, Chandra | about 16–18 GB bf16, about 9 GB FP8, about 6 GB 4-bit |

---

## 6. Imaging-specific (non-LLM) models and toolkits

Generalist VLMs describe images. Task-specific networks still give the best measurable accuracy on narrow tasks, such as CXR pathology probabilities or organ volumes, and are cheaper to run. A practical system can call these as "tools" and pass their outputs to the LLM.

| Tool | Owner | Created | Task | Architecture | License | Notes |
|---|---|---|---|---|---|---|
| TorchXRayVision | Cohen et al. (mlmed) | 2020 (arXiv:2111.00595) | CXR classification (18 pathologies), autoencoders, segmentation | DenseNet-121 / ResNet trained on NIH, CheXpert, MIMIC, PadChest and others | NOASSERTION on GitHub (Apache-2.0 per PyPI, unverified) | About 1.2k stars. CPU-friendly; good for baselines and feature extraction |
| MONAI | Project MONAI (NVIDIA, KCL and others) | 2019 | PyTorch framework: transforms, networks (UNet, UNETR, SwinUNETR), training, MONAI Label, Deploy | framework | Apache-2.0 | About 8.8k stars. The model-zoo bundles (Apache-2.0) include pretrained CT/MRI segmenters |
| nnU-Net | DKFZ (Isensee et al.) | 2018 (arXiv:1809.10486; Nature Methods 2021) | self-configuring segmentation: picks patch size, network depth, normalization and augmentation from dataset properties | U-Net (2D, 3D, cascade); v2 adds residual encoders | Apache-2.0 | About 9k stars. Still a top baseline in segmentation challenges |
| TotalSegmentator | University Hospital Basel | 2022 (arXiv:2208.05868) | CT (and later MRI) segmentation, originally 104 structures and now more | nnU-Net trained on 1,204 CT scans (unverified count) | Apache-2.0 code; some subtasks need a commercial license (unverified) | About 3k stars. Dice 0.943 on its test set. Runs on CPU (slow) or GPU |
| MedSAM | Ma et al. (Bo Wang lab) | 2023 (Nature Communications 2024) | box-prompted 2D segmentation | SAM ViT-B fine-tuned on 1.57M image-mask pairs, 10 modalities | Apache-2.0 | About 4.4k stars |
| MedSAM2 | same lab | 2025-04 | 3D volume and video segmentation | SAM2 (Hiera encoder + memory attention) fine-tuned on about 455k 3D image-mask pairs and 76k video frames | Apache-2.0 | Supports human-in-the-loop annotation |
| SAM-Med3D | OpenGVLab / uni-medical | 2023–2024 | promptable 3D segmentation | fully 3D SAM-style ViT | Apache-2.0 | Trained on SA-Med3D-140K (22K 3D images, 143K masks) |
| BiomedParse | Microsoft | 2024 | text-prompted segmentation, detection and recognition | see section 3.1 | CC-BY-NC-SA-4.0 (weights) | Non-commercial |

Ultrasound note: none of the generalist VLMs are strong on ultrasound. MedGemma lists only a dermatology "US-DermMCQA" set, which is US-sourced dermatology, not ultrasound. Ultrasound work will probably need:

- dedicated fine-tuning
- or MONAI / nnU-Net models trained on public ultrasound datasets, such as BUSI (breast) and CAMUS (cardiac) (unverified dataset licenses)

---

## 7. How architecture choices affect results

1. **Same backbone, different data.** MedGemma uses an unchanged Gemma 3, and Lingshu an unchanged Qwen2.5-VL. The large medical gains come from data and training:
   - MedGemma-4B against Gemma3-4B: EyePACS 14.4 → 64.9, PathMCQA 37.1 → 69.8.
   - Medical skill mostly comes from data curation, an in-domain vision encoder (MedSigLIP) and RL post-training, not from new layers.
2. **Vision-token budget and resolution.**
   - Gemma 3 compresses every image to 256 tokens at 896². This is cheap and good for whole-image classification, but fine detail can be lost: small nodules, fine text in scanned documents.
   - Qwen2.5-VL / Qwen3-VL and NaViT encoders (UniMedVL, PaddleOCR-VL, dots.ocr) keep native resolution and aspect ratio. Token count grows with image size. This helps OCR and localization but costs memory.
   - Hulu-Med and DeepSeek-OCR show the middle path: learned or heuristic token reduction.
3. **2D against 3D.** Most VLMs see CT/MRI as a few 2D slices. MedGemma 1.5, Hulu-Med, RadFM and SAM-Med3D / MedSAM2 add volume handling. For CT/MRI this matters more than parameter count. MedGemma 1.5 4B beats MedGemma 1 27B on MRI classification (64.7 against 57.4).
4. **Unified generation (UniMedVL) against understanding-only.**
   - MoT doubles the weights (14B) to add image generation, but its VQA scores land in the same range as 7–8B understanding-only models (Lingshu-7B, Hulu-Med-7B).
   - Generation helps synthetic data creation, modality translation and super-resolution. These are not core to the issue's tasks.
5. **Context length and attention pattern.** Gemma 3's 5:1 local/global attention makes 128K context affordable. This lets the model read long medical histories, EHR/FHIR records or many OCR'd pages in one prompt. Qwen2.5-based models default to 32K, and 128K with YaRN.
6. **Reasoning and RL.**
   - Text diagnosis gains most from reasoning training with verifiers: HuatuoGPT-o1, m1, Baichuan-M2/M3, II-Medical.
   - The m1 result (about 4K thinking tokens is optimal) says knowledge, not reasoning length, is the limit. This argues for retrieval (clinical guidelines) alongside a reasoning model.
   - Patient-simulator RL (Baichuan-M2, AgentClinic) is the closest public approach to "ask follow-up questions, then give a ranked differential".
7. **Instruction tuning changes report style.** MedGemma IT models score lower on RadGraph F1 than PT models (21.9 against 29.5). The benchmark rewards MIMIC-style wording, so model choice depends on the target output format.
8. **Metric mismatch.** Model cards use different metrics on the same dataset, such as tokenized F1 or accuracy on SLAKE and VQA-RAD. Scores are not comparable across cards. Use one harness (MedEvalKit, VLMEvalKit or lmms-eval) for our own comparisons.

---

## 8. Gaps and open questions for the case study

- **No open handwriting-prescription benchmark found.** We need a private evaluation set with consent and anonymization.
- **Licenses need review before production.**
  - MedGemma: HAI-DEF terms, gated.
  - MAIRA-2 and BiomedParse: non-commercial.
  - Surya and Chandra weights: restricted OpenRAIL variants.
  - Llama models: community licenses.
  - Apache-2.0 / MIT options include UniMedVL, Lingshu, Hulu-Med, Qwen-based OCR (PaddleOCR-VL, dots.ocr, olmOCR-2, DeepSeek-OCR) and Baichuan-M2/M3.
- **Russian and other non-English inputs.** If the clinic data is not English (the issue language suggests Russian may matter, unverified), most medical models are English or Chinese-centric. OCR models with broad language support (PaddleOCR-VL 109 languages, Chandra 40+, PP-OCRv6 50) and multilingual bases (Qwen3, Gemma 3 with 140+ languages) are safer starting points.
- **No regulatory clearance.** None of these models are cleared as medical devices (FDA/CE). All model cards say research or development use only.

---

## Sources

### MedGemma / MedSigLIP / Gemma

- https://huggingface.co/google/medgemma-27b-it; https://huggingface.co/google/medgemma-27b-text-it; https://huggingface.co/google/medgemma-4b-it
- https://huggingface.co/google/medgemma-4b-pt; https://huggingface.co/google/medgemma-1.5-4b-it; https://huggingface.co/google/medsiglip-448
- https://huggingface.co/unsloth/medgemma-27b-it (ungated mirror, used to read the model card and config); https://developers.google.com/health-ai-developer-foundations/medgemma; https://developers.google.com/health-ai-developer-foundations/terms
- https://github.com/google-health/medgemma; https://arxiv.org/abs/2507.05201 (MedGemma technical report); https://arxiv.org/abs/2604.05081 (MedGemma 1.5 technical report)
- https://arxiv.org/abs/2503.19786 (Gemma 3 technical report); https://arxiv.org/abs/2607.02770 (Gemma 4 technical report)

### UniMedVL and Bagel

- https://huggingface.co/General-Medical-AI/UniMedVL; https://github.com/uni-medical/UniMedVL; https://arxiv.org/abs/2510.15710
- https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT; https://arxiv.org/abs/2505.14683 (Bagel)

### Medical VLMs and baselines

- https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b, https://arxiv.org/abs/2306.00890; https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B, https://arxiv.org/abs/2406.19280; https://huggingface.co/lingshu-medical-mllm/Lingshu-7B, https://huggingface.co/lingshu-medical-mllm/Lingshu-32B, https://arxiv.org/abs/2506.07044
- https://github.com/alibaba-damo-academy/MedEvalKit; https://huggingface.co/ZJU-AI4H/Hulu-Med-7B, https://github.com/ZJUI-AI4H/Hulu-Med, https://arxiv.org/abs/2510.08668; https://huggingface.co/JZPeterPan/MedVLM-R1, https://arxiv.org/abs/2502.19634
- https://huggingface.co/StanfordAIMI/CheXagent-2-3b, https://arxiv.org/abs/2401.12208; https://github.com/chaoyi-wu/RadFM, https://arxiv.org/abs/2308.02463; https://github.com/snap-stanford/med-flamingo, https://arxiv.org/abs/2307.15189
- https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224, https://arxiv.org/abs/2303.00915; https://huggingface.co/microsoft/maira-2, https://arxiv.org/abs/2406.04449; https://huggingface.co/microsoft/BiomedParse, https://arxiv.org/abs/2405.12971
- https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct, https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct; https://huggingface.co/OpenGVLab/InternVL3-8B; https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

### Text LLMs and benchmarks

- https://huggingface.co/epfl-llm/meditron-70b, https://arxiv.org/abs/2311.16079; https://huggingface.co/BioMistral/BioMistral-7B, https://arxiv.org/abs/2402.10373; https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B
- https://huggingface.co/m42-health/Llama3-Med42-70B, https://arxiv.org/abs/2408.06142; https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-72B, https://arxiv.org/abs/2412.18925; https://huggingface.co/UCSC-VLAA/m1-7B-23K, https://arxiv.org/abs/2504.00869
- https://huggingface.co/baichuan-inc/Baichuan-M2-32B, https://arxiv.org/abs/2509.02208; https://huggingface.co/baichuan-inc/Baichuan-M3-235B, https://arxiv.org/abs/2602.06570; https://huggingface.co/Intelligent-Internet/II-Medical-8B
- https://arxiv.org/abs/2009.13081 (MedQA); https://arxiv.org/abs/2203.14371 (MedMCQA); https://arxiv.org/abs/1909.06146 (PubMedQA)
- https://arxiv.org/abs/2205.09148 (DDXPlus); https://arxiv.org/abs/2405.07960 (AgentClinic); https://openai.com/index/healthbench/ (HealthBench)

### OCR

- https://github.com/tesseract-ocr/tesseract; https://github.com/PaddlePaddle/PaddleOCR, https://huggingface.co/PaddlePaddle/PaddleOCR-VL, https://arxiv.org/abs/2606.13108 (PP-OCRv6); https://github.com/mindee/doctr
- https://github.com/datalab-to/surya, https://huggingface.co/datalab-to/chandra; https://huggingface.co/microsoft/trocr-base-handwritten, https://arxiv.org/abs/2109.10282; https://huggingface.co/naver-clova-ix/donut-base, https://arxiv.org/abs/2111.15664
- https://huggingface.co/facebook/nougat-base; https://huggingface.co/stepfun-ai/GOT-OCR2_0, https://arxiv.org/abs/2409.01704; https://huggingface.co/allenai/olmOCR-2-7B-1025, https://huggingface.co/datasets/allenai/olmOCR-bench
- https://huggingface.co/rednote-hilab/dots.ocr; https://huggingface.co/deepseek-ai/DeepSeek-OCR, https://huggingface.co/deepseek-ai/DeepSeek-OCR-2; https://huggingface.co/zai-org/GLM-OCR, https://arxiv.org/abs/2603.10910
- https://huggingface.co/nanonets/Nanonets-OCR2-3B; https://huggingface.co/ibm-granite/granite-docling-258M; https://huggingface.co/microsoft/Florence-2-large
- https://huggingface.co/lightonai/LightOnOCR-1B-1025; https://github.com/opendatalab/OmniDocBench

### Imaging tools

- https://github.com/mlmed/torchxrayvision, https://arxiv.org/abs/2111.00595; https://github.com/Project-MONAI/MONAI, https://github.com/Project-MONAI/model-zoo; https://github.com/MIC-DKFZ/nnUNet, https://arxiv.org/abs/1809.10486
- https://github.com/wasserth/TotalSegmentator, https://arxiv.org/abs/2208.05868; https://github.com/bowang-lab/MedSAM, https://github.com/bowang-lab/MedSAM2; https://github.com/uni-medical/SAM-Med3D
