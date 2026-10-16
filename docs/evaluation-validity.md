# Evaluation validity: what a medbench score can show

This page lists what we checked about the benchmark's labels, splits,
sample sizes, scoring rules and sources. It also lists what we could not
check. It answers [issue #7](https://github.com/konard/medical-neural-network-ml/issues/7).

The short version: the real teaching images and pages are a **smoke test**.
They show that a model can read real files and answer in a usable format.
They are far too few, and too easy, to rank models or support a clinical
claim. The labels were reviewed by the maintainers, **not by clinicians**.

## Label review

`src/medbench/data/label_review.json` has one entry for each of the 37 real
images and pages. You can print it with:

```bash
medbench review                     # every task
medbench review --task imaging-xray # one task
```

Each entry records:

| Field | Meaning |
|---|---|
| `view` | Projection, sequence or plane, as stated by the source. Two X-rays are **AP**, not PA: on AP, heart and mediastinum look larger. The X-ray prompt no longer says every X-ray is frontal PA. |
| `laterality` | The side of the finding and how it was read. Sides are the **patient's** side. On X-ray and MRI that is mirrored in the image (radiological convention). Ultrasound sides that the source does not state are not scored. |
| `label_source` | Where the label comes from: the Commons uploader's description, the RxHandBD dataset's own label, or a proofread Wikisource transcription. |
| `source_group` | Items that may share a patient, uploader or source. Splits keep a group together. |
| `uncertainty` | Known doubts, e.g. "projection not stated", "side cannot be checked from pixels", "only *mass* is scorable, not the histology". |
| `also_present` | Other true findings that a good answer may name (e.g. mediastinal shift with a large effusion). |

How the review was done (`method` in the file): each label was read from the
source page and compared with the image or scan by eye; doubts were written
down. Nobody with medical training checked the images. `clinician_reviewed`
is `false` and stays `false` until a clinician signs the file. A test
(`tests/test_medbench_validity.py`) checks that the review covers exactly the
real imaging and OCR samples and that its sides match the scored `side`.

### Class balance and source overlap

| Task | Normal | Abnormal |
|---|---:|---:|
| `imaging-xray` | 3 | 6 |
| `imaging-mri` | 3 | 4 |
| `imaging-ultrasound` | 3 | 7 |

All three tasks have more abnormal than normal items. A model that always
answers "abnormal" gets 67–70% accuracy, so read sensitivity and
specificity, not accuracy.

The items are not independent. Hellerhoff uploaded 8 of the 26 images
(X-ray, MRI and ultrasound). Häggström, Heilman, Novak and the
`ptrump16` account uploaded 2–4 each. Images from one uploader share
equipment, style and post-processing. `medbench.validity.LabelReview.group_of`
returns that group, so a split never puts one uploader's images on both
sides.

## Splits

`medbench.validity.split_of` gives each group a fixed split (60% train,
20% dev, 20% test) from a SHA-256 hash of `medbench-split-v1:<group>`. The
same group always lands in the same split. Adding data does not move old
items. `grouped_split` applies it to a list and never splits a group.

* **DDXPlus.** Each row is a separate synthetic patient, so the row id is
  the group. `medbench run --task diagnosis-ddxplus --split dev` takes dev
  patients only. Tune prompts and parsers on `dev` and report on `test`. If
  you leave out `--split`, the task keeps the old behaviour: the first rows
  of the test file, so the numbers compare with older reports.
* **Teaching images and pages.** With 14 source groups, a 60/20/20 split
  gives 30 train, 6 dev and 1 test item. That is not a usable split. These
  37 items are **test-only**: do not tune prompts, parsers or thresholds on
  them, and `--split` does not apply to them. The prompts and parsers were
  written before these items were scored, and no parameter was fitted to
  them. Any later change that is checked against them must be reported as
  such.

## Sample-size targets

`medbench.validity.required_sample_size` finds the smallest test set whose
95% Wilson interval is at most ± a chosen half-width. `medbench review`
prints the targets:

| Task | Metric | Expected | ± | Needed | Have now |
|---|---|---:|---:|---:|---:|
| `diagnosis-ddxplus` | top-1 accuracy | 0.5 | 0.05 | 381 patients | 134,529 in the test file (DDXPlus paper) |
| imaging (per finding) | sensitivity and specificity | 0.8 | 0.10 | 60 per class | 3–7 per class |
| OCR | key-term recall | 0.8 | 0.05 | 245 key terms | 45 |

The half-widths are planning choices, not clinical standards. For ±3 points
at 50%, 1,064 items are needed. Imaging needs real public sets (e.g.
NIH ChestX-ray14, VinDr-CXR, BUSI) with their own licenses and splits; see
[data-sources.md](https://github.com/konard/medical-neural-network-ml/blob/main/docs/case-studies/issue-1/data-sources.md).

## Metrics and answer quality

Every report has:

* `requested_items`, `items_run` and `coverage` (`items_run / items_requested`),
  so a run that stops at its time budget is visible.
* A Wilson 95% interval for each proportion (`metrics.wilson_interval`).
  Unlike the bootstrap, it does not shrink to a point when 2 of 2 answers
  are right: (0.34, 1.0).
* `invalid_rate` (the parser found no answer), `abstention_rate` (the
  answer declines or says it is unsure), `empty_rate` and `truncation_rate`
  (the answer used all of `max_tokens`).

### Ranked diagnosis parser

`parse_ranked_diagnoses` matches the 49 DDXPlus condition names and their
aliases as whole words only. It skips lines that rule a diagnosis out. Tests
(`tests/test_medbench_ddxplus_parser.py`) cover:

* each of the 49 names and every alias;
* adversarial outputs that the old substring match got wrong: "measles"
  read as SLE, "parainfluenza" as influenza, and "Not pneumonia" as
  pneumonia;
* a network test that checks the name list against the pinned
  `release_conditions.json`.

`experiments/issue11_ddxplus_parser_probe.py` shows the old and the new
parser side by side.

### Negation, doses and OCR

`medbench.clinical_text` replaces plain keyword overlap with explicit rules:

* `affirmed_mention` is a NegEx-style check (Chapman et al., 2001). "No
  pneumothorax" no longer earns credit for finding a pneumothorax. It also
  handles pseudo-negations ("cannot be ruled out") and clause breaks
  ("but").
* `extract_doses` and `compare_doses` read number and unit pairs with
  canonical units, so "5 mg" vs "5 mcg" counts as a unit error and "50 mg"
  vs "500 mg" as a value error.
* `normalise_ocr` applies Unicode NFKC, case folding and number/unit spacing
  before CER and WER. Raw CER is still reported.
* Imaging side accuracy uses the patient-side convention above.

These rules were checked on hand-written examples in
`tests/test_medbench_clinical_text.py`. Those examples are **not
expert-reviewed**. Known gaps: uncertainty words ("possible", "likely") are
not graded, finding synonyms are a short fixed list, and there is no
finding-level localisation beyond left/right/both. An expert-reviewed
example set is still open (see "What is still open").

## Source terms and provenance

* Original code stays under the Unlicense. Data and model terms are kept
  separately in [THIRD_PARTY_NOTICES.md](https://github.com/konard/medical-neural-network-ml/blob/main/THIRD_PARTY_NOTICES.md).
* **Not every source is public domain.** The 37 teaching items are under
  CC BY 4.0 (12), CC BY-SA 4.0 (9), CC BY-SA 3.0 (6), CC0 (5), public domain
  (5), CC BY 3.0 (2) and CC BY 2.5 (1). DDXPlus and RxHandBD are CC BY 4.0.
* Labels come from the uploaders' descriptions and the dataset authors.
  Short quotes and the derived labels in `real_samples.json` and
  `label_review.json` keep their source's attribution. Images and crops
  derived from CC BY-SA files keep the share-alike license. `medbench fetch`
  writes `ATTRIBUTION.md`, and each report lists its `sources`.
* The five history pages are public domain in the US. Fothergill died in
  1955, so his 1907 book became public domain in "life + 70" countries only
  on 1 January 2026. Check your own jurisdiction before you redistribute
  the scans.

## Baselines and ready evaluation libraries

| Tool | License (checked on GitHub) | Use here |
|---|---|---|
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Apache-2.0 | `--backend tesseract`; the `ocr-baseline` CI job runs it on both OCR tasks and uploads JSON reports. |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 | Next OCR baseline; needs a GPU or a slow CPU run. |
| [TorchXRayVision](https://github.com/mlmed/torchxrayvision) | Apache-2.0 (code; check each model's license) | Specialist chest X-ray classifier baseline for `imaging-xray`. |
| [MedEvalKit](https://github.com/alibaba-damo-academy/MedEvalKit) | No LICENSE file (README badge says Apache-2.0) | Reference for medical VQA sets; do not vendor it until the license is clear. |
| [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | Apache-2.0 | Wide VLM benchmark runner; useful for public medical VQA sets. |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | MIT | Same, built on lm-evaluation-harness. |
| [MultiMedEval](https://github.com/corentin-ryr/MultiMedEval) | MIT | Medical VLM suite (VQA, report generation, classification). |
| [HealthBench (simple-evals)](https://github.com/openai/simple-evals) | MIT | Rubric-graded health conversations; text only. |
| [medspaCy](https://github.com/medspacy/medspacy) / [negspaCy](https://github.com/jenojp/negspacy) | MIT | Fuller ConText/NegEx; our rules are a small dependency-free subset. |
| [Hugging Face evaluate](https://github.com/huggingface/evaluate) | Apache-2.0 | Standard CER/WER for cross-checking our edit distance. |

medbench does not replace these tools. It runs a small set of real,
license-checked files on the hardware we have, with the same answer-quality
counters for every backend. The library runs and the specialist baselines
are the next step in [research-plan.md](https://github.com/konard/medical-neural-network-ml/blob/main/docs/case-studies/issue-1/research-plan.md).

Reproducible reports: the `ocr-baseline` job in
`.github/workflows/model-tests.yml` installs Ubuntu's `tesseract-ocr` and
runs every OCR item. It uploads the reports and `tesseract-version.txt` as
the `medbench-ocr-baseline-reports` artifact. Model reports from the GPU
runner are uploaded the same way.

## What is still open

* A clinician has not reviewed the labels or the scoring examples.
  `clinician_reviewed` must stay `false` until one does.
* The teaching set is far below the sample-size targets. Public sets with
  patient-level splits are needed before any model comparison.
* PaddleOCR, TorchXRayVision and the evaluation libraries above have not
  been run in this repository yet.
