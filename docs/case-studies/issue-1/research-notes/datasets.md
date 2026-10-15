# Medical datasets catalogue (issue #1)

Research notes for [issue #1](https://github.com/konard/medical-neural-network-ml/issues/1).
The issue asks for public domain or permissively licensed datasets that cover
typical medical-document processing. That means OCR of medical histories,
handwritten prescriptions, X-ray, MRI/CT and ultrasound reading, and first-pass
diagnosis that returns several likely options from symptoms.

Compiled 2026-10-14. Licenses were checked on the original publisher's page
or through an API (Hugging Face, Kaggle, Zenodo, Figshare, OSF, GitHub, PhysioNet)
where I could. Anything I could not confirm at the source is marked **(unverified)**.
Treat this as engineering notes, not legal advice. Before shipping a product,
re-read the license on the original source.

## How to read this file

**Access types**

- **Open**: direct download, no account needed.
- **Registration**: a free account or a click-through form.
- **Credentialed**: identity check, ethics training (e.g. CITI) and a signed
  Data Use Agreement (DUA), e.g. PhysioNet credentialed access.
- **Gated HF**: Hugging Face "accept terms" gate. It is usually automatic, but
  it is still a contract.

**License tiers.** Every table puts the truly permissive rows first, then a
separator row, then the restricted rows.

- **Permissive**: CC0, public domain, CC BY, CC BY-SA, ODbL, MIT, Apache-2.0.
  Commercial use is allowed. CC BY-SA and ODbL are share-alike: derived
  *datasets* must keep the same license. Trained model weights are generally
  not considered a derived dataset, but that is untested in court.
- **Restricted**: non-commercial (NC), no-derivatives (ND), research-only
  agreements, credentialed DUAs, and "no license stated". No license means
  all rights reserved by default.

**"Good for" codes**

- **CI**: small enough (under about 50 MB, or easy to subset) and permissive
  enough to download inside a public CI job.
- **Eval**: big and clean enough for a real benchmark number.
- **FT**: big enough to fine-tune on, and the license allows it.
- In restricted rows, "(internal)" means internal research use only. Never
  redistribute or ship a model trained on it without legal review.

**Mirror warning.** Kaggle and Hugging Face license tags are set by whoever
uploaded the copy, and nobody checks them. Several popular re-uploads say "CC0"
or "MIT" when the original is non-commercial or credentialed (see
[Surprises](#surprises-datasets-commonly-assumed-open-that-are-not)). The
license of the **original** source always wins.

---

## 1. Text, OCR and clinical NLP

| Dataset | Task | Size | License | Access | Commercial? | URL | Good for |
|---|---|---|---|---|---|---|---|
| Synthea (generator) | Synthetic EHR: FHIR R4/STU3/DSTU2, Bulk FHIR ndjson, C-CDA, CSV | Unlimited (you generate it) | Apache-2.0 (code). The README states no separate license for generated output. | Open (GitHub) | Yes | https://github.com/synthetichealth/synthea | CI, FT (structured records; no free-text notes beyond templates) |
| Synthea Coherent Data Set | Synthetic multimodal patients: FHIR, DICOM, genomics, ECG, simple notes | Approx. 1,278 patients (unverified) | CC BY 4.0 (AWS Open Data registry) | Open (`s3://synthea-open-data/coherent/`, `--no-sign-request`) | Yes | https://registry.opendata.aws/synthea-coherent-data | CI (subset), Eval of pipelines, FT |
| NHS England synthetic clinical notes | Fully synthetic clinical notes | Small (approx. 1k rows, unverified) | MIT (HF tag and GitHub repo) | Open | Yes | https://huggingface.co/datasets/NHSEDataScience/synthetic_clinical_notes ; https://github.com/nhsengland/synthetic_clinical_notes | CI, light FT |
| MIMIC-IV Clinical Database Demo v2.2 | Structured EHR for 100 patients (no notes, unverified) | 15.5 MB | ODbL v1.0 | Open (PhysioNet) | Yes (share-alike for databases) | https://physionet.org/content/mimic-iv-demo/2.2/ | CI (schema tests for MIMIC-style pipelines) |
| MIMIC-III Clinical Database Demo v1.4 | Structured EHR for 100 patients (notes table empty, unverified) | 103 MB | ODbL | Open (PhysioNet) | Yes (share-alike) | https://physionet.org/content/mimiciii-demo/1.4/ | CI |
| CORD (v1/v2) | Receipt OCR plus key-information extraction (forms proxy) | Approx. 1,000 receipts (unverified count) | CC BY 4.0 | Open (GitHub / HF `naver-clova-ix/cord-v2`) | Yes | https://github.com/clovaai/cord | CI, Eval, FT for form/key-value extraction |
| **Restricted / research-only / credentialed** | | | | | | | |
| MIMIC-IV-Note v2.2 | De-identified discharge summaries and radiology reports | 331,794 discharge notes (145,915 patients); 2,321,355 radiology reports | PhysioNet Credentialed Health Data License 1.5.0 | Credentialed + DUA | Research only. No sharing; no third-party online LLM APIs (PhysioNet policy, unverified this session) | https://physionet.org/content/mimic-iv-note/2.2/ | Eval, FT (internal) |
| MIMIC-III v1.4 | Full EHR including NOTEEVENTS | Approx. 2M notes, approx. 46k patients (unverified) | PhysioNet Credentialed Health Data License 1.5.0 | Credentialed + DUA | Research only | https://physionet.org/content/mimiciii/1.4/ | Eval, FT (internal) |
| MIMIC-IV-ED | Emergency department triage and visits | Approx. 425k ED stays (unverified) | PhysioNet Credentialed | Credentialed + DUA | Research only | https://physionet.org/content/mimic-iv-ed/ | Eval (triage) (internal) |
| MedNLI | Clinical natural language inference (from MIMIC) | Approx. 14k pairs (unverified) | PhysioNet Credentialed | Credentialed + DUA | Research only | https://physionet.org/content/mednli/ | Eval (internal) |
| RadGraph | Entity and relation graph over radiology reports | Approx. 600 annotated reports plus silver set (unverified) | PhysioNet Credentialed | Credentialed + DUA | Research only | https://physionet.org/content/radgraph/ | Eval (internal) |
| n2c2 / i2b2 NLP challenges | De-id, medications, relations, smoking, obesity, etc. | Hundreds to approx. 1.5k notes per task (unverified) | Harvard DBMI DUA | Registration + per-user DUA | No product use. No redistribution, not even on GitHub. | https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/ | Eval (internal) |
| Asclepius Synthetic Clinical Notes | Synthetic notes plus instruction QA (made from PMC-Patients) | Approx. 158k notes | CC BY-NC-SA 4.0 | Open (HF) | No | https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes | Eval, FT (non-commercial) |
| PMC-Patients (V1/V2) | Patient summaries extracted from PMC case reports | 167k (V1), 250,294 (V2) | CC BY-NC-SA 4.0 | Open (HF) | No | https://huggingface.co/datasets/zhengyun21/PMC-Patients | Eval, FT (non-commercial) |
| augmented-clinical-notes | PMC-Patients notes plus GPT-generated dialogues and summaries | Approx. 30k (unverified) | HF tag says MIT, but it derives from PMC-Patients (CC BY-NC-SA) plus GPT outputs | Open (HF) | Treat as No | https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes | Eval (non-commercial) |
| MTSamples | Sample medical transcription reports in 40 specialties | Approx. 5k reports (Kaggle CSV 17 MB) | No open license. Site allows "print, share, or link ... for educational purposes" with credit. Kaggle copy claims CC0; HF `harishnair04/mtsamples` claims Apache-2.0. Both are uploader claims. | Open (scrape or mirrors) | No (treat as copyrighted) | https://mtsamples.com/ ; https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions | CI-sized, but use only for internal experiments |
| MedDialog (EN/ZH) | Doctor-patient dialogues | EN approx. 0.26M, ZH approx. 1.1M dialogues | HF license "unknown". The card says copyright belongs to healthcaremagic.com, icliniq.com and haodf.com. | Open (HF `bigbio/meddialog`, `UCSD26/medical_dialog`) | No | https://huggingface.co/datasets/bigbio/meddialog | Eval (internal) |
| FUNSD | Form understanding (noisy scanned forms) | 199 forms, 31,485 words, 9,707 entities | Official terms: "non-commercial, research and educational purposes" only. The HF `nielsr/funsd` copy has no tag. | Open | No | https://guillaumejaume.github.io/FUNSD/ | CI-sized, but non-commercial. Prefer CORD. |
| SROIE (ICDAR 2019) | Receipt OCR and key-value extraction | 1,000 receipts | Original RRC terms (unverified, research). The GitHub repo `zzzDavid/ICDAR-2019-SROIE` is MIT, but it is a third-party copy. Kaggle `urbikn/sroie-datasetv2` says "Other". | Registration (RRC) | Unclear, treat as No | https://rrc.cvc.uab.es/?ch=13 | Eval (internal) |
| DocVQA | Visual QA over scanned documents (UCSF Industry Documents Library) | Approx. 12k images, 50k questions (unverified) | RRC portal terms (unverified, research use) | Registration | Unclear, treat as No | https://rrc.cvc.uab.es/?ch=17 | Eval (internal) |

Notes:

- There is no large, permissive, *real* clinical-notes corpus. Real notes
  always come with a DUA (MIMIC, n2c2). Permissive options are synthetic:
  Synthea, Coherent and the NHS synthetic notes.
- For OCR of printed medical histories, the practical route is to render
  synthetic records (Synthea plus an HTML/PDF template, then rasterise and add
  noise). Use CORD for real-world layout and key-value evaluation.

## 2. Handwriting and prescriptions

| Dataset | Task | Size | License | Access | Commercial? | URL | Good for |
|---|---|---|---|---|---|---|---|
| RxHandBD | Handwritten prescription word recognition (Bangladesh) | 5,578 word crops at 512x512 (4,463 train / 1,115 test), 1,559 unique words | CC BY 4.0 (Mendeley Data) | Open | Yes | https://data.mendeley.com/datasets/dsb5r6vskg | CI (subset), Eval, FT |
| Doctor's Handwritten Prescription BD | Medicine-name word classification/OCR | 4,680 word images, 78 medicine classes (60 each), 128x128 | Mendeley v2 (10 Aug 2026): CC BY 4.0, BUT the page also says "free to use for educational and research purposes only". Kaggle says "Database: Open Database, Contents: (c) Original Authors". HF mirror `lance-format/handwriting-ocr` claims ODbL. | Open | Conflicting. Ask the authors before commercial use. | https://data.mendeley.com/datasets/zjjtptvn6f/2 ; https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset | CI (21 MB), Eval |
| Kaggle handwriting-recognition (names) | Handwritten name transcription (generic HTR) | Approx. 400k images, 1.4 GB | CC0 (uploader claim, original provenance unverified) | Open (Kaggle) | Probably yes (unverified) | https://www.kaggle.com/datasets/landlord/handwriting-recognition | FT (HTR pre-training) |
| HF Medical_Prescription_Handwritten_Words | Handwritten prescription words | Under 1k (unverified) | MIT (HF tag; provenance unverified) | Open (HF) | Yes per tag (unverified) | https://huggingface.co/datasets/avi-kai/Medical_Prescription_Handwritten_Words ; https://huggingface.co/datasets/MMMuzammil/Medical_Prescription_Handwritten_Words | CI |
| Synthetic handwriting (see group 8) | Render prescriptions or histories in handwriting fonts and models | Unlimited | MIT tools (TRDG, text-to-handwriting, HWT, DiffusionPen) | Open | Yes | See group 8 | CI, FT |
| **Restricted / research-only / unclear** | | | | | | | |
| IAM Handwriting Database | English handwritten text lines and words (generic HTR benchmark) | 657 writers, 1,539 pages, 13,353 lines, 115,320 words | "may be used for non-commercial research purpose only". HF `Teklia/IAM-line` claims MIT (derivative, not authoritative). | Registration | No | https://fki.tic.heia-fr.ch/databases/iam-handwriting-database | Eval (non-commercial) |
| RIMES | French handwritten letters and lines | Approx. 12k lines (unverified) | Original terms (unverified). HF `Teklia/RIMES-2011-line` claims MIT; Zenodo "RIMES complete" 10812725 is listed as open (unverified). | Open via mirrors | Unclear | https://huggingface.co/datasets/Teklia/RIMES-2011-line ; https://zenodo.org/records/10812725 | Eval |
| Jadavpur Univ. prescription handwritten vs printed | Separating handwritten from printed text on prescriptions | 11,340 images | License unverified (IEEE DataPort) | Registration (IEEE DataPort) | Unverified | https://ieee-dataport.org (search "handwritten printed prescription") | Eval (internal) |
| 100-handwritten-medical-records (HF) | Handwritten medical record pages | 100 pages | CC BY-ND 4.0 (HF tag). Other similar uploads have no license. | Open (HF) | ND: no modified copies | https://huggingface.co/datasets/chaithanyakota/100-handwritten-medical-records | CI and Eval only (no derivatives) |

Notes:

- Real handwritten prescriptions with an unambiguous permissive license are
  rare. RxHandBD is the cleanest. Expect a domain gap: all real sets are
  Bangladeshi prescriptions with mostly brand-name drugs.
- Plan on synthetic handwriting (group 8) for full-page prescriptions, and
  keep a small real set (RxHandBD) for evaluation.

## 3. X-ray

| Dataset | Task | Size | License | Access | Commercial? | URL | Good for |
|---|---|---|---|---|---|---|---|
| NIH ChestX-ray14 | Chest X-ray, 14 multi-label findings plus boxes for approx. 1k images | 112,120 images, 30,805 patients (approx. 45 GB) | README: "There are no restrictions on the use of the NIH chest x-ray images". Attribution required (link to the NIH site, cite Wang et al. CVPR 2017, acknowledge NIH Clinical Center). Kaggle copy says CC0. | Open (NIH Box, Kaggle `nih-chest-xrays/data`) | Yes (with attribution) | https://nihcc.app.box.com/v/ChestXray-NIHCC | CI (pick a few files), Eval, FT |
| Kermany pediatric CXR and OCT (v2) | Pneumonia vs normal (pediatric CXR); retinal OCT classes | Approx. 5.8k CXR plus approx. 109k OCT (unverified counts) | CC BY 4.0 (Mendeley, verified) | Open | Yes | https://data.mendeley.com/datasets/rscbjbr9sj/2 | CI (subset), Eval, FT |
| RSNA Pneumonia Detection 2018 | Pneumonia opacity boxes (images from NIH CXR14) | Approx. 30k images (unverified) | RSNA page allows academic use and "other commercial or non-commercial purposes as long as you meet the following attribution requirements". The original Kaggle competition rules were non-commercial (conflict). | Open (direct S3 under `s3.amazonaws.com/east1.public.rsna.org/AI/2018/`) | Yes per RSNA page (with attribution) | https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018 | Eval, FT |
| COVID-QU-Ex | COVID/non-COVID pneumonia/normal CXR plus lung and infection masks | Approx. 33,920 images (unverified), 1.2 GB | CC BY-SA 4.0 (Kaggle) | Open (Kaggle) | Yes (share-alike) | https://www.kaggle.com/datasets/anasmohammedtahir/covidqu | Eval, FT (segmentation) |
| MedNIST | 6-class toy set (AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT), 64x64 | Approx. 58k images (unverified) | CC BY-SA 4.0 (Mayo Clinic / Dr. B. Erickson, via MONAI) | Open (`monai.apps.MedNISTDataset`) | Yes (share-alike) | https://github.com/Project-MONAI/tutorials | CI |
| Montgomery County and Shenzhen TB CXR | TB vs normal, lung masks (Montgomery) | Montgomery 138 (80 normal / 58 TB); Shenzhen 662 (326 / 336) | NLM gives no formal license; citation requested (unverified). Kaggle copies: "Unknown"; `nikhilpandey360/chest-xray-masks-and-labels` claims CC0. | Open | Probably yes, US government work (unverified) | https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html | CI, Eval |
| **Restricted / research-only / credentialed** | | | | | | | |
| CheXpert | 14 CXR observations, uncertainty labels | 224,316 images, 65,240 patients | Stanford AIMI Research Use Agreement (non-commercial). Kaggle `ashery/chexpert` ("CheXpert-v1.0-small", 11.5 GB) claims CC0, which is a laundered re-upload. | Registration (Stanford AIMI) | No | https://stanfordmlgroup.github.io/competitions/chexpert/ | Eval, FT (non-commercial) |
| CheXpert Plus | CXR image plus radiology report pairs | 223,462 pairs, 187,711 studies, 64,725 patients | Stanford AIMI RUA, non-commercial, no redistribution | Registration (Redivis) | No | https://doi.org/10.71718/6nvz-pm34 | Eval (report generation) (non-commercial) |
| MURA | Musculoskeletal radiograph abnormality | 40,561 images, 14,863 studies, 12,173 patients | Stanford AIMI RUA, non-commercial | Registration (Redivis) | No | https://stanfordmlgroup.github.io/competitions/mura/ | Eval (non-commercial) |
| MIMIC-CXR v2.1.0 / MIMIC-CXR-JPG | CXR plus free-text reports | 377,110 images, 227,835 studies | PhysioNet Credentialed Health Data License 1.5.0 | Credentialed + DUA | Research only | https://physionet.org/content/mimic-cxr/2.1.0/ ; https://physionet.org/content/mimic-cxr-jpg/2.1.0/ | Eval, FT (internal) |
| Chest ImaGenome | Scene graphs over MIMIC-CXR | Approx. 242k images (unverified) | PhysioNet Credentialed | Credentialed + DUA | Research only | https://physionet.org/content/chest-imagenome/ | Eval (internal) |
| VinDr-CXR v1.0.0 | CXR with 22 local and 6 global labels, boxes | 18,000 DICOM (15k train / 3k test) | PhysioNet Credentialed Health Data License 1.5.0 | Credentialed + DUA | Research only | https://physionet.org/content/vindr-cxr/1.0.0/ | Eval (internal) |
| VinDr-PCXR / VinDr-SpineXR / VinDr-Mammo | Pediatric CXR, spine X-ray, mammography | Approx. 9k / 10k / 20k images (unverified) | PhysioNet Restricted Health Data License 1.5.0 | Restricted (PhysioNet) | Research only | https://physionet.org/content/vindr-spinexr/ | Eval (internal) |
| PadChest | Spanish CXR with 174 findings, reports in Spanish | 160,868 images, approx. 67k patients | BIMCV Research Use Agreement. Research only, no sale, non-academic use needs approval, no redistribution, derivatives restricted. | Registration + agreement | No | https://bimcv.cipf.es/bimcv-projects/padchest/ | Eval (internal) |
| IU X-ray (Open-i, Indiana) | CXR plus reports (report generation) | 7,470 images, 3,955 reports | CC BY-NC-ND 4.0 (Kaggle `raddar/chest-xrays-indiana-university`; HF `ykumards/open-i` card text). The original Open-i site was under maintenance, so not confirmed at source (unverified). | Open (mirrors) | No; also no derivatives | https://openi.nlm.nih.gov/ | Eval only |
| JSRT | Lung nodule CXR | 247 images (154 nodule / 93 non-nodule), 2048x2048, 12-bit | Registration; terms unverified (research). Kaggle `raddar/nodules-in-chest-xrays-jsrt` says "Unknown". | Registration | Unverified | http://db.jsrt.or.jp/eng.php | Eval (internal) |
| QaTa-COV19 | COVID CXR segmentation | 5.3 GB | CC BY-NC-SA 3.0 IGO (Kaggle) | Open (Kaggle `aysendegerli/qatacov19-dataset`) | No | https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset | Eval (non-commercial) |
| COVID-19 Radiography DB / TB Chest X-ray DB | COVID/viral/normal CXR; TB CXR | Approx. 21k / 7k images (unverified) | Kaggle: "(c) Original Authors" (no open license) | Open (Kaggle) | No (not granted) | https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database | Eval (internal) |

Notes:

- The only large, real, permissive CXR sources are NIH CXR14 (and RSNA
  Pneumonia, built on it), Kermany pediatric CXR and COVID-QU-Ex.
- Every report-paired CXR set (MIMIC-CXR, CheXpert Plus, PadChest, IU X-ray)
  is non-commercial or credentialed. For report generation, permissive
  training data would have to come from Synthea Coherent DICOM or synthetic
  reports.

## 4. MRI and CT

| Dataset | Task | Size | License | Access | Commercial? | URL | Good for |
|---|---|---|---|---|---|---|---|
| Medical Segmentation Decathlon (MSD) | 10 segmentation tasks: 01 BrainTumour MRI 750; 02 Heart MRI 30; 03 Liver CT 201; 04 Hippocampus MRI 394; 05 Prostate MRI 48; 06 Lung CT 96; 07 Pancreas CT 420; 08 HepaticVessel CT 443; 09 Spleen CT 61; 10 Colon CT 190 (case counts unverified) | Task04 Hippocampus approx. 27 MB (unverified); others 0.1 to 7 GB | CC BY-SA 4.0 (verified on medicaldecathlon.com) | Open (Google Drive / AWS; `monai.apps.DecathlonDataset`) | Yes (share-alike) | http://medicaldecathlon.com/ | CI (Task04, Task09), Eval, FT |
| TotalSegmentator v2 | Whole-body CT segmentation, 117 structures | 1,228 CT, 23.6 GB | CC BY 4.0 (Zenodo 10047292). v1 (Zenodo 6802614): 1,204 CT, 104 structures, CC BY 4.0. The tool is Apache-2.0. | Open (Zenodo) | Yes | https://zenodo.org/records/10047292 | Eval, FT (a small subset can serve CI) |
| AMOS | Abdominal multi-organ segmentation, 15 organs | 500 CT plus 100 MRI (unverified), 24.2 GB | CC BY 4.0 (Zenodo 7262581 / 7155725) | Open (Zenodo) | Yes | https://zenodo.org/records/7262581 | Eval, FT |
| LIDC-IDRI | Lung nodule detection and characterisation, CT | 1,018 cases, approx. 125 GB | CC BY 3.0 (TCIA collection page, verified) | Open (TCIA / NBIA; `idc-index`) | Yes | https://www.cancerimagingarchive.net/collection/lidc-idri/ | Eval, FT |
| LUNA16 | Lung nodule detection (LIDC subset) | 888 CT (unverified); 47.7 GB part 1 | CC BY 4.0 (Zenodo 3723295) | Open (Zenodo) | Yes | https://zenodo.org/records/3723295 | Eval, FT |
| IXI | Healthy brain MRI: T1, T2, PD, MRA, DTI (NIfTI) | Nearly 600 subjects | CC BY-SA 3.0 | Open | Yes (share-alike) | https://brain-development.org/ixi-dataset/ | CI (a few volumes), Eval, FT (self-supervised, synthesis) |
| Kaggle Brain Tumor MRI (masoudnickparvar) | 4-class brain tumour slice classification | Approx. 7k images, 168 MB | CC BY 4.0 (uploader claim; it merges figshare, SARTAJ and Br35H, so provenance is unclear) | Open (Kaggle) | Per tag yes (unverified) | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset | CI, quick Eval |
| Kaggle Brain Tumor Classification (sartajbhuvaji) | 4-class slices | 93 MB | MIT (uploader claim) | Open (Kaggle) | Per tag yes (unverified) | https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri | CI |
| **Restricted / research-only / mixed** | | | | | | | |
| BraTS (2020 to 2024) | Glioma (and other tumour) MRI segmentation | Approx. 1,250 to 2,000 cases per year (unverified) | Mixed. Some BraTS 2021 parts on TCIA are CC BY 4.0, some newer validation files are "TCIA Restricted", and source collections carry their own terms. Kaggle `awsaf49/brats20` claims CC0 (42.8 GB); `dschettler8845/brats-2021-task1` says "(c) Original Authors". | Registration (Synapse syn25829067) | Check per release | https://www.synapse.org/brats | Eval, FT (check each year) |
| fastMRI (NYU) | Raw k-space and DICOM reconstruction: knee, brain, prostate, breast | Knee approx. 1,500 raw plus 10,000 DICOM; brain 6,970; prostate 312; breast 300 | NYU Data Sharing Agreement. "Internal research or educational purposes only", no commercial use, no redistribution, destroy data after use. | Registration + agreement | No | https://fastmri.med.nyu.edu/ | Eval, FT (internal) |
| OASIS-1/2/3 | Brain MRI with ageing and Alzheimer's labels | OASIS-3: approx. 1,300 subjects (unverified) | OASIS Data Use Agreement (no re-identification, no face reconstruction, acknowledgment required). Commercial use not stated (unverified). Kaggle `ninadaithal/imagesoasis` claims Apache-2.0; `jboysen/mri-and-alzheimers` claims CC0. | Registration (NITRC-IR / XNAT) | Unclear | https://sites.wustl.edu/oasisbrains/ | Eval (internal) |
| KiTS23 | Kidney tumour CT segmentation | 599 cases (489 train) | Data CC BY-NC-SA 4.0 per the GitHub README (code MIT). Training only for the challenge is not counted as commercial. | Open (GitHub script) | No | https://github.com/neheller/kits23 | Eval, FT (non-commercial) |
| C4KC-KiTS (KiTS19 on TCIA) | Kidney tumour CT | 210 cases (unverified) | TCIA lists CC BY 3.0, which conflicts with KiTS's own NC terms | Open (TCIA) | Conflicting | https://www.cancerimagingarchive.net/collection/c4kc-kits/ | Eval |
| Kaggle Brain MRI tumour detection (navoneel) | Yes/no tumour slices | 253 images, 8.7 MB | "(c) Original Authors" (no grant) | Open (Kaggle) | No | https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection | Internal smoke only |

Notes:

- TCIA hosts hundreds of CT/MRI collections. Many are CC BY 3.0/4.0, but some
  are "TCIA Restricted" or NC. Check each collection. The MIT-licensed
  `idc-index` package (NCI Imaging Data Commons) can query licenses per series
  and download small slices without a full download.
- For CI, MSD Task04 Hippocampus and Task09 Spleen are the smallest
  real-volume permissive sets.

## 5. Ultrasound

| Dataset | Task | Size | License | Access | Commercial? | URL | Good for |
|---|---|---|---|---|---|---|---|
| HC18 | Fetal head circumference measurement and segmentation | 1,334 images (999 train / 335 test, unverified), 0.18 GB | CC BY 4.0 (Zenodo 1327317 / 1322001) | Open (Zenodo) | Yes | https://zenodo.org/records/1327317 | CI, Eval, FT |
| **Restricted / research-only / unclear** | | | | | | | |
| BUSI | Breast ultrasound classification (normal/benign/malignant) plus masks | 780 PNG images from 600 women (133 / 437 / 210) | No license stated at the original source (Data in Brief 2020, doi 10.1016/j.dib.2019.104863) (unverified). Kaggle `aryashah2k/...` claims CC0 (266 MB); `sabahesaraki/...` says "Unknown". Known duplicate problems; a curated 2025 release exists (unverified). | Open (mirrors) | Unclear (not granted) | https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset | Eval (internal). Note the article is open access (Data in Brief), but the data license is still unstated. |
| CAMUS | 2D echocardiography segmentation (apical 2- and 4-chamber views) | 500 patients | CC BY-NC-SA 4.0 per third-party docs (zea); not confirmed on the Creatis page (unverified) | Registration (Creatis Girder) | No | https://www.creatis.insa-lyon.fr/Challenge/camus/ | Eval (non-commercial) |
| EchoNet-Dynamic | Echo video ejection fraction and LV segmentation | 10,030 apical-4-chamber videos | Stanford AIMI non-commercial RUA (code MIT) | Registration | No | https://echonet.github.io/dynamic/ | Eval, FT (non-commercial) |
| TN3K | Thyroid nodule segmentation | 3,493 images (2,303 / 576 / 614) | Unverified | Open (GitHub, from the TRFE-Net paper, unverified) | Unverified | https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation (unverified) | Eval (internal) |
| DDTI | Thyroid ultrasound (nodule) | 347 to 637 images depending on version | Unverified. Kaggle `dasmehdixtr/...` (18.7 MB) says "Other". | Open (Kaggle) | Unverified | https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images | CI-sized, but use internally |
| POCUS COVID-19 lung ultrasound | Lung ultrasound video classification | Over 200 videos | Repo has no license (null). Per-video sources have mixed licenses (unverified). The authors now point to COVID-BLUES (371 videos, 63 patients). | Open (GitHub) | No (not granted) | https://github.com/jannisborn/covid19_ultrasound | Eval (internal) |

Notes:

- Ultrasound is the weakest modality for permissive data. HC18 is the only
  clearly CC BY real set found. Everything else is NC, unlicensed or unclear.
- Fallbacks: simulated ultrasound phantoms (group 8, plus the repo's own
  phantom generator), and the breast and thyroid ultrasound sets inside MedMNIST
  (BreastMNIST is derived from BUSI and published under CC BY 4.0 by MedMNIST).
  Note the upstream BUSI license is unstated.

## 6. Multi-task and VQA benchmarks

| Dataset | Task | Size | License | Access | Commercial? | URL | Good for |
|---|---|---|---|---|---|---|---|
| MedMNIST v2 / MedMNIST+ | 12 2D plus 6 3D standardised classification sets (CXR, derm, OCT, path, blood, organ CT, nodules, etc.), 28/64/128/224 px (2D) and 28/64 (3D) | 46.1 GB in total. 28px files include: breastmnist 0.6 MB, pneumoniamnist 4.2 MB, retinamnist 3.3 MB, organcmnist 15.5 MB, organsmnist 16.5 MB, bloodmnist 35.5 MB, organamnist 38.2 MB, octmnist 54.9 MB, chestmnist 82.8 MB, tissuemnist 125 MB, pathmnist 205.6 MB; 3D: adrenalmnist3d 0.3 MB, vesselmnist3d 0.4 MB, fracturemnist3d 3.3 MB, nodulemnist3d 29.3 MB, organmnist3d 32.7 MB, synapsemnist3d 38.0 MB | CC BY 4.0 (Zenodo 10519652), **except DermaMNIST: CC BY-NC 4.0** (from HAM10000). Code Apache-2.0. | Open (`pip install medmnist`; `python -m medmnist download --size=28`; HF `albertvillanova/medmnist-v2`) | Yes (not DermaMNIST) | https://zenodo.org/records/10519652 ; https://medmnist.com | **CI (best choice)**, Eval, FT |
| VQA-RAD | Radiology visual QA (CT/MRI/CXR) | 315 images, 3,515 QA (unverified) | CC0 1.0 (OSF, verified; HF `flaviagiammarino/vqa-rad` also CC0) | Open | Yes | https://osf.io/89kps/ | CI, Eval |
| SLAKE | Bilingual (EN/ZH) radiology VQA with semantic labels | 642 images, 14,028 QA (450 / 96 / 96 image split) | HF `BoKelvin/SLAKE` tag says CC BY 4.0; one paper says CC BY-SA 4.0 (conflict). Either way it is permissive. | Open | Yes | https://huggingface.co/datasets/BoKelvin/SLAKE ; http://www.med-vqa.com/slake | CI, Eval |
| PubMedVision | Medical image-caption and VQA from PubMed Central | 1M to 10M rows | Apache-2.0 (HF tag). Images come from PMC articles with mixed per-article licenses. | Open (HF) | Per tag yes; check PMC image provenance | https://huggingface.co/datasets/FreedomIntelligence/PubMedVision | FT |
| GMAI-MMBench | Large multimodal medical benchmark (many modalities and tasks) | Approx. 26k QA (unverified) | Apache-2.0 (HF and GitHub `uni-medical/GMAI-MMBench`) | Gated HF (auto) | Yes per tag; built from source datasets with their own terms | https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench | Eval |
| MMMU (health subsets) | College-level multimodal QA including Clinical Medicine, Basic Medical Science, Diagnostics, Pharmacy, Public Health | Approx. 11.5k questions overall (unverified) | Apache-2.0 | Open (HF) | Yes | https://huggingface.co/datasets/MMMU/MMMU | Eval |
| **Restricted / research-only / unclear** | | | | | | | |
| PathVQA | Pathology VQA | 4,998 images, 32,799 QA | HF `flaviagiammarino/path-vqa` claims MIT, but the images come from copyrighted textbooks and PEIR. The original repo `UCSD-AI4H/PathVQA` has no license. | Open (HF) | Treat as No | https://huggingface.co/datasets/flaviagiammarino/path-vqa | Eval (internal) |
| PMC-VQA | VQA generated from PMC figures | 227k QA over 149k images | GitHub repo `xiaoman-zhang/PMC-VQA` is MIT (code). HF `RadGenome/PMC-VQA` has no license tag. Images have mixed per-article PMC licenses. | Open (HF) | Unclear | https://huggingface.co/datasets/RadGenome/PMC-VQA | Eval, FT (check) |
| ROCOv2 | Radiology image captioning and concept detection (PMC OA) | Approx. 80k images (unverified), 6.4 GB | CC BY-NC 4.0 (Zenodo 8333645). HF `eltorio/ROCOv2-radiology` says CC BY-NC-SA 4.0. The original ROCO repo has no license. | Open | No | https://zenodo.org/records/8333645 | Eval, FT (non-commercial) |
| MedPix 2.0 | Teaching case database (images plus case text) | Thousands of cases (unverified) | "MedPix License" plus NLM copyright, not Creative Commons. Personal and local teaching use only; many cases cannot be redistributed. | Open web | No | https://medpix.nlm.nih.gov/ | Eval (internal) |
| OmniMedVQA | Large medical VQA from 73 source datasets (unverified) | Approx. 118k images, 128k QA (unverified) | Only a subset is published. Users must follow each source dataset's license. | Open (HF, subset) | Mixed | https://huggingface.co/datasets/foreverbeliever/OmniMedVQA | Eval |
| MedTrinity-25M | Multigranular image-text triples | Approx. 24.8M rows; 7.4 GB captions | No license tag | Gated HF (auto) | Unclear | https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M | FT (check) |
| MultiMedBench (Med-PaLM M) | Composite of 14 existing tasks | Not released as one package (unverified) | Inherits each source's license (includes MIMIC, so credentialed) | N/A | No (in part) | https://arxiv.org/abs/2307.14334 | Reference only |

## 7. Symptom-to-diagnosis and differential diagnosis

| Dataset | Task | Size | License | Access | Commercial? | URL | Good for |
|---|---|---|---|---|---|---|---|
| DDXPlus | Synthetic patients with symptoms and antecedents, plus a ground-truth **ranked differential diagnosis** | Approx. 1.3M patients, 49 pathologies. Files: train zip 141 MB, validate 18.7 MB, test 19 MB, plus `release_evidences.json` and `release_conditions.json` | CC BY 4.0 (Figshare 20043374, verified via API; HF `aai530-group6/ddxplus` also CC BY 4.0) | Open (Figshare) | Yes | https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374 | **CI (test zip or JSON files), Eval, FT. The best fit for "multiple most likely options".** |
| Symptom2Disease | Free-text symptom description to disease | Approx. 1,200 rows, 24 diseases (unverified), 230 KB | CC0 (Kaggle) | Open | Yes | https://www.kaggle.com/datasets/niyarrbarman/symptom2disease | CI, small Eval |
| gretelai/symptom_to_diagnosis | Same task (cleaned version of Symptom2Disease, unverified) | 1K to 10K rows | Apache-2.0 (HF) | Open | Yes | https://huggingface.co/datasets/gretelai/symptom_to_diagnosis | CI, Eval |
| Disease Prediction Using ML (kaushil268) | Binary symptom vector to disease | 132 symptoms, 41 diseases, 1.4 MB | ODbL (Kaggle tag) | Open | Yes (share-alike) | https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning | CI (tabular baseline) |
| Disease Symptom Description (itachi9604) | Symptom list to disease, plus descriptions and precautions | Approx. 4.9k rows, 41 diseases (unverified) | CC BY-SA 4.0 (Kaggle tag) | Open | Yes (share-alike) | https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset | CI |
| Symptoms to diseases (abhishekgodara) | Symptom vector to disease (large) | 213 MB | ODbL (Kaggle tag) | Open | Yes (share-alike) | https://www.kaggle.com/datasets/abhishekgodara/symptoms-to-diseases | Eval, FT (synthetic-looking; verify quality) |
| MedQA (USMLE) | Multiple-choice board-exam QA (vignette to diagnosis or next step) | Approx. 12,723 EN questions (unverified) plus ZH | MIT (GitHub `jind11/MedQA`). HF `GBaker/MedQA-USMLE-4-options` says CC BY 4.0; `bigbio/med_qa` says "unknown". | Open | Yes per repo | https://github.com/jind11/MedQA | CI (slice), Eval |
| MedMCQA | Indian medical-entrance MCQ | 194k questions | MIT (GitHub); Apache-2.0 (HF `openlifescienceai/medmcqa`) | Open | Yes | https://huggingface.co/datasets/openlifescienceai/medmcqa | Eval, FT |
| PubMedQA | Yes/no/maybe QA over abstracts | 1k expert-labelled plus 211k artificial | MIT | Open (HF `qiaojin/PubMedQA`) | Yes | https://github.com/pubmedqa/pubmedqa | CI, Eval |
| MMLU medical subsets | anatomy, clinical_knowledge, college_medicine, college_biology, medical_genetics, professional_medicine | Approx. 1.9k questions (unverified) | MIT (`cais/mmlu`) | Open | Yes | https://huggingface.co/datasets/cais/mmlu | CI, Eval |
| HealthBench | Rubric-graded health conversations | 5,000 conversations | MIT (HF `openai/healthbench`; `openai/simple-evals`) | Open | Yes | https://github.com/openai/simple-evals | Eval |
| MedXpertQA | Hard expert-level medical QA (text plus multimodal) | Approx. 4,460 items | MIT (HF and GitHub) | Open | Yes | https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA | Eval |
| medical-o1-reasoning-SFT | Chain-of-thought medical reasoning SFT data (GPT-4o-generated) | Approx. 90k (unverified) | Apache-2.0 (HF) | Open | Yes per tag; generated with a commercial LLM, so check that provider's terms | https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT | FT |
| **Restricted / research-only / unclear** | | | | | | | |
| AgentClinic | Interactive diagnostic agent benchmark | Several hundred cases (unverified) | Code MIT. NEJM-derived cases are NEJM copyright (unverified). MedQA-derived cases follow MedQA. | Open (GitHub) | Code yes; NEJM part no | https://github.com/SamuelSchmidgall/AgentClinic | Eval (use the MedQA-based part) |
| NEJM Image Challenge | Image-plus-vignette diagnosis quiz | Approx. 1k cases (unverified) | Copyrighted (NEJM) | Open web | No | https://www.nejm.org/image-challenge | Eval (internal, cite only) |
| MIMIC-IV-ED, MIMIC-IV-Note | Real triage and discharge diagnoses | See group 1 | PhysioNet Credentialed | Credentialed + DUA | Research only | See group 1 | Eval (internal) |

Notes:

- DDXPlus was generated from a proprietary knowledge base and a commercial
  rule-based system. The data are CC BY 4.0, but the generator is not
  available. It was released in French, with an English version since May 2023
  (NeurIPS 2022 Datasets and Benchmarks, arXiv 2205.09148).
- The small Kaggle symptom-disease tables are synthetic or scraped and very
  easy (near 100% accuracy). They are useful only as CI fixtures.

## 8. Synthetic data tools

| Tool | Task | Size | License | Access | Commercial? | URL | Good for |
|---|---|---|---|---|---|---|---|
| Synthea | Synthetic patient records (FHIR, C-CDA, CSV) | Unlimited | Apache-2.0 (also `synthea-international`) | Open | Yes | https://github.com/synthetichealth/synthea | CI, FT (input for rendering synthetic histories) |
| Faker (Python) | Fake names, addresses and dates for documents | N/A | MIT | Open (PyPI) | Yes | https://github.com/joke2k/faker | CI |
| TextRecognitionDataGenerator (TRDG) | Synthetic printed and handwritten-style text-line images | Unlimited | MIT | Open (PyPI `trdg`) | Yes | https://github.com/Belval/TextRecognitionDataGenerator | CI, FT (OCR) |
| text-to-handwriting | Render text in handwriting fonts | N/A | MIT | Open | Yes | https://github.com/saurabhdaware/text-to-handwriting | CI (fake prescriptions) |
| Handwriting-Transformers (HWT) | Learned handwriting style synthesis | N/A | MIT (code; pretrained on IAM, so the weights inherit IAM non-commercial concerns, unverified) | Open | Code yes; weights check | https://github.com/ankanbhunia/Handwriting-Transformers | FT augmentation |
| DiffusionPen | Diffusion-based handwriting synthesis | N/A | MIT (code; IAM-trained weights, same caveat) | Open | Code yes; weights check | https://github.com/koninik/DiffusionPen | FT augmentation |
| phantominator | Shepp-Logan and other MRI/CT phantoms | N/A | MIT | Open (PyPI) | Yes | https://github.com/mckib2/phantominator | CI |
| DeepDRR | Synthetic X-ray (DRR) from CT volumes | N/A | GPL-3.0 (copyleft; affects distribution of linked code) | Open | Yes, under GPL terms | https://github.com/arcadelab/deepdrr | FT (CXR from TotalSegmentator/LIDC CT) |
| MONAI, GenerativeModels, model-zoo | Medical imaging deep-learning framework, diffusion and VAE models | N/A | Apache-2.0 | Open | Yes | https://github.com/Project-MONAI/MONAI | CI, FT |
| NV-Generate-CTMR (MAISI) | 3D CT/MR synthesis | N/A | Code Apache-2.0. `nvidia/NV-Generate-CT` weights: NVIDIA Open Model License. **`nvidia/NV-Generate-MR` weights: NVIDIA OneWay Non-Commercial License (academic research).** | Open / HF | CT weights yes (per NVIDIA OML); MR weights no | https://huggingface.co/nvidia/NV-Generate-CT | FT augmentation |
| RoentGen-v2 | Text-to-CXR diffusion model, plus a 565k synthetic CXR dataset | 565k images | Repo MIT. HF model `stanfordmimi/RoentGen-v2` and dataset `stanfordmimi/RoentGen-v2-synthetic-dataset` are gated and tagged MIT. Trained on MIMIC-CXR-derived data, so check PhysioNet terms. | Gated HF (auto) | Per tag yes; legal review advised | https://huggingface.co/stanfordmimi/RoentGen-v2 | FT (augmentation), Eval |
| idc-index (NCI Imaging Data Commons) | Query and download TCIA/IDC series with per-series licenses | N/A | MIT | Open (PyPI) | Yes | https://github.com/ImagingDataCommons/idc-index | CI (fetch 1 to 2 small public series) |
| PTB-XL (bonus: ECG) | 12-lead ECG classification | 21,799 records (unverified), 3.0 GB | CC BY 4.0 | Open (PhysioNet) | Yes | https://physionet.org/content/ptb-xl/ | Eval, FT |
| MIMIC-IV-ECG (bonus) | Diagnostic 12-lead ECGs | Approx. 800k ECGs (unverified), 90.4 GB | ODbL (Open on PhysioNet) | Open | Yes (share-alike) | https://physionet.org/content/mimic-iv-ecg/ | Eval, FT |
| **Restricted / unclear** | | | | | | | |
| sjvasquez/handwriting-synthesis | RNN handwriting synthesis (Graves) | N/A | **No license** (all rights reserved) | Open (GitHub) | No | https://github.com/sjvasquez/handwriting-synthesis | Avoid; use HWT or DiffusionPen |
| RoentGen v1 | Text-to-CXR (Stable Diffusion fine-tune) | N/A | Weights via Stanford user agreement; repo has no license | Registration | No | https://github.com/StanfordMIMI/RoentGen | Eval (internal) |
| microsoft/radedit | CXR editing diffusion model | N/A | "other", research only | Gated HF | No | https://huggingface.co/microsoft/radedit | Eval (internal) |
| XCAT phantoms (Duke) | Anatomical computational phantoms | N/A | Duke licence agreement (unverified) | Licence request | Unverified, likely paid for commercial use | https://cvit.duke.edu/resource/xcat-phantom-program/ (unverified) | Reference |

---

## Permissive vs restricted at a glance

**Truly permissive, safe for public CI and commercial fine-tuning** (with attribution where CC BY):

- CC0: VQA-RAD; Symptom2Disease (Kaggle tag).
- Public domain style ("no restrictions" plus attribution): NIH ChestX-ray14.
- CC BY 4.0: MedMNIST (not DermaMNIST), HC18, TotalSegmentator, AMOS, LUNA16,
  Kermany CXR/OCT, RxHandBD, CORD, DDXPlus, Synthea Coherent, PTB-XL, SLAKE (tag).
- CC BY 3.0: LIDC-IDRI.
- CC BY-SA (share-alike): MSD, MedNIST, IXI, COVID-QU-Ex.
- ODbL (share-alike database): MIMIC-IV/III demos, MIMIC-IV-ECG, kaushil268 symptoms.
- MIT / Apache-2.0: Synthea code, NHS synthetic notes, MedQA, MedMCQA, PubMedQA,
  MMLU, HealthBench, MedXpertQA, MMMU, GMAI-MMBench (gated), gretelai
  symptom_to_diagnosis, all MIT/Apache tools in group 8.

**Restricted (non-commercial, credentialed or no license); internal research only:**

- Credentialed (PhysioNet DUA): MIMIC-III, MIMIC-IV-Note, MIMIC-IV-ED,
  MIMIC-CXR(-JPG), Chest ImaGenome, MedNLI, RadGraph, VinDr-CXR.
- Restricted (PhysioNet): VinDr-PCXR, VinDr-SpineXR, VinDr-Mammo.
- Research use agreements: CheXpert, CheXpert Plus, MURA, EchoNet-Dynamic
  (Stanford); fastMRI (NYU); PadChest (BIMCV); n2c2 (Harvard); OASIS (DUA).
- NC Creative Commons: DermaMNIST, ROCOv2, KiTS23, CAMUS (unverified),
  QaTa-COV19, Asclepius, PMC-Patients, IU X-ray (NC-ND).
- Non-commercial custom terms: FUNSD, IAM, MedPix.
- No license or copyright retained: MTSamples, MedDialog, PathVQA images,
  BUSI (unverified), POCUS COVID, COVID-19 Radiography DB, sjvasquez
  handwriting-synthesis.

---

## Recommended starter set

Selection rule: permissive license verified at the source, downloadable
without credentials, and a small slice that fits in CI.

| Need from issue #1 | CI smoke test (small, permissive) | Real evaluation | Fine-tuning | License |
|---|---|---|---|---|
| X-ray reading | MedMNIST `pneumoniamnist` 28px (4.2 MB), `chestmnist` (82.8 MB) | NIH ChestX-ray14 (official test split); Kermany pediatric CXR | NIH CXR14, RSNA Pneumonia, COVID-QU-Ex | CC BY 4.0; NIH "no restrictions" plus attribution; CC BY-SA 4.0 |
| MRI | MSD Task04 Hippocampus (small); a few IXI volumes; phantominator phantoms | MSD BrainTumour / Prostate; IXI | IXI (self-supervised), MSD | CC BY-SA 4.0; CC BY-SA 3.0; MIT |
| CT | MedMNIST `organamnist` / `nodulemnist3d`; MSD Task09 Spleen | TotalSegmentator v2, AMOS, LIDC-IDRI / LUNA16 | TotalSegmentator, AMOS | CC BY 4.0; CC BY 3.0 |
| Ultrasound | MedMNIST `breastmnist` (0.6 MB); HC18 (180 MB) | HC18 | HC18 plus simulated phantoms | CC BY 4.0 |
| OCR of medical histories | Synthea record rendered to PDF/PNG with TRDG noise; CORD sample | CORD test split; synthetic held-out set | Synthea Coherent plus rendered documents | Apache-2.0; CC BY 4.0; MIT |
| Handwritten prescriptions | 50 RxHandBD crops; text-to-handwriting renders | RxHandBD test split (1,115) | RxHandBD train plus synthetic (TRDG, HWT code) | CC BY 4.0; MIT |
| Initial diagnosis with multiple options | DDXPlus test zip (19 MB) slice; Symptom2Disease or gretelai | DDXPlus test set (top-k accuracy against the ranked differential); MedQA; MMLU medical; HealthBench | DDXPlus train; MedMCQA | CC BY 4.0; CC0 / Apache-2.0; MIT |
| Multimodal VQA | VQA-RAD (CC0), SLAKE | VQA-RAD, SLAKE, GMAI-MMBench, MMMU health | PubMedVision (check PMC provenance) | CC0; CC BY 4.0; Apache-2.0 |
| Clinical text | NHS synthetic notes; MIMIC-IV demo (schema only) | MedQA, PubMedQA | Synthea plus NHS synthetic notes | MIT; ODbL; Apache-2.0 |

Practical rules for the repo:

1. CI must only fetch: MedMNIST 28px files, HC18, MSD Task04, the DDXPlus test
   zip and JSON files, VQA-RAD, small HF slices of MedQA/PubMedQA/MMLU, a few
   RxHandBD crops, and generated data (Synthea, phantoms, TRDG). Every one of
   these is CC0, CC BY, CC BY-SA, MIT or Apache.
2. Keep a `LICENSES` manifest next to each loader, recording the source URL,
   SPDX id, and the date the license was checked.
3. Never pull from Kaggle or HF mirrors whose tag disagrees with the original
   (see Surprises). Pull from the original host (Zenodo, Figshare, Mendeley,
   OSF, PhysioNet, TCIA).
4. Restricted sets (MIMIC, CheXpert, fastMRI, etc.) belong in an opt-in
   "internal evaluation" path that needs local credentials. Never use them in
   public CI, and never send them to third-party LLM APIs.

---

## Surprises: datasets commonly assumed open that are not

- **Laundered re-uploads.** Kaggle and HF tags that contradict the original license:
  - CheXpert-v1.0-small on Kaggle is tagged CC0; the original is a Stanford
    non-commercial RUA.
  - BraTS2020 on Kaggle is tagged CC0; the original terms are mixed.
  - MTSamples on Kaggle is tagged CC0 (Apache-2.0 on HF); the site only allows
    educational sharing.
  - IAM-line on HF is tagged MIT; IAM is non-commercial research only.
  - PathVQA on HF is tagged MIT; the images come from copyrighted textbooks.
  - augmented-clinical-notes is tagged MIT; it derives from CC BY-NC-SA data.
  - OASIS on Kaggle is tagged Apache-2.0 or CC0; the original is a DUA.
  - BUSI on Kaggle is tagged CC0; no license is stated at the source.
- **FUNSD and IAM**, the default form-understanding and handwriting
  benchmarks, are both non-commercial only. CORD (CC BY 4.0) is the
  permissive alternative for forms.
- **DermaMNIST is CC BY-NC 4.0** inside an otherwise CC BY 4.0 MedMNIST.
- **ROCOv2 is CC BY-NC 4.0** and **IU X-ray / Open-i is CC BY-NC-ND 4.0**,
  even though both are "from open-access literature or NLM".
- **CheXpert, CheXpert Plus, MURA and EchoNet-Dynamic** (Stanford) are all
  non-commercial research agreements.
- **fastMRI** is internal research only: no redistribution, and data must be
  destroyed after use.
- **VinDr-CXR** is PhysioNet credentialed, and VinDr-SpineXR, -PCXR and -Mammo
  are "Restricted". Many people assume VinDr is open because of the Kaggle
  competition.
- **KiTS23 is CC BY-NC-SA**, while the older C4KC-KiTS on TCIA is listed as
  CC BY 3.0.
- **RSNA Pneumonia**: the RSNA page allows commercial use with attribution, but
  the original Kaggle competition rules were non-commercial.
- **Doctor's Handwritten Prescription BD** is labelled CC BY 4.0 on Mendeley,
  yet the same page says "educational and research purposes only".
- **MedDialog**: copyright belongs to the source websites (HealthCareMagic,
  iCliniq, haodf), despite wide use in fine-tuning.
- **MedPix** (NLM) is not Creative Commons: personal and local teaching use only.
- **sjvasquez/handwriting-synthesis** (the popular Graves RNN demo) has no
  license, so all rights are reserved.
- **NV-Generate-MR weights** are non-commercial, while the CT weights and the
  code are more permissive.
- **PhysioNet credentialed data** (all of MIMIC) must not be sent to
  third-party online LLM APIs under PhysioNet's responsible-use guidance
  (unverified this session). This rules out "just ask GPT" on MIMIC notes.
- **MIMIC demos are open (ODbL) but have no clinical notes** (unverified), so
  they cannot stand in for MIMIC-IV-Note.

---

## Sources

License and metadata were checked on 2026-10-14 unless marked otherwise.

- Synthea: https://github.com/synthetichealth/synthea (LICENSE, README); https://registry.opendata.aws/synthea-coherent-data ; Walonoski et al., Electronics 2022, doi:10.3390/electronics11081199
- NHS synthetic notes: https://huggingface.co/api/datasets/NHSEDataScience/synthetic_clinical_notes ; https://github.com/nhsengland/synthetic_clinical_notes
- PhysioNet project pages (license and access tier): mimic-iv-demo/2.2, mimiciii-demo/1.4, mimic-iv-note/2.2, mimiciii/1.4, mimic-iv-ed, mednli, radgraph, mimic-cxr/2.1.0, mimic-cxr-jpg/2.1.0, chest-imagenome, vindr-cxr/1.0.0, vindr-pcxr, vindr-spinexr, vindr-mammo, ptb-xl, mimic-iv-ecg (https://physionet.org/content/...)
- PhysioNet Credentialed Health Data License 1.5.0: https://physionet.org/content/mimiciii/view-license/1.4/
- n2c2: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- HF dataset API (`https://huggingface.co/api/datasets/<id>`) for: starmpcc/Asclepius-Synthetic-Clinical-Notes, zhengyun21/PMC-Patients, AGBonnet/augmented-clinical-notes, bigbio/meddialog, UCSD26/medical_dialog, harishnair04/mtsamples, nielsr/funsd, Teklia/IAM-line, Teklia/RIMES-2011-line, lance-format/handwriting-ocr, avi-kai/Medical_Prescription_Handwritten_Words, MMMuzammil/Medical_Prescription_Handwritten_Words, chaithanyakota/100-handwritten-medical-records, alkzar90/NIH-Chest-X-ray-dataset, ykumards/open-i, hf-vision/chest-xray-pneumonia, albertvillanova/medmnist-v2, flaviagiammarino/vqa-rad, BoKelvin/SLAKE, flaviagiammarino/path-vqa, RadGenome/PMC-VQA, eltorio/ROCOv2-radiology, foreverbeliever/OmniMedVQA, UCSC-VLAA/MedTrinity-25M, FreedomIntelligence/PubMedVision, OpenGVLab/GMAI-MMBench, MMMU/MMMU, aai530-group6/ddxplus, gretelai/symptom_to_diagnosis, GBaker/MedQA-USMLE-4-options, bigbio/med_qa, openlifescienceai/medmcqa, qiaojin/PubMedQA, cais/mmlu, openai/healthbench, TsinghuaC3I/MedXpertQA, FreedomIntelligence/medical-o1-reasoning-SFT, stanfordmimi/RoentGen-v2, stanfordmimi/RoentGen-v2-synthetic-dataset, nvidia/NV-Generate-CT, nvidia/NV-Generate-MR, microsoft/radedit
- Kaggle API (`https://www.kaggle.com/api/v1/datasets/view/<owner/slug>`, licenseName and totalBytes) for: tboyle10/medicaltranscriptions, urbikn/sroie-datasetv2, mamun1113/doctors-handwritten-prescription-bd-dataset, landlord/handwriting-recognition, nih-chest-xrays/data, ashery/chexpert, raddar/tuberculosis-chest-xrays-montgomery, raddar/tuberculosis-chest-xrays-shenzhen, raddar/nodules-in-chest-xrays-jsrt, raddar/chest-xrays-indiana-university, nikhilpandey360/chest-xray-masks-and-labels, anasmohammedtahir/covidqu, aysendegerli/qatacov19-dataset, tawsifurrahman/covid19-radiography-database, tawsifurrahman/tuberculosis-tb-chest-xray-dataset, paultimothymooney/chest-xray-pneumonia, andrewmvd/pediatric-pneumonia-chest-xray, awsaf49/brats20-dataset-training-validation, dschettler8845/brats-2021-task1, ninadaithal/imagesoasis, jboysen/mri-and-alzheimers, masoudnickparvar/brain-tumor-mri-dataset, sartajbhuvaji/brain-tumor-classification-mri, navoneel/brain-mri-images-for-brain-tumor-detection, aryashah2k/breast-ultrasound-images-dataset, sabahesaraki/breast-ultrasound-images-dataset, dasmehdixtr/ddti-thyroid-ultrasound-images, niyarrbarman/symptom2disease, kaushil268/disease-prediction-using-machine-learning, itachi9604/disease-symptom-description-dataset, abhishekgodara/symptoms-to-diseases
- Zenodo API (`https://zenodo.org/api/records/<id>`): 10519652 (MedMNIST+), 10047292 and 6802614 (TotalSegmentator), 7262581 and 7155725 (AMOS), 3723295 (LUNA16), 1327317 and 1322001 (HC18), 8333645 (ROCOv2), 10812725 (RIMES)
- Figshare API: https://api.figshare.com/v2/articles/20043374 (DDXPlus); Tchango et al., arXiv:2205.09148
- OSF: https://osf.io/89kps/ (VQA-RAD)
- Mendeley Data: https://data.mendeley.com/datasets/rscbjbr9sj/2 (Kermany), https://data.mendeley.com/datasets/zjjtptvn6f/2 (BD prescriptions), https://data.mendeley.com/datasets/dsb5r6vskg (RxHandBD)
- MTSamples terms: https://mtsamples.com/
- FUNSD terms: https://guillaumejaume.github.io/FUNSD/
- CORD: https://github.com/clovaai/cord
- IAM terms: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- NIH CXR14 README: https://nihcc.app.box.com/v/ChestXray-NIHCC
- RSNA Pneumonia terms: https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
- Stanford AIMI datasets (CheXpert, CheXpert Plus, MURA, EchoNet): https://aimi.stanford.edu/shared-datasets ; https://doi.org/10.71718/6nvz-pm34
- PadChest / BIMCV: https://bimcv.cipf.es/bimcv-projects/padchest/
- JSRT: http://db.jsrt.or.jp/eng.php ; mini JSRT: http://imgcom.jsrt.or.jp/minijsrtdb/
- NLM TB sets: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html
- MONAI MedNIST: https://github.com/Project-MONAI/tutorials
- MSD: http://medicaldecathlon.com/ ; Antonelli et al., Nat. Commun. 2022
- TCIA: https://www.cancerimagingarchive.net/collection/lidc-idri/ ; https://www.cancerimagingarchive.net/collection/c4kc-kits/
- BraTS: https://www.synapse.org/brats (syn25829067); Baid et al., arXiv:2107.02314
- fastMRI: https://fastmri.med.nyu.edu/
- IXI: https://brain-development.org/ixi-dataset/
- OASIS: https://sites.wustl.edu/oasisbrains/
- KiTS23: https://github.com/neheller/kits23 (README and LICENSE)
- BUSI: Al-Dhabyani et al., Data in Brief 2020, doi:10.1016/j.dib.2019.104863
- CAMUS: https://www.creatis.insa-lyon.fr/Challenge/camus/
- EchoNet-Dynamic: https://echonet.github.io/dynamic/ ; https://github.com/echonet/dynamic
- POCUS: https://github.com/jannisborn/covid19_ultrasound ; COVID-BLUES: https://github.com/NinaWie/COVID-BLUES
- TN3K / DDTI: Gong et al. 2023; Pedraza et al. 2015 (licenses unverified)
- MedPix: https://medpix.nlm.nih.gov/
- GitHub repo licenses (via `gh api repos/<owner>/<repo>`): jind11/MedQA, medmcqa/medmcqa, pubmedqa/pubmedqa, openai/simple-evals, SamuelSchmidgall/AgentClinic, xiaoman-zhang/PMC-VQA, UCSD-AI4H/PathVQA, razorx89/roco-dataset, uni-medical/GMAI-MMBench, zzzDavid/ICDAR-2019-SROIE, synthetichealth/synthea, synthetichealth/synthea-international, joke2k/faker, faker-js/faker, Belval/TextRecognitionDataGenerator, saurabhdaware/text-to-handwriting, ankanbhunia/Handwriting-Transformers, koninik/DiffusionPen, sjvasquez/handwriting-synthesis, mckib2/phantominator, arcadelab/deepdrr, Project-MONAI/MONAI, Project-MONAI/GenerativeModels, Project-MONAI/model-zoo, NVIDIA-Medtech/NV-Generate-CTMR, StanfordMIMI/RoentGen, StanfordMIMI/RoentGen-v2, ImagingDataCommons/idc-index, echonet/dynamic, nhsengland/synthetic_clinical_notes
