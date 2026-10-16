"""Write src/medbench/data/label_review.json (issue #11, part of issue #7).

The review is by the repository maintainers, not by clinicians. Each entry
records what the image or page shows according to its source, the view,
the laterality convention, the label source, open doubts and a source group
used to keep possibly related items in one split. Re-run after editing:

    python experiments/issue11_build_label_review.py
"""

from __future__ import annotations

import json
from pathlib import Path

COMMONS = "Wikimedia Commons uploader description, quoted in real_samples.json"
RX = "RxHandBD Prescription_Labels.csv (the dataset authors' own label)"
WS = "English Wikisource transcription (proofread status Validated), checked by eye"
PATIENT_SIDE = "patient side; the image shows it mirrored (radiological convention)"


def entry(sample_id, view, laterality, label_source, group, uncertainty, also=()):
    return {
        "sample_id": sample_id,
        "view": view,
        "laterality": laterality,
        "label_source": label_source,
        "source_group": group,
        "uncertainty": uncertainty,
        "also_present": list(also),
    }


ENTRIES = [
    entry(
        "xray-pneumothorax-1",
        "frontal, projection not stated",
        f"left, {PATIENT_SIDE}",
        COMMONS,
        "hellerhoff",
        "German description; side read from 'links (rechts im Bild)'. Projection (PA/AP) not stated.",
    ),
    entry(
        "xray-normal-1",
        "PA (stated in the file title)",
        "not applicable (normal)",
        COMMONS,
        "haggstrom",
        "'Normal' means 'without any signs of injury' in the description; not a full radiology report.",
    ),
    entry(
        "xray-pneumothorax-2",
        "frontal, projection not stated",
        f"right, {PATIENT_SIDE}",
        COMMONS,
        "hellerhoff",
        "Description also mentions a small basal fluid level and an earlier pneumothorax on the other side.",
        ["small basal fluid level"],
    ),
    entry(
        "xray-normal-2",
        "frontal, projection not stated; photograph of a film",
        "not applicable (normal)",
        COMMONS,
        "doctoroftcm",
        "Photograph of film with glare; 'healthy man' is the uploader's statement, not a report.",
    ),
    entry(
        "xray-pleural-effusion-1",
        "frontal, projection not stated",
        f"right, {PATIENT_SIDE}",
        COMMONS,
        "heilman",
        "Side from the description 'Right sided effusion'; size not stated.",
    ),
    entry(
        "xray-normal-3",
        "AP (not PA); photograph of a monitor",
        "not applicable (normal)",
        COMMONS,
        "heilman",
        "AP view: heart size and mediastinum look larger than on PA, so 'cardiomegaly' answers may be view artefacts.",
    ),
    entry(
        "xray-pleural-effusion-2",
        "AP (not PA; 'CR ap' in the file name)",
        f"left, {PATIENT_SIDE}",
        COMMONS,
        "hellerhoff",
        "Large effusion with mediastinal shift to the right; the shift is a second correct finding.",
        ["mediastinal shift"],
    ),
    entry(
        "xray-lobar-pneumonia-1",
        "frontal, projection not stated",
        f"right (middle lobe), {PATIENT_SIDE}",
        COMMONS,
        "haggstrom",
        "Only a 'Sin' marker; lobe from the description, not from a lateral view.",
    ),
    entry(
        "xray-lung-mass-1",
        "PA ('CR pa' in the file name)",
        f"right (upper zone), {PATIENT_SIDE}",
        COMMONS,
        "hellerhoff",
        "Histology (adenocarcinoma) is from the description and cannot be seen on an X-ray; only 'mass' is scorable.",
    ),
    entry(
        "mri-vestibular-schwannoma-1",
        "axial T1 after contrast",
        f"right, {PATIENT_SIDE}",
        COMMONS,
        "hellerhoff",
        "Side from 'right side of the patient (left in the image)'.",
    ),
    entry(
        "mri-normal-1",
        "axial T2, lateral ventricles",
        "not applicable (normal)",
        COMMONS,
        "novak",
        "Single slice; 'normal' covers this slice only.",
    ),
    entry(
        "mri-hemorrhagic-metastasis-1",
        "axial T2",
        f"left frontal, {PATIENT_SIDE}",
        COMMONS,
        "dawes",
        "A known melanoma metastasis with a large haematoma; 'tumour' and 'haemorrhage' are both correct answers.",
        ["tumour", "haematoma with fluid-fluid level"],
    ),
    entry(
        "mri-normal-2",
        "axial T2, lateral ventricles",
        "not applicable (normal)",
        COMMONS,
        "novak",
        "Same uploader as mri-normal-1, described as a different subject; kept in one group because that cannot be checked.",
    ),
    entry(
        "mri-cerebral-infarct-1",
        "axial DWI",
        f"right superior frontal, {PATIENT_SIDE}",
        "Wikimedia Commons description of a slice isolated by Mikael Häggström from a Hellerhoff original",
        "hellerhoff",
        "Derived from a Hellerhoff series; grouped with the other Hellerhoff images.",
    ),
    entry(
        "mri-normal-3",
        "axial T2, orbits and midbrain",
        "not applicable (normal)",
        COMMONS,
        "ptrump16",
        "Single slice; 'normal' covers this slice only.",
    ),
    entry(
        "mri-hemorrhagic-contusion-1",
        "axial T2* (haem-sensitive)",
        f"right frontal, {PATIENT_SIDE}",
        COMMONS,
        "hellerhoff",
        "Contre-coup contusion; 'intracerebral haemorrhage' is an equally correct answer.",
        ["intracerebral haemorrhage"],
    ),
    entry(
        "us-gallstones-1",
        "gallbladder, plane not stated",
        "not applicable (midline organ)",
        COMMONS,
        "hellerhoff",
        "One large stone with a shadow; 'calcified' is the uploader's guess ('mutmaßlich').",
    ),
    entry(
        "us-normal-1",
        "right kidney, plane not stated",
        "organ is the right kidney; no lesion side",
        COMMONS,
        "ptrump16",
        "Length caliper 'L 9.34 cm' is burned in.",
    ),
    entry(
        "us-gallstones-2",
        "gallbladder, plane not stated",
        "not applicable (midline organ)",
        COMMONS,
        "haggstrom",
        "Also shows biliary sludge.",
        ["biliary sludge"],
    ),
    entry(
        "us-normal-2",
        "gallbladder and common bile duct",
        "not applicable (midline organ)",
        COMMONS,
        "ptrump16",
        "Scanner text only; 'normal' is the uploader's statement.",
    ),
    entry(
        "us-hydronephrosis-1",
        "kidney, plane not stated",
        "left (from the description; not visible in the image)",
        COMMONS,
        "morning2k",
        "Cause is a left ureteral stone that is probably not in view; the side cannot be checked from pixels.",
        ["ureteral stone (cause, probably not in view)"],
    ),
    entry(
        "us-normal-3",
        "liver, sagittal, midclavicular line",
        "not applicable (normal)",
        COMMONS,
        "haggstrom",
        "Single image; 'normal liver' is the uploader's statement.",
    ),
    entry(
        "us-hydronephrosis-2",
        "kidney, plane not stated",
        "not stated",
        COMMONS,
        "hansen-nielsen-ewertsen",
        "Pediatric patient; side not stated, so laterality is not scored.",
    ),
    entry(
        "us-kidney-stone-1",
        "kidney, plane not stated",
        "not stated",
        COMMONS,
        "hansen-nielsen-ewertsen",
        "Also shows hydronephrosis, which the task accepts as a term.",
        ["hydronephrosis"],
    ),
    entry(
        "us-kidney-stone-2",
        "kidney, plane not stated",
        "not stated",
        COMMONS,
        "hansen-nielsen-ewertsen",
        "Staghorn calculus; side not stated.",
    ),
    entry(
        "us-ovarian-cyst-1",
        "sagittal left ovary ('SAG LT OV' burned in)",
        "left (burned-in label names the organ side)",
        COMMONS,
        "heilman",
        "Burned-in 'LT OV' gives away the organ and side but not the finding.",
    ),
    entry(
        "rx-word-p0007",
        "one handwritten word crop",
        "not applicable",
        RX,
        "rxhandbd",
        "Brand name; spelling variants are not in the label.",
    ),
    entry(
        "rx-word-p1126",
        "one handwritten word crop",
        "not applicable",
        RX,
        "rxhandbd",
        "Brand plus strength ('Dexter 60'); the strength has no unit in the label.",
    ),
    entry(
        "rx-word-p3096",
        "one handwritten word crop",
        "not applicable",
        RX,
        "rxhandbd",
        "Generic name.",
    ),
    entry(
        "rx-word-p3113",
        "one handwritten word crop",
        "not applicable",
        RX,
        "rxhandbd",
        "Generic name.",
    ),
    entry(
        "rx-word-p3284",
        "one handwritten word crop",
        "not applicable",
        RX,
        "rxhandbd",
        "Generic name.",
    ),
    entry(
        "rx-word-p4623",
        "one handwritten word crop",
        "not applicable",
        RX,
        "rxhandbd",
        "'5% dns' (dextrose in normal saline); the abbreviation is the label, not an expansion.",
    ),
    entry(
        "history-blood-examination-p41",
        "printed page scan",
        "not applicable",
        WS,
        "fothergill-1907",
        "One missing degree sign corrected against the scan.",
    ),
    entry(
        "history-blood-examination-p37",
        "printed page scan",
        "not applicable",
        WS,
        "fothergill-1907",
        "Contains the apothecary sign '℥'; starts with a word fragment from the previous page.",
    ),
    entry(
        "history-fremantle-plague-p13",
        "printed page scan, degraded 1903 print",
        "not applicable",
        WS,
        "fremantle-1903",
        "Footer kept in the reference text; page number dropped.",
    ),
    entry(
        "history-harlow-iron-bar-p8",
        "grayscale page scan",
        "not applicable",
        WS,
        "harlow-1868",
        "Two Wikisource errors corrected against the scan; line-break hyphens joined.",
    ),
    entry(
        "history-alcohol-medicine-p60",
        "printed page scan with a watermark",
        "not applicable",
        WS,
        "allen-temperance",
        "Quoted hospital report inside a temperance book, not an original record; watermark not in the reference.",
    ),
]

DOCUMENT = {
    "reviewer": "medbench maintainers (not clinicians)",
    "clinician_reviewed": False,
    "method": (
        "Each label was read from the source page, compared with the image or "
        "scan by eye, and the doubts were written down. Nobody with medical "
        "training checked the images; the labels are the uploaders' or dataset "
        "authors' statements."
    ),
    "reviewed_on": "2026-10-16",
    "entries": ENTRIES,
}

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "src/medbench/data/label_review.json"
    out.write_text(
        json.dumps(DOCUMENT, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(ENTRIES)} entries to {out}")
