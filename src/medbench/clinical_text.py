"""Small, readable rules for reading clinical free text in model answers.

The scores in :mod:`medbench.benchmark` used plain keyword overlap, which
gives credit for "no pneumothorax" on a pneumothorax image and cannot tell
"5 mg" from "5 mcg". These helpers make the common cases explicit:

* :func:`affirmed_mention` - a term is mentioned and not negated, using a
  NegEx-style window (Chapman et al., 2001, J Biomed Inform 34:301-310);
* :func:`is_abstention` - the answer declines or says it is unsure;
* :func:`extract_doses` / :func:`compare_doses` - number + unit pairs with
  canonical units, so a wrong unit and a wrong value are counted apart;
* :func:`normalise_ocr` - Unicode NFKC, case folding and number/unit spacing
  before an OCR error rate is computed.

They are rules, not a clinical language model. They were checked against the
hand-written examples in ``tests/test_medbench_clinical_text.py``, which are
**not** clinician-reviewed; see docs/evaluation-validity.md for the limits.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

__all__ = [
    "Dose",
    "DoseComparison",
    "affirmed_mention",
    "compare_doses",
    "extract_doses",
    "is_abstention",
    "is_negated_at",
    "normalise_ocr",
]

# Words before a term that negate it, within NEGATION_WINDOW words.
_PRE_NEGATION = re.compile(
    r"\b(?:no|not|without|never|neither|nor|denies|denied|absent|absence of|"
    r"negative for|free of|no evidence of|no signs? of|rules? out|ruled out|"
    r"exclude[ds]?|excluding)\b"
)
# Words right after a term that negate it ("pneumothorax is not seen").
_POST_NEGATION = re.compile(
    r"^\W*(?:is |are |was |were |has been |have been )?"
    r"(?:not (?:seen|present|identified|visible|detected|evident|likely)|"
    r"ruled out|excluded|absent|unlikely|negative)\b"
)
# Phrases that contain a cue but keep the term possible.
_PSEUDO_NEGATION = re.compile(
    r"\b(?:not only|no change in|no increase in|cannot be (?:ruled out|excluded)|"
    r"can ?not rule out|not (?:been )?(?:ruled out|excluded))\b"
)
# A negation does not cross these (NegEx "termination terms").
_CLAUSE_BREAK = re.compile(
    r"[.;:!?\n]|\b(?:but|however|although|though|except|apart from|aside from|"
    r"whereas|yet)\b"
)
NEGATION_WINDOW = 5

_ABSTENTION = re.compile(
    r"\b(?:unknown|uncertain|unsure|indeterminate|not sure|i don'?t know|"
    r"i do not know|cannot (?:determine|tell|read|say|assess|be determined)|"
    r"can'?t (?:determine|tell|read|say)|unable to|not possible to|"
    r"insufficient (?:information|detail)|i (?:cannot|can'?t) (?:help|provide))\b"
)


def is_abstention(text: str) -> bool:
    """Whether an answer declines or says it does not know.

    An empty answer is not an abstention; it is reported as invalid.
    """
    return _ABSTENTION.search(text.lower()) is not None


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Start and end of the clause holding ``text[start:end]``."""
    left = 0
    for match in _CLAUSE_BREAK.finditer(text, 0, start):
        left = match.end()
    following = _CLAUSE_BREAK.search(text, end)
    return left, following.start() if following else len(text)


def is_negated_at(text: str, start: int, end: int) -> bool:
    """Whether the mention at ``text[start:end]`` is negated.

    ``text`` must already be lower case. A pre-negation cue counts when it is
    within :data:`NEGATION_WINDOW` words before the term in the same clause; a
    post-negation cue must follow the term directly ("effusion is not seen").
    Pseudo-negations such as "cannot be ruled out" keep the term possible.
    """
    left, right = _clause_bounds(text, start, end)
    clause = text[left:right]
    if _PSEUDO_NEGATION.search(clause):
        return False
    before = " ".join(text[left:start].split()[-NEGATION_WINDOW:])
    if _PRE_NEGATION.search(before):
        return True
    after = text[end:right]
    # Finish the word the term starts ("pneumothora|x"), then look for a cue.
    after = re.sub(r"^\w*", "", after)
    return _POST_NEGATION.search(after) is not None


def affirmed_mention(text: str, term: str) -> bool:
    """Whether ``term`` occurs in ``text`` at a word start and is not negated.

    ``term`` may be a stem such as ``"hemorrhag"``; it must start a word, so
    ``"stone"`` does not match inside ``"gallstone"`` but ``"hemorrhag"``
    matches ``"hemorrhagic"``.
    """
    lowered = text.lower()
    pattern = re.compile(r"(?<!\w)" + re.escape(term.lower()))
    return any(
        not is_negated_at(lowered, match.start(), match.end())
        for match in pattern.finditer(lowered)
    )


_UNITS = {
    "mg": "mg",
    "mgs": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "mcg": "mcg",
    "μg": "mcg",
    "ug": "mcg",
    "microgram": "mcg",
    "micrograms": "mcg",
    "g": "g",
    "gm": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "ml": "ml",
    "cc": "ml",
    "l": "l",
    "iu": "iu",
    "unit": "iu",
    "units": "iu",
    "meq": "meq",
    "mmol": "mmol",
    "%": "%",
}
_DOSE = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?![\d.,])|\d+(?:[.,]\d+)?)\s*"
    r"(" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True)) + r")?"
    r"(?![\w])"
)


@dataclass(frozen=True)
class Dose:
    """A number with a canonical unit (``""`` when no unit follows)."""

    value: float
    unit: str


def normalise_ocr(text: str) -> str:
    """Normalise text before OCR error rates and term matching.

    Applies Unicode NFKC (full-width digits, ligatures, the micro sign),
    case folding, a space between a number and its unit (``500mg`` becomes
    ``500 mg``), and removes punctuation except decimal points, ``%`` and
    ``/``. Whitespace runs collapse to one space.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(\d)(?=[a-zμ%])", r"\1 ", text)
    kept = re.sub(r"[^\w%/.]+", " ", text)
    return " ".join(re.sub(r"(?<!\d)\.|\.(?!\d)", " ", kept).split())


def extract_doses(text: str) -> list[Dose]:
    """Every number in ``text`` with the unit that follows it, in order.

    A comma before exactly three digits groups thousands (``2,000 mg``); any
    other comma between digits is a decimal mark (``0,5 mg``). Numbers
    without a known unit keep ``unit=""`` so that "Dexter 60" still has a
    strength to compare.
    """
    normalised = unicodedata.normalize("NFKC", text).casefold()
    doses = []
    for match in _DOSE.finditer(normalised):
        number, unit = match.group(1), match.group(2) or ""
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", number):
            number = number.replace(",", "")
        doses.append(Dose(float(number.replace(",", ".")), _UNITS.get(unit, unit)))
    return doses


@dataclass(frozen=True)
class DoseComparison:
    """How the doses read by a model compare with the reference doses.

    Attributes:
        reference: Number of doses in the reference.
        exact: Reference doses read with the same value and unit.
        unit_errors: Same value but a different unit ("5 mg" read as "5 mcg").
        value_errors: Same unit but a different value ("500 mg" as "50 mg").
        missing: Reference doses with no counterpart at all.
        extra: Doses in the answer that match nothing in the reference.
    """

    reference: int
    exact: int
    unit_errors: int
    value_errors: int
    missing: int
    extra: int


def compare_doses(reference: str, hypothesis: str) -> DoseComparison:
    """Match doses of ``hypothesis`` to ``reference`` one to one.

    Exact matches are taken first; a leftover reference dose then pairs with
    a leftover answer dose of the same value (a unit error) or, failing that,
    of the same unit (a value error).
    """
    wanted = extract_doses(reference)
    read = extract_doses(hypothesis)
    exact = Counter(wanted) & Counter(read)
    left_wanted = list((Counter(wanted) - exact).elements())
    left_read = list((Counter(read) - exact).elements())
    unit_errors = value_errors = 0
    missing = 0
    for dose in left_wanted:
        same_value = next((d for d in left_read if d.value == dose.value), None)
        same_unit = next((d for d in left_read if d.unit == dose.unit), None)
        if same_value is not None:
            unit_errors += 1
            left_read.remove(same_value)
        elif same_unit is not None:
            value_errors += 1
            left_read.remove(same_unit)
        else:
            missing += 1
    return DoseComparison(
        reference=len(wanted),
        exact=sum(exact.values()),
        unit_errors=unit_errors,
        value_errors=value_errors,
        missing=missing,
        extra=len(left_read),
    )
