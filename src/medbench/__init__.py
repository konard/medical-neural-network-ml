"""Tools to test medical AI models on real, openly licensed data.

``medbench`` is the code part of the research plan for issue #1 (see
``docs/case-studies/issue-1``). It contains:

* :mod:`medbench.models` - the candidate models and their sizes;
* :mod:`medbench.resources` - memory, speed and training-cost calculator;
* :mod:`medbench.support` - which model, runtime, weight format and task
  combinations are supported, and which of them were measured;
* :mod:`medbench.hardware` - host detection and resource budgets;
* :mod:`medbench.real_data` - real public domain and Creative Commons
  documents and scans, downloaded at test time and checked by SHA-256;
* :mod:`medbench.ddxplus` - DDXPlus patients (CC BY 4.0) for the
  differential diagnosis task;
* :mod:`medbench.synthetic_cases` - a small text-only toy diagnosis set and
  the answer parser;
* :mod:`medbench.augment` - label-preserving data augmentation;
* :mod:`medbench.metrics` - scoring functions and confidence intervals;
* :mod:`medbench.clinical_text` - negation, abstention, dose/unit and OCR
  normalisation rules used by the scores;
* :mod:`medbench.backends` - ways to call a model;
* :mod:`medbench.benchmark` - tasks and the benchmark runner.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
