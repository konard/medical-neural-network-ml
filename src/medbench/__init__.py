"""Tools to test medical AI models on synthetic, shareable data.

``medbench`` is the code part of the research plan for issue #1 (see
``docs/case-studies/issue-1``). It contains:

* :mod:`medbench.models` - the candidate models and their sizes;
* :mod:`medbench.resources` - memory, speed and training-cost calculator;
* :mod:`medbench.hardware` - host detection and resource budgets;
* :mod:`medbench.synthetic_cases`, :mod:`medbench.synthetic_documents`,
  :mod:`medbench.synthetic_images` - generators of fake patients, documents
  and scans with known answers;
* :mod:`medbench.augment` - label-preserving data augmentation;
* :mod:`medbench.metrics` - scoring functions;
* :mod:`medbench.backends` - ways to call a model;
* :mod:`medbench.benchmark` - tasks and the benchmark runner.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
