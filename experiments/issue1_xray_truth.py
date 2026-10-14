"""Print the ground truth of the first imaging phantoms used by the CPU smoke job.

Usage: python experiments/issue1_xray_truth.py [modality] [count]
"""

import sys

from medbench.synthetic_images import generate_phantoms

modality = sys.argv[1] if len(sys.argv) > 1 else "xray"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for i, phantom in enumerate(generate_phantoms(modality, count, 0, 128)):
    print(i, phantom.has_finding, phantom.side)
