"""Print the ground truth of the first real images used by the CPU smoke job.

Usage: python experiments/issue1_xray_truth.py [modality] [count]
"""

import sys

from medbench.benchmark import ImagingTask

modality = sys.argv[1] if len(sys.argv) > 1 else "xray"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for item in ImagingTask(modality=modality, count=count).items():
    truth = item.truth
    print(item.item_id, truth.truth, truth.license, truth.source_url)
