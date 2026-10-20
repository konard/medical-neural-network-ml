"""Temporary cloud GPU hosts for the GPU model tests (issue #23).

``python -m scripts.cloud_gpu`` creates a GPU machine or cluster on AWS,
Azure, Google Cloud or Hetzner, runs ``scripts/gpu-host-run.sh`` on it,
collects the reports and deletes everything again. It also drives GPU hosts
we keep (over SSH), existing Kubernetes clusters and ephemeral GitHub Actions
runners. Only the Python standard library is used, so any CI system (GitHub
Actions, GitLab CI, a cron job) can call it; the cloud work itself is done by
Terraform or OpenTofu (``infra/terraform``), ``ssh`` and ``kubectl``.
See docs/cloud-gpu.md.
"""
