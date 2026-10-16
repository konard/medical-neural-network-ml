# Peak resources of every GitHub workflow, 2026-10-16

All seven workflows ran on commit `165a710` of pull request #12 and passed:

| Workflow | Run |
| --- | --- |
| CI/CD Pipeline | [38950419706](https://github.com/konard/medical-neural-network-ml/actions/runs/38950419706) |
| Docs | [38950419686](https://github.com/konard/medical-neural-network-ml/actions/runs/38950419686) |
| Security | [38950419680](https://github.com/konard/medical-neural-network-ml/actions/runs/38950419680) |
| Compatibility | [38950419689](https://github.com/konard/medical-neural-network-ml/actions/runs/38950419689) |
| Workflows | [38950419783](https://github.com/konard/medical-neural-network-ml/actions/runs/38950419783) |
| Broken Link Checker | [38950419799](https://github.com/konard/medical-neural-network-ml/actions/runs/38950419799) |
| Model Tests | [38950419674](https://github.com/konard/medical-neural-network-ml/actions/runs/38950419674) |

`table.md` was printed by:

```bash
python -I experiments/issue11_ci_resources.py /tmp/ci-res-20261016 \
  38950419706 38950419686 38950419680 38950419689 38950419783 38950419799 38950419674
```

It has one row per job, with the wall time from the GitHub API, then one `↳`
row per step run under `medbench monitor`. The JSON reports it read are kept
here, under `<run>/artifacts/<artifact>/`. Job logs, job lists and the other
artifacts are not.

How to read the columns:

- **Host used GB** is the memory the whole runner used, other processes
  included: at the start of the step, then the highest sample. Samples are
  taken every 2 seconds, or every 5 for the model download and run
  (`interval_seconds` in each report), so a shorter spike can be missed.
  On macOS the value comes from `vm_stat`.
- **Largest child RSS GB** is the peak resident memory of the largest process
  the step started (`ru_maxrss` of the waited-for children). It is exact,
  but it is the peak of one process, not the sum.
- **Disk used GB** is free disk at the start minus the lowest free disk seen.
  Disk is sampled at the same interval.

What the numbers show:

- The heaviest step is the MedGemma 4B Q4_K_M CPU smoke test: host memory
  went from 1.85 to 5.74 GB of the runner's 16.8 GB, and llama-server peaked
  at 4.71 GB RSS. That is within the "at least 6 GB free RAM" target in
  `docs/hardware-workflows.md`. The job took 440 s; the medical tasks took
  375 s of it.
- The model came from `actions/cache` in this run, so "Download model" took
  2.4 s and used no new disk. A cold download is about 3.3 GB and is not
  measured here.
- Every other monitored step stays below 0.3 GB RSS. The macOS dependency
  audit is the largest disk user, 1.84 GB, for its two isolated environments.
- macOS runners report 7.5 GB in total and about 4.1–4.3 GB already in use
  before the step starts, so only about 3 GB is free for a step there.

Not measured:

- The Workflows and Broken Link Checker jobs, and steps that run only a
  pinned container or action (CodeQL, Dependency Review, rulesets), are not
  wrapped in `medbench monitor`. Only their wall time is listed.
- The publishing jobs of `release.yml` run only on main after a release
  commit and are not wrapped. They upload packages and need no model.
- The self-hosted GPU job was skipped; it runs only when dispatched by hand
  from main. Its reports also include GPU memory, and they are the source
  for the MedGemma 27B and UniMedVL numbers once someone runs it.

Hosted runners differ between runs: the same job can land on a machine with
different CPUs or disk. These are single measurements, not ranges.
