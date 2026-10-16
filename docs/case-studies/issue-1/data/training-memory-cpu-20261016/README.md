# Training memory on a CPU, 2026-10-16

No GPU was available, so these runs check how the training-memory terms grow
on small models, not how large a MedGemma run is. No patient data is used: the
text inputs are random token ids and the image is a plain grey square.

Hardware: 6-core x86-64 CPU ("AMD EPYC Processor (with IBPB)") without bf16
instructions, in a container limited to 3.1 GB of memory. Python 3.15.0,
torch 2.14.1+cpu, transformers 5.20.1, peft 0.21.2, accelerate 1.15.0;
bitsandbytes is not installed.

Models, both Apache-2.0, read from the Hugging Face cache offline:

- `HuggingFaceTB/SmolLM2-135M`, commit `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- `HuggingFaceTB/SmolVLM-256M-Instruct`, commit
  `7e3e67edbbed1bf9888184d9df282b700a323964`.

Every run used fp32 weights, no autocast, batch size 1, gradient
checkpointing, `torch.optim.AdamW` and 3 steps. Each JSON report holds the
full configuration, versions, the estimate it was compared with and the
measured peaks. The reports were produced by:

```bash
HF_HOME=<cache> experiments/issue11_training_memory_cpu.sh <out> <python>
```

The script runs `scripts/measure_training_memory.py` once per row below,
under `timeout` and under `experiments/issue11_rss_watchdog.py`, which kills
a run whose resident memory passes 2.3 GB.

| Report | Status | Image tokens | Load peak GB | Step peak GB | Estimate GB (range) | Peak / estimate | Optimizer state GB | Checkpoint GB | Step s |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `preflight-smollm2-135m-lora-r16-seq256` | refused | – | – | – | 2.08 (1.67–3.12) | – | – | – | – |
| `smollm2-135m-qlora-seq256` | unsupported | – | – | – | 1.88 (1.51–2.82) | – | – | – | – |
| `smollm2-135m-lora-r8-seq256` | measured | – | 1.201 | 1.260 | 2.04 (1.63–3.05) | 0.62 | 0.020 | 0.030 | 5.7, 6.1, 23.4 |
| `smollm2-135m-lora-r16-seq256` | measured | – | 1.202 | 1.296 | 2.08 (1.67–3.12) | 0.62 | 0.039 | 0.059 | 7.3, 8.7, 5.3 |
| `smollm2-135m-lora-r64-seq256` | measured | – | 1.201 | 1.511 | 2.36 (1.89–3.54) | 0.64 | 0.156 | 0.235 | 21.2, 5.5, 4.5 |
| `smollm2-135m-lora-r16-seq1024` | measured | – | 1.202 | 1.998 | 2.74 (2.19–4.11) | 0.73 | 0.039 | 0.059 | 35.1, 23.1, 20.4 |
| `smolvlm-256m-img384-split-token-probe` | refused | 1088 | 1.970 | – | 2.16 (1.73–3.24) | – | – | – | – |
| `smolvlm-256m-lora-r16-img512-nosplit-seq256` | measured | 64 | 1.971 | 1.947 | 2.33 (1.86–3.49) | 0.84 | 0.046 | 0.070 | 28.7, 20.1, 29.0 |
| `smolvlm-256m-lora-r16-img512-split-seq1280` | killed above 2.3 GB | – | – | – | 3.21 (2.57–4.81) | – | – | – | – |
| `smollm2-135m-full-seq256` | killed above 2.3 GB | – | – | – | 3.87 (3.10–5.81) | – | – | – | – |

How to read it:

- **Load peak** is the highest resident memory while loading the model.
  **Step peak** is the highest resident memory during the
  training steps; the probe restarts the kernel's peak counter after loading
  (`host_peak_includes_load: false`), so it is not the loading peak again.
- **Estimate** is `medbench`'s fine-tuning formula for the same settings,
  with its 0.8–1.5× range.
- **refused**: the first run had no `--force`, and the probe stopped before
  loading because the estimate's high end, 3.12 GB, exceeded the 1.78 GB it
  could use (1.98 GB free minus 10 % headroom). The token probe stopped
  because 64 tokens are fewer than the 1 137 of one image prompt, 1 088 of
  them image tokens; that refusal is how the image token count was measured.
- **unsupported**: QLoRA here means bitsandbytes NF4, which needs CUDA.
- **killed above 2.3 GB**: the watchdog printed `killed at 2.32 GB` (SmolVLM
  with image splitting at 1 280 tokens) and `killed at 2.41 GB` (full
  fine-tune). SIGKILL leaves no time to update the report, so it still says
  `started`. Both estimates were above 2.3 GB, so they were expected not to
  fit.

What the numbers show:

- The step peak grew by 0.70 GB from 256 to 1 024 tokens with LoRA rank 16
  (0.68 GB in an earlier run of the same script): 18–19 bytes per token and
  vocabulary entry for SmolLM2's 49 152-token vocabulary. `medbench` counts
  16, for the fp32 logits and their gradient.
- LoRA rank 8 to 64 added 0.25 GB, of which 0.14 GB is optimizer state.
  Checkpoints of the adapter and optimizer state stayed below 0.24 GB.
- SmolVLM's processor turned one 384 px image into 1 088 image tokens with
  its default image splitting (the token probe), and one 512 px image into
  64 with `do_image_splitting=false`. A processor-only check,
  `experiments/issue11_smolvlm_image_tokens.py`, gave 1 088 and 64 for both
  sizes, because the processor resizes the image before it splits it into
  tiles. The image size alone does not fix the count; the
  processor settings do.
- For SmolVLM without splitting, loading (1.97 GB) was the larger peak, not
  the step (1.95 GB).
- Measured peaks were 0.62–0.84 of the estimate. The estimate assumes bf16
  weights plus 1.5 GB of GPU framework overhead; these runs held fp32 weights
  with no CUDA context. The ratio is not a calibration for GPU runs.

Not measured: anything on a GPU or on Apple silicon, QLoRA, MedGemma and
UniMedVL. `docs/hardware-workflows.md` lists the commands for those runs.
