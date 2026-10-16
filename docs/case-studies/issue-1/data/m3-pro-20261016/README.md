# M3 Pro local validation, 2026-10-16

Hardware: Mac15,7, M3 Pro, 12 CPU cores (6 performance / 6 efficiency),
18 GPU cores, 18 GiB unified memory; macOS 15.7.7, native arm64 Python 3.12.
No device serial numbers, usernames, tokens or patient records are included.

- `metal-smoke.json`: the bounded Metal matrix check.
- `host.json`: reclaimable RAM at capture time, using the new Mac detector.
- `environment.txt`: installed dependency versions (the distribution was
  still named `my-package` when installed; the source metadata is corrected).
- `m3-pro-smolvlm-diagnosis.json`: one toy text case, 32 output tokens.
- `m3-pro-smolvlm-xray.json`: one real X-ray case, 48 output tokens.

Model: `mlx-community/SmolVLM-256M-Instruct-4bit`, commit
`69cb5195f414ceb6398c5581254673d2c6c8d0d8`, Apache-2.0, downloaded into an
ignored cache. Weight SHA-256:
`a283e0ef139e8a31fe5dc25bcdb0f3ee4ecf28aa6820840e49eb3519cdbec3e6`.
The generation latencies exclude loading. The model did not follow either
medical answer format. These are wiring tests, not a medical baseline.

The X-ray input is by Hellerhoff, CC BY-SA 3.0; its source and license are in
the report's `sources` field. No image is committed. Input preparation
resized and converted it to PNG. Model outputs are preserved as evidence;
the source media retain their original terms. See `THIRD_PARTY_NOTICES.md`.

Run commands are in `docs/apple-silicon.md`. Each worker ran at nice level 10
with a 120-second hard process deadline and dynamic MLX memory preflight.
