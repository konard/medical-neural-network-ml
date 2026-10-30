Additional reproducible downstream impact with the current **0.2.2** PyPI release:

```sh
uv pip compile --python-version 3.15 \
  --python-platform x86_64-manylinux_2_28 --only-binary :all: \
  --output-file /tmp/sentencepiece-repro.txt - <<'REQ'
sentencepiece==0.2.2
REQ
```

This has no compatible wheel. The same resolution on Python 3.14 succeeds. Our UniMedVL environment otherwise resolves current torch 2.15.0, torchvision 0.29.2 and transformers 5.22.1; SentencePiece is the remaining wheel availability limit preventing its move to Python 3.15.2.

We keep that isolated model environment on Python 3.14.10 in https://github.com/konard/medical-neural-network-ml/pull/26 while the main project uses 3.15.2. Publishing the cp315 wheel set in the planned 0.2.3 release should remove this hold. Thanks for the status about master already producing the wheels.
