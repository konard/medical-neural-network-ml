# Research notes

- Python 3.15 locks: PyYAML ships no cp315 wheels, and building from the sdist is
  blocked by `--only-binary`. Upstream: https://github.com/yaml/pyyaml/issues/981,
  https://github.com/yaml/pyyaml/issues/951, https://github.com/yaml/pyyaml/issues/994.
- pytest-asyncio 1.4.1.post1 is a post release, and the old `[\d.]+` regex
  cut the floor at `1.4.1`.
- sympy 1.14.0 caps `mpmath<1.4`. torch 2.15.0 pins its CUDA wheel set exactly.
- Tools considered: Renovate could group these updates the same way. Dependabot is
  kept because it is already configured and covers every ecosystem here.
  `pre-commit autoupdate`, `uv pip compile --upgrade`, `pnpm update --latest` and
  `terraform init -upgrade` did the actual updates.
