Reproduced on **Node 26.13.0 / npm 12.2.1** while replacing an unlocked CI scanner install in https://github.com/konard/medical-neural-network-ml/pull/26:

```sh
mkdir repro && cd repro
printf '%s\n' '{"private":true,"devDependencies":{"npm":"12.2.1"}}' > package.json
npm install --ignore-scripts
npm audit
npm audit fix --ignore-scripts
```

The published npm tarball bundles `undici@6.29.0` and also `ip-address@10.7.2`. The audit reports GHSA-2m5v-pch6-74hx and other undici advisories fixed in 6.29.1, plus GHSA-rxp7-6hxx-x8q5 (`ip-address<=10.7.3`, fixed in 10.7.4; current release 10.7.5).

The audit says `fixAvailable: true`, but `npm audit fix` says both are bundled and cannot be fixed automatically. Nested overrides and a fresh install with those overrides also leave both bundled versions unchanged. No denial-of-service payload was executed.

Suggested fix: refresh both bundled dependencies in the npm release, and make the audit's proposed fix distinguish an available reification from a bundled dependency requiring an npm release. We chose pnpm 12.16.1 for the scanner's new frozen lock instead, so we do not add npm's affected bundle to our project dependency tree; its equivalent project audit reports zero vulnerabilities.
