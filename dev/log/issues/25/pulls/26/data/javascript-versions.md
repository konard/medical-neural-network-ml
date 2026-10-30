# Complete JavaScript and pnpm toolchain inventory

npm registry queried 2026-10-30. Before this PR only secretlint and its
preset were pinned (13.0.7); all transitive packages and the package manager
were resolved by npx at execution time. All pnpm platform executables are
included below, even those not installed on this Linux host.

| Dependency | Before | Now | Registry latest | Reason if held |
| --- | --- | --- | --- | --- |
| @azu/format-text | unlocked / new | 1.0.2 | 1.0.2 | — |
| @azu/style-format | unlocked / new | 1.0.1 | 1.0.1 | — |
| @babel/code-frame | unlocked / new | 7.29.7 | 8.0.6 | parse-json@8.3.0 requires `^7.26.2` |
| @babel/helper-validator-identifier | unlocked / new | 7.29.7 | 8.0.6 | @babel/code-frame@7.29.7 requires `^7.29.7` |
| @pnpm/exe.android-arm64 | unlocked / new | 12.16.1 | 0.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.android-x64 | unlocked / new | 12.16.1 | 0.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.darwin-arm64 | unlocked / new | 12.16.1 | 12.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.darwin-x64 | unlocked / new | 12.16.1 | 12.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.freebsd-x64 | unlocked / new | 12.16.1 | 0.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.linux-arm64-musl | unlocked / new | 12.16.1 | 12.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.linux-arm64 | unlocked / new | 12.16.1 | 12.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.linux-ppc64 | unlocked / new | 12.16.1 | 0.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.linux-riscv64 | unlocked / new | 12.16.1 | 0.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.linux-s390x | unlocked / new | 12.16.1 | 0.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.linux-x64-musl | unlocked / new | 12.16.1 | 12.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.linux-x64 | unlocked / new | 12.16.1 | 12.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.win32-arm64 | unlocked / new | 12.16.1 | 12.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @pnpm/exe.win32-x64 | unlocked / new | 12.16.1 | 12.0.0 | pnpm@12.16.1 requires `12.16.1` |
| @secretlint/config-creator | unlocked / new | 13.1.0 | 13.1.0 | — |
| @secretlint/config-loader | unlocked / new | 13.1.0 | 13.1.0 | — |
| @secretlint/core | unlocked / new | 13.1.0 | 13.1.0 | — |
| @secretlint/formatter | unlocked / new | 13.1.0 | 13.1.0 | — |
| @secretlint/node | unlocked / new | 13.1.0 | 13.1.0 | — |
| @secretlint/profiler | unlocked / new | 13.1.0 | 13.1.0 | — |
| @secretlint/resolver | unlocked / new | 13.1.0 | 13.1.0 | — |
| @secretlint/secretlint-rule-preset-recommend | 13.0.7 | 13.1.0 | 13.1.0 | — |
| @secretlint/source-creator | unlocked / new | 13.1.0 | 13.1.0 | — |
| @secretlint/types | unlocked / new | 13.1.0 | 13.1.0 | — |
| @secretlint/walker | unlocked / new | 13.1.0 | 13.1.0 | — |
| @textlint/ast-node-types | unlocked / new | 15.8.0 | 15.8.0 | — |
| @textlint/linter-formatter | unlocked / new | 15.8.0 | 15.8.0 | — |
| @textlint/module-interop | unlocked / new | 15.8.0 | 15.8.0 | — |
| @textlint/resolver | unlocked / new | 15.8.0 | 15.8.0 | — |
| @textlint/types | unlocked / new | 15.8.0 | 15.8.0 | — |
| @types/normalize-package-data | unlocked / new | 2.4.4 | 2.4.4 | — |
| ajv | unlocked / new | 8.20.0 | 8.20.0 | — |
| ansi-escapes | unlocked / new | 7.3.0 | 7.3.0 | — |
| ansi-regex | unlocked / new | 5.0.1 | 6.5.0 | strip-ansi@6.0.1 requires `^5.0.1` |
| ansi-styles | unlocked / new | 4.3.0 | 7.0.0 | slice-ansi@4.0.0 requires `^4.0.0` |
| argparse | unlocked / new | 2.0.1 | 3.0.4 | js-yaml@4.3.2 requires `^2.0.1` |
| astral-regex | unlocked / new | 2.0.0 | 2.0.0 | — |
| binaryextensions | unlocked / new | 6.11.0 | 6.11.0 | — |
| boundary | unlocked / new | 2.0.0 | 2.0.0 | — |
| chalk | unlocked / new | 5.6.2 | 6.0.1 | @secretlint/formatter@13.1.0 requires `^5.6.2` |
| color-convert | unlocked / new | 2.0.1 | 3.1.3 | ansi-styles@4.3.0 requires `^2.0.1` |
| color-name | unlocked / new | 1.1.4 | 2.1.1 | color-convert@2.0.1 requires `~1.1.4` |
| debug | unlocked / new | 4.4.3 | 4.4.3 | — |
| editions | unlocked / new | 6.22.0 | 6.22.0 | — |
| emoji-regex | unlocked / new | 8.0.0 | 11.0.0 | string-width@4.2.3 requires `^8.0.0` |
| environment | unlocked / new | 1.1.0 | 1.1.0 | — |
| fast-deep-equal | unlocked / new | 3.1.3 | 3.1.3 | — |
| fast-uri | unlocked / new | 3.1.9 | 4.2.3 | ajv@8.20.0 requires `^3.0.1` |
| has-flag | unlocked / new | 5.0.1 | 5.0.1 | — |
| hosted-git-info | unlocked / new | 9.0.3 | 10.1.1 | normalize-package-data@8.0.0 requires `^9.0.0` |
| ignore | unlocked / new | 7.0.14 | 7.0.14 | — |
| index-to-position | unlocked / new | 1.2.0 | 1.2.0 | — |
| is-fullwidth-code-point | unlocked / new | 3.0.0 | 5.1.0 | slice-ansi@4.0.0 requires `^3.0.0`; string-width@4.2.3 requires `^3.0.0` |
| istextorbinary | unlocked / new | 9.5.0 | 9.5.0 | — |
| js-tokens | unlocked / new | 4.0.0 | 10.0.0 | @babel/code-frame@7.29.7 requires `^4.0.0` |
| js-yaml | unlocked / new | 4.3.2 | 5.5.0 | @textlint/linter-formatter@15.8.0 requires `^4.3.0`; rc-config-loader@4.1.4 requires `^4.1.1` |
| json-schema-traverse | unlocked / new | 1.0.0 | 1.0.0 | — |
| json5 | unlocked / new | 2.2.3 | 2.2.3 | — |
| lodash.truncate | unlocked / new | 4.4.2 | 4.4.2 | — |
| lodash | unlocked / new | 4.18.1 | 4.18.1 | — |
| lru-cache | unlocked / new | 11.5.3 | 11.5.3 | — |
| ms | unlocked / new | 2.1.3 | 2.1.3 | — |
| normalize-package-data | unlocked / new | 8.0.0 | 9.0.0 | read-pkg@10.1.0 requires `^8.0.0` |
| p-map | unlocked / new | 7.1.1 | 7.1.1 | — |
| parse-json | unlocked / new | 8.3.0 | 8.3.0 | — |
| picocolors | unlocked / new | 1.1.1 | 1.1.1 | — |
| picomatch | unlocked / new | 4.0.7 | 4.0.7 | — |
| pluralize | unlocked / new | 2.0.0 | 8.0.0 | @textlint/linter-formatter@15.8.0 requires `^2.0.0` |
| pluralize | unlocked / new | 8.0.0 | 8.0.0 | — |
| pnpm | unlocked / new | 12.16.1 | 12.16.1 | — |
| rc-config-loader | unlocked / new | 4.1.4 | 4.1.4 | — |
| read-pkg | unlocked / new | 10.1.0 | 10.1.0 | — |
| require-from-string | unlocked / new | 2.0.2 | 2.0.2 | — |
| secretlint | 13.0.7 | 13.1.0 | 13.1.0 | — |
| semver | unlocked / new | 7.8.5 | 7.8.5 | — |
| slice-ansi | unlocked / new | 4.0.0 | 9.0.1 | table@6.9.0 requires `^4.0.0` |
| spdx-correct | unlocked / new | 3.2.0 | 3.2.0 | — |
| spdx-exceptions | unlocked / new | 2.5.0 | 2.5.0 | — |
| spdx-expression-parse | unlocked / new | 3.0.1 | 5.0.0 | spdx-correct@3.2.0 requires `^3.0.0`; validate-npm-package-license@3.0.4 requires `^3.0.0` |
| spdx-license-ids | unlocked / new | 3.0.24 | 3.0.24 | — |
| string-width | unlocked / new | 4.2.3 | 8.3.0 | @textlint/linter-formatter@15.8.0 requires `^4.2.3`; table@6.9.0 requires `^4.2.3` |
| strip-ansi | unlocked / new | 6.0.1 | 7.2.0 | @textlint/linter-formatter@15.8.0 requires `^6.0.1`; string-width@4.2.3 requires `^6.0.1`; table@6.9.0 requires `^6.0.1` |
| structured-source | unlocked / new | 4.0.0 | 4.0.0 | — |
| supports-color | unlocked / new | 10.2.2 | 11.0.0 | supports-hyperlinks@4.6.0 requires `^10.2.2` |
| supports-hyperlinks | unlocked / new | 4.6.0 | 4.6.0 | — |
| table | unlocked / new | 6.9.0 | 6.9.0 | — |
| tagged-tag | unlocked / new | 1.0.0 | 1.0.0 | — |
| terminal-link | unlocked / new | 5.0.0 | 5.0.0 | — |
| text-table | unlocked / new | 0.2.0 | 0.2.0 | — |
| textextensions | unlocked / new | 6.11.0 | 6.11.0 | — |
| type-fest | unlocked / new | 4.41.0 | 5.10.0 | parse-json@8.3.0 requires `^4.39.1` |
| type-fest | unlocked / new | 5.10.0 | 5.10.0 | — |
| unicorn-magic | unlocked / new | 0.4.1 | 0.4.1 | — |
| validate-npm-package-license | unlocked / new | 3.0.4 | 3.0.4 | — |
| version-range | unlocked / new | 4.15.0 | 4.15.0 | — |
