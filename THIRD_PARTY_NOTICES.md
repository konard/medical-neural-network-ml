# Third-party material and licensing boundaries

Original software in this repository is dedicated to the public domain under
[LICENSE](LICENSE). Contributors must preserve that dedication. The Unlicense
does not grant rights to someone else's work.

## Dataset manifest and excerpts

`src/medbench/data/real_samples.json` records provenance, checksums and labels.
It includes proofread historical page transcriptions and short prescription
transcriptions. Their provenance and terms are recorded per entry. Treat any
source-derived expression under the source terms rather than assuming that
our Unlicense dedicates it to the public domain.

External images, document scans and DDXPlus files are downloaded into ignored
caches, not bundled in source distributions or wheels. Every manifest entry
records the author, source page, license and license link. The complete list
is in [data-sources.md](docs/case-studies/issue-1/data-sources.md).

- DDXPlus: Fansi Tchango et al., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  [dataset source](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585).
- RxHandBD: Md Masudul Islam, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  [dataset source](https://data.mendeley.com/datasets/dsb5r6vskg/1).
- Wikimedia Commons teaching images: per-entry CC0, public domain, CC BY or
  CC BY-SA. Attribution and share-alike obligations remain attached to the
  respective images and adaptations.
- Wikisource historical case histories: public-domain status recorded per
  source; jurisdiction-specific limits are discussed in the data guide.

`medbench fetch` creates `ATTRIBUTION.md`. Benchmark JSON reports include
`sources` with credits for the files used. If sharing images, crops,
transcriptions or reports containing source-derived text, preserve the
applicable attribution, license links and notices of modifications. Image
preparation can resize images, change color mode and convert them to PNG.
CC BY-SA adaptations must retain the appropriate share-alike license; that
does not change the license of our independent benchmark code.

## Models and dependencies

Model weights are not distributed with the package. MedGemma is subject to
Google's [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).
A third-party GGUF or MLX conversion does not remove those terms. Review and
accept required terms yourself before downloading; do not put credentials or
weights in Git.

MLX, MLX-VLM, llama.cpp, Transformers and other dependencies retain their own
licenses. Installing them does not dedicate them to the public domain. We
use their APIs and do not vendor their source. SmolVLM's optional MLX smoke
checkpoint is Apache-2.0, not a validated medical model. Historical research
logs can contain third-party quotations and model output; source references
and any accompanying attribution continue to apply.
