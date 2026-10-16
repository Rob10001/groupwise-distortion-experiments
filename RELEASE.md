# Reproduction snapshots

The first public experiment snapshot is commit
[`32b51cb4521e2b799a8d180571c4f0f36c01287d`](https://github.com/Rob10001/groupwise-distortion-experiments/tree/32b51cb4521e2b799a8d180571c4f0f36c01287d)
(October 1, 2026). It contains the inputs, implementation, and frozen numerical
results described in the manuscript. Subsequent documentation, example, and
licensing improvements do not change those results.

To identify the exact code used for a reproduction, record:

```sh
git rev-parse HEAD
python --version
python -m pip freeze
```

Also retain the reproduction command and the generated `run_info.json` files.
The bundled beta-source manifests contain file checksums; split metadata
records input and respondent-split hashes. Some historical local paths and
source revision fields were removed when preparing the original distribution.
Those omissions do not alter the numerical arrays or input bytes.

## Paper and release versions

The paper is available as
[arXiv:2610.05450](https://arxiv.org/abs/2610.05450), first submitted on
October 4, 2026. [Version 1](https://arxiv.org/abs/2610.05450v1) describes
the experiments in the original public snapshot linked above.

The arXiv identifier identifies the paper; record a repository commit to
identify the code used for a reproduction. For future manuscript-linked
releases, run the tests, minimal example, and figure regeneration checks,
then create an annotated tag and record the tag or commit in the paper's
reproduction instructions. Keep published tags fixed; use a new tag and
describe any numerical changes for later revisions.
