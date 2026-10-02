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

## When publishing the arXiv version

1. Add the actual manuscript URL and identifier to `README.md` and
   `CITATION.cff`.
2. Run the tests, minimal example, and figure regeneration checks.
3. Create an annotated tag for the manuscript-linked commit, and record that
   tag or commit in the manuscript's reproduction instructions.
4. Keep published tags fixed. Use a new tag and describe any numerical changes
   for later revisions.

There is no arXiv-specific tag or identifier yet. The commit link above is the
immutable reference for the original public experiment snapshot.
