# Experiment artifacts and Git

Generated run output stays on disk but is ignored by Git: datasets, raw predictions,
API-response caches, checkpoints, archives, logs, packaged source copies and cloud
session state. Ignoring or untracking these files does not delete the local copies.
Existing Git history is unchanged.

The explicit exceptions in the root `.gitignore` retain:

- The [final pilot report](deepseek-reviewed-v2-20261022/v10-split-01/RESULTS.md),
  compact statistics, configuration, grading/reward audits and preservation records
  linked by the [short writeup](../docs/FINAL_PILOT_WRITEUP.md).
- Benchmark/validation notes and two small packaging metadata files.
- Runner source under the four `deepseek-*-20261021` directories used by the
  current scripts and tests. Their generated output remains ignored.
- Two recorded benchmark fixtures used by the Modal benchmark/cycle commands:
  `gasp-vllm-20261017-053328/results/vllm.json` and
  `gasp-modal-20261017-061436/A100-80GB/results/vllm-generated.json`.

Raw evidence and model weights are local assets, not Git backups. A fresh checkout
contains the implementation and curated reports, but reproducing historical runs
or optional archived-data tests requires the corresponding local artifacts. Some
older documentation links intentionally refer to that local evidence. Historical
full-file checksum receipts describe the files at experiment completion; their
referenced large files are not included in the repository.

Three existing cycle/Modal packaging tests additionally require the prepared
`data/full-v4/resolved_config.json` and raw GSM8K/MATH files under `data/raw/`.
These datasets were already ignored before this cleanup. Run data preparation
before those checks; they are not self-contained tests of a source-only checkout.

To retain a new report, add a narrow exception for that file and its parent
directories. Do not force-add entire run folders. Keep credentials, authentication
state and model exports ignored.
