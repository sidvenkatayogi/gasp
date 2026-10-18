# Completed experiment

Updated 2026-10-18T15:20:58.589157+00:00.

Both unified and split reached 70 global updates (35 per role), completed endpoint
evaluation, and passed the paired identity/schedule checks. Primary and predeclared
strict analyses finished 2026-10-18 10:28 UTC. The sequential controller exited 0.

Modal app and container listings are empty; no GPUs are active. No new API or GPU
work was launched. Conservative ledger: 9.752238140554724 GPU-hours of 10, with the
extra 0.05-hour disconnect-tail planning reserve still within budget.

Primary corruption accuracy: initial 57.75%, unified 52.75%, split 50.50%.
Split-minus-unified 95% paired interval: -7.50 to +3.00 percentage points.
Strict subset: initial 58.95%, unified 54.47%, split 50.79%. Neither analysis establishes
a split-versus-unified winner. Training-output quality and answer formatting are
material limitations; see comparison/RESULTS.md for the reviewed interpretation.

All artifacts are preserved. Large final checkpoints remain on gasp-exploratory-artifacts
at gasp-experiment-20261018-0037/train/{unified,split}/checkpoint.pt.
