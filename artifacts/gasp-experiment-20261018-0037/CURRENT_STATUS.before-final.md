# Current experiment status

Updated 2026-10-18T04:47:33.750373+00:00. This is a checkpointed status, not a live feed.

The shared baseline completed successfully: 100 clean and 400 corrupted answers,
829.034 worker seconds (789.220 evaluation seconds). Modal confirms its app is
stopped with zero tasks. The model hash and frozen endpoint were verified.

Unified training has been launched in verified detached Modal app
[ap-JfnAqevYvX6rH8pQBkdqfp](https://modal.com/apps/sidvenkatayogi/main/ap-JfnAqevYvX6rH8pQBkdqfp),
with a 10,500-second hard timeout. The retry has matching A100-SXM4-80GB hardware. Initialization and the
32-answer shared-baseline identity/prediction check passed (56.759 seconds);
training is now running. Its first optimizer update is in progress. The sequential controller session is 98888;
pipeline-status.json and unified-training-console.log record progress. It queues
split only after unified reaches the frozen endpoint, then runs both CPU analyses.
It stops for inspection on errors or incomplete training. A detached remote stage
survives a local network interruption; launching the next stage still requires
the local controller and connectivity.

The frozen schedule is 35 updates per role, seven 5+5 cycles, and 70 global updates in each
condition, seed 42 and G=16. Train unified then split sequentially from the same
initial checkpoint. The paired-plan is already frozen: do not change this
schedule using baseline or trained scores.

Training uses 100 retained MATH questions. Corpus-v4 is the reviewed 100-question/
400-corruption GPT-6.1 Sol corpus, SHA256
`eed191f8d8e1781d6bea0d97dbb58dedf57a31c41d28a479577c8871d2620260`.
All 400 have a mathematical error. Six disclosed prompt-instruction misses remain;
the required pre-outcome quality sensitivity excludes five questions, leaving
95 questions / 380 corruptions. Report both full and strict results. The policy and
exact exceptions are in corpus-v4/corpus-review.json. API generation is complete;
no further API requests are planned.

Use run_detached.py for all GPU stages. Keep the total 36,000-second ledger guard.
The baseline result bundle has returned; its actual runtime was used
to set the remaining wall-time caps. The first-condition cap is the smaller
of 10,500 seconds or half the remaining budget after a 180-second cleanup reserve,
minus its 60-second lifecycle allowance. The second condition can use the actual
remaining budget, with the same fixed 70-update endpoint.

If Modal assigns a different A100 variant, the existing baseline identity check
rejects it before any update. Preserve any failed initialization artifacts before
retrying. Do not weaken the identity check. Do not download large .pt training
checkpoints to this Mac.

After both endpoints, run scripts/analyze_exploratory.py with the baseline,
unified/step-000070 and split/step-000070 outputs. Then run
analyze_quality_sensitivity.py against the comparison directory, the saved
results/corpus-review.json and results/prepare/eval.jsonl.manifest.json.
It must reproduce the full-set primary estimates before applying exclusions.

Budget after baseline and one rejected initialization: 4.011670774844 charged GPU-hours, including 660-second
conservative reservations for each of two failed helper starts. The server's
stopped-app timestamps are now saved in modal-lifecycle-evidence.json. If
reconciling these reservations, do it only after the active worker finishes,
preserve original charges, and retain a lifecycle allowance. Do not modify the
ledger while a worker holds an in-memory copy. The 35-update-per-role schedule remains fixed.

The earlier network-interruption heartbeat workaround failed. Only the later
supported App.run(detach=True) launcher is verified; do not repeat that workaround.

The first unified attempt used an A100 PCIe and was rejected before any updates
by the unchanged GPU identity check. Its 36.619 worker seconds plus 60-second
allowance remain charged. The two initialization metadata files were preserved
locally and remotely in results/rejected-initializations/train-unified-20261018-044322
(remote paths omit results/), read-back verified, then removed from the active
condition directory. No history or checkpoint existed. The rejected app stopped
with zero tasks. archive_rejected_initialization.py records the guarded procedure.
