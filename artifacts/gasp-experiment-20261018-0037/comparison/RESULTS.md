# Completed exploratory comparison

Both conditions completed the frozen endpoint: 35 agent and 35 polluter updates,
seven alternating cycles each, seed 42. The pipeline and both analyses finished
2026-10-18 at 10:28 UTC (05:28 CDT). Modal reported no active apps or containers
when checked at approximately 15:20 UTC. No further GPU or API work was launched.

This run showed no improvement over the initial model and did not establish a
split-versus-unified advantage. Training-output quality and answer formatting
are material limitations of the result.

## Held-out results

| Model | Corrupted accuracy, 100 questions / 400 windows | Clean accuracy on selected 100 questions | Strict corpus: 95 questions / 380 windows |
|---|---:|---:|---:|
| Initial | 57.75% | 80.00% | 58.95% |
| Unified | 52.75% | 70.00% | 54.47% |
| Split | 50.50% | 70.00% | 50.79% |

Primary split-minus-unified difference: **−2.25 percentage points**, paired
question-bootstrap 95% interval **[−7.50, +3.00]**. Strict-corpus difference:
**−3.68 points**, interval **[−8.95, +1.58]**. Both intervals include zero;
neither establishes equivalence or a winner.

Relative to the initial model, unified changed by −5.00 points [−10.25, 0.00]
and split by −7.25 points [−12.25, −2.00] on the primary corruption metric.
The strict analysis also shows lower point estimates for both trained models.
Clean accuracy here describes the selected questions, not general GSM8K accuracy.

Intervals use 10,000 paired question resamples, analysis seed 10002. They are
conditional on one training seed and one frozen corruption corpus; they do not
measure training-seed or regenerated-corruption variability. See
[summary.json](summary.json) and [quality-sensitivity.json](quality-sensitivity.json).

## Execution and training checks

Both endpoints have identical question/alpha schedule hashes, matching source,
configuration, tokenizer, datasets, package versions, and A100-SXM4-80GB hardware.
The initial 32-answer prediction agreement checks passed for both conditions.
All 14 frozen-opponent checks per condition passed. Every recorded gradient
norm was finite, and all 70 updates per condition are present in order.

| Diagnostic | Unified | Split |
|---|---:|---:|
| Agent groups with nonconstant rewards | 24/35 | 21/35 |
| Polluter groups with nonconstant rewards | 34/35 | 33/35 |
| Nonempty agent repair targets | 35/35 | 31/35 |
| Median repair target length | 40 tokens | 22 tokens |
| Nonempty polluter-training windows | 521/560 | 520/560 |
| Exact polluter-training tag compliance | 30/560 | 32/560 |
| Unparsed final answers on 400 evaluation corruptions | 103 | 106 |

The initial model had 50 unparsed answers on those same 400 corruptions. These
formatting failures contribute to the recorded metric; the score drop cannot
be interpreted solely as loss of mathematical reasoning ability. Exact tag
compliance is a structural check and does not establish a mathematical error.

A post-run qualitative spot check inspected agent-stage windows/repairs at
global steps 1, 31, 61, 65 and candidate zero at polluter steps 6, 36, 66, 70 in
both conditions. Fifteen of these sixteen windows were nonempty and stated
correct solutions; the other was empty. The sampled repairs often supplied
answers rather than a local correction, and two unified repairs retained
pollution tags or prompt text. This small, deterministic spot check is not an
estimate of the entire training corpus's defect rate. It does show that successful
execution and nonzero gradients do not establish successful adversarial training.
See [training-diagnostics.json](training-diagnostics.json) and
[qualitative-training-review.json](qualitative-training-review.json).

Individual output-cap hits were not retained in the training histories, so a
precise cap-hit rate cannot be reconstructed from aggregate token counts alone.
The frozen verifier, outputs, endpoints and corpus were not changed after seeing
these results. The diagnostic review does not establish the cause of degradation.

## Runtime and accounting

| Stage | Measured worker time |
|---|---:|
| Shared baseline | 13.82 minutes |
| Unified, including initialization/checkpoints/evaluation | 2.42 hours |
| Split, including initialization/checkpoints/evaluation | 3.29 hours |

The two training workers took **5.71 hours combined**, somewhat longer than the
earlier 4.5–5.5-hour remaining-time estimate. Checkpoint time was 12.36 minutes
for unified and 30.21 minutes for split. The corresponding training-update
totals were 1.79 and 2.21 hours. These timings include workload and infrastructure
variation and do not isolate the cost of the architecture alone.

The final conservative ledger is **9.7522 of 10 GPU-hours**, including prior
diagnostics, preparation, failed attempts, and startup allowances. It still
includes full 660-second reservations for two brief failed helper starts; this
is budget accounting, not Modal's metered bill. The extra 0.05-hour planning
reserve for the old disconnect tail also fits under ten hours. Both final
checkpoints remain on the Modal volume; large checkpoint files were not
downloaded to the Mac.

## Preserved assumptions and qualifications

This is the approved reduced, single-seed experiment, not a full paper replication.
The split treatment changes parameter sharing and optimizer histories. Both
conditions use the same 100-question MATH training pool, with 70 scheduled
contexts and 35 active contexts per role, G=16, full tuning, and fixed caps.

All 400 evaluation windows were reviewed for mathematical errors before scores.
Six disclosed generation-instruction misses remain, including four windows with
correct-answer hints. The strict analysis excludes all four windows for each of
five affected questions, as fixed before outcomes. Both analyses are reported.
Inherited late-cut clean prefixes can also contain answers. Twelve source traces
were replaced with reviewed draws before corpus freezing; this extra semantic
acceptance step is a documented reconstruction choice. The final corpus is bound
by SHA256 `eed191f8d8e1781d6bea0d97dbb58dedf57a31c41d28a479577c8871d2620260`.
