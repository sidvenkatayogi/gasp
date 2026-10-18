# Exploratory unified-versus-split plan

Status: both conditions and both analyses completed on 2026-10-18 UTC within the
10 GPU-hour budget (conservative ledger: 9.7522 hours). No active Modal apps or
containers remained at the final check. See the
[results and diagnostic review](../artifacts/gasp-experiment-20261018-0037/comparison/RESULTS.md).
The user approved the revised plan on 2026-10-18 UTC. Implementation
passed 133 local tests (four CUDA-only skips) and 137 tests on the real GPU.
Preparation is complete: 100 retained MATH questions and 100 retained GSM8K
questions / 400 contexts. GPT corruption generation and the full semantic audit
are complete. Six outputs do not fully follow the generator prompt; these are disclosed in corpus-v4.
The required stricter sensitivity uses 95 questions / 380 corruptions. No model
scores informed these exclusions. The shared baseline completed in 829 worker seconds;
unified and split each completed all 70 global updates and endpoint evaluation.
Measured preparation costs and the trace quality
correction require **35 updates
per role, seven cycles per condition**, under the approved whole-cycle budget
adjustment. This decision precedes baseline and trained-model scores.
The original request was
100 optimizer updates per role and 20 alternating cycles. The user subsequently
specified **about 10 total GPU-hours, prioritizing lower compute cost**.

Measured results and calculations are in
[the cycle report](../artifacts/CYCLE_BENCHMARK.md). The initial approved target
was 50 updates per role / 10 cycles, subject to reducing whole cycles to fit the
measured full pipeline. Keep 100 evaluation questions / 400 corruptions and the
100-question training pool. GPT API work runs on CPU.

The current allocation and calculation are recorded in
`artifacts/gasp-experiment-20261018-0037/training-budget-decision.json`:
3.738 charged GPU-hours after source correction (including conservative full
reservations for two helper startup failures), 3.979 projected paired training
and checkpoint hours plus a 20% training allowance, 1.15 hours reserved for
evaluations/initialization/spot-checks, and 0.05 hours for an uncertain disconnect
cleanup tail: **9.713 hours total**. These are planning reserves, not a completion
guarantee. The ten-hour watchdog applies. The original source contexts are
archived; reviewed version 2 replaces 12 traces while keeping all 100 questions.

The execution record is under `artifacts/gasp-experiment-20261018-0037`. Its frozen
source uses offline preparation batches of 16 questions (64 K=4 samples), while
live training/evaluation batches remain 16 sequences. This execution choice is
recorded in both conditions' common config. The shared-baseline spot-check repeats
one complete clean and one complete corrupted batch in each condition to preserve
padding shapes: 64 extra answers, for 1,564 total evaluation answers. The compute
ledger includes 2,883 seconds from the recent cycle diagnostics and all new stage
worker time plus a lifecycle allowance. GPU jobs release between preparation,
CPU/API corpus generation/review, baseline and training.

Use a fresh 70-global-update schedule in each condition, evaluate before training
and at update 70, and visit 70 of the shared 100-question pool without replacement:
35 active contexts per role, 1,120 trained GRPO trajectories per condition.
The schedule has seven warmup updates and guidance decays over the final seven
agent updates; both conditions use the same schedule. Preserve the five-update
role blocks. Do not resume
the four-question timing diagnostic as the actual experiment. Use the benchmark's
matched opt-in HF batch size 16 and declared 2048 input cap (1024 output cap).

## Scientific design after independent paper review

An independent GPT-6 Astra reviewer read the complete paper and appendices before
reviewing this plan. Its [review](EXPLORATORY_PAPER_REVIEW.md) supports the reduced
design as a **single-seed, fixed-budget recoverability comparison**, subject to the
following revisions. It does not establish convergence, equivalence, broad
robustness or an advantage reproducible across training seeds.

The primary question is: at the same allocated role-update and rollout budget,
how does a separately parameterized agent/polluter change endpoint recoverability
relative to our shared-parameter implementation? The treatment includes separate
optimizer histories and greater total parameter storage; it does not isolate
gradient interference alone. Keep the 1.5B backbone, G=16 and repair objective.
No additional training baseline or smaller-model sweep is required for this
first question. Thirty-five agent updates see only thirty-five active question contexts;
1,120 trajectories across both roles do not represent 1,120 independent problems.

1. **Include the initial model as a baseline.** Report each condition's change
   from it as well as the split-minus-unified endpoint difference. One shared
   full baseline is sufficient after verifying initial weight, tokenizer, prompt,
   decoding and corpus identity, plus a small prediction spot-check in each
   condition. This reduces full evaluation from 2,000 to 1,500 generated answers
   (100 clean + 400 corrupted at baseline and at each endpoint), plus spot-checks.
2. **Pair inference by question.** Average the four alpha-specific correctness
   values within each question, then average the 100 paired question differences.
   Compute a paired 95% bootstrap interval by resampling question IDs, carrying
   all four corruptions and all compared model outputs together. Use a fixed
   analysis seed (10002) and 10,000 CPU resamples. Report the corresponding pre/post gains
   and descriptive per-alpha scores. These intervals describe held-out-question
   uncertainty conditional on one training seed and the frozen corpus; they do
   not estimate training-seed or regenerated-corruption variability. Question-level paired intervals are implemented in the analysis script.
3. **Freeze and audit evaluation construction.** Preserve K=4 selection, all
   four alphas and the same independently generated GPT corpus for both models.
   Structurally validate all 400 rows. Before inspecting trained-model outcomes,
   semantically inspect at least 40 seeded pairs covering all four alphas, expand
   review if defects appear, and audit all rows if practical. Record checks for an actual
   mathematical error, local coherence, edit-budget compliance and answer leakage.
   Predeclare audit sampling and invalid-example handling; report any unreviewed
   portion. Do not select, regenerate or remove corruptions based on which model
   they fool. A failing construction procedure requires a documented corpus
   correction before the paired comparison, not post-hoc cherry-picking.
4. **Use accurate metric names and inspect the learning process.** The existing
   `clean_full` field is clean retention on the selected 100-question subset,
   not full GSM8K accuracy or general capability retention. Report nonempty and
   semantically useful corruptions/repairs, mixed-reward groups, clean retries,
   capped outputs, gradients and actual resource use. Keep GRPO and repair signal
   separate. Nine informative groups in the timing probe do not demonstrate
   improvement, and format tags alone do not establish useful corruption.
5. **Fix the endpoint and measure the remaining work.** Freeze the budget,
   schedule, analysis and corpus before paired training. Complete bounded batched
   preparation/evaluation and measure their costs before assigning the remaining
   GPU-hours to training. If reducing updates is necessary, reduce both conditions
   together in whole cycles before launch. Do not choose checkpoints, extend runs
   or alter settings using final held-out scores. Interpret a flat result as
   inconclusive about equivalence; an improvement confined to one run remains
   evidence about this finite-budget setup only.

Optional, not prerequisites: 50 unfiltered clean GSM8K sentinel questions would
add 150 answers across the shared baseline and two endpoints if broader clean
competence is to be discussed. A small predeclared step-95/100 probe can examine
endpoint-phase sensitivity, but the final polluter block's learning rates are
already below 0.8% of peak. Neither extension should displace the primary paired
comparison under the compute budget. These optional extensions are omitted.
The primary endpoint is after cycle seven.

Bounded preparation, batched evaluation, shared-baseline handling and question-level
intervals are now implemented and locally tested. Their full real-model pipeline
is being measured; the earlier training benchmark alone did not validate it.

## Original 100-updates-per-role proposal (historical)

The sections below preserve the original 100-updates-per-role proposal for
comparison; its measured training/checkpoint projection already exceeds ten
GPU-hours before preparation or evaluation.

## Fixed training duration

- 100 agent updates and 100 polluter updates per condition: 200 global updates.
- One cycle is five agent updates with a fixed opponent, followed by five
  polluter updates with a fixed opponent. Twenty cycles complete the run.
- Unified and split use the same budget and this repository's implementation.
  The learning-rate schedule spans the full 200 global updates, including any
  initial timed cycle; it must not restart when continuing from that cycle.

## Recommended question pool and sampling budget

Use **200 distinct retained MATH training questions**, one context per update,
G=16 samples per context, and one paired training seed (42). Keep the current
pinned DeepSeek-R1-Distill-Qwen-1.5B backbone and full tuning.

The current sampler shuffles the question pool and visits every row before
reshuffling. With 200 questions and 200 updates, each condition therefore consumes
the entire pool once: 100 contexts during agent updates and 100 during polluter
updates. This does not mean both roles each receive all 200 questions as their
active-role contexts. Both experimental conditions receive the same question
order and role assignments. The agent also participates as the fixed opponent
during polluter updates.

With only 50 questions, the same budget traverses the pool four times. Fresh
traces, corruptions and alpha choices make these different episodes, but there
are still only 50 underlying problems. A 200-question pool increases problem
diversity without increasing training update/rollout counts. It does increase
the one-time preparation work needed to find enough qualifying questions.

| Option | Contexts per update | Total context presentations | Distinct-question target | GRPO trajectories trained |
|---|---:|---:|---:|---:|
| Recommended first comparison | 1 | 200 | 200 | 3,200 |
| Larger data exposure | 2 | 400 | 400 | 6,400 |
| Larger data exposure | 4 | 800 | 800 | 12,800 |

All rows keep 100 updates per role and G=16. Contexts within an update have
independent GRPO groups and their gradients are averaged. Increasing contexts
increases generation/backward work approximately proportionally and changes the
gradient estimate; it is a separately declared setting, matched across conditions.
A pool larger than the presentation budget is possible, but some rows will not
be visited during this run.

The recommended option requires 3,200 GRPO trajectory backward passes per
condition, versus 512,000 in the original full profile: 160x fewer. It also has
up to 100 repair backward passes and at least 5,200 generated trajectories,
assuming one successful fresh clean-trace attempt per context. Additional retries,
preparation, evaluation, checkpoint I/O and startup prevent interpreting this as
a 160x wall-time speedup.

## Data and evaluation assumptions

Training count means retained questions, not raw candidates. Shuffle raw MATH
training questions with preparation seed 42 and retain the first 200 passing the
same initial-model K=4 all-correct filter. Record all candidate outcomes and the
actual retained count. Do not change the filter or choose questions based on
trained-model scores to reach the target. Freeze one shared cache for both
conditions. Training still generates fresh on-policy clean traces per context.

The current `prepare-train --candidates` flag limits raw candidates; it does not
specify the retained target or implement an early stop at 200 retained questions.
Implement and validate that bounded preparation separately before launch.

The previously user-selected evaluation target is 100 retained GSM8K questions /
400 GPT-6.1 Sol corruptions. A subsequent suggestion of 25 questions / 100
corruptions was a proposal, not a separately confirmed replacement. Retain the
100-question target in this plan and evaluate before training and at the fixed
endpoint. Keep training/evaluation questions disjoint and use the identical
frozen evaluation corpus for unified and split. Use separate development data
for diagnosing prompts or choosing settings; do not tune on the final corpus.

## What the first run can establish

This is an exploratory comparison under a smaller budget, not a paper replication.
Report clean and corrupted accuracy, their paired differences, mixed-reward group
frequency, visible pollution/repair quality, retry counts, clipping and runtime.
G=16 can produce more constant-reward groups than G=64; such groups contribute
zero GRPO advantages. A flat result may indicate insufficient training signal or
budget and does not establish equivalence between unified and split. One seed
does not estimate variation across training seeds.

The complete five-agent/five-polluter cycle and one additional agent update have
now been timed for both conditions using the full 200-update schedule. Both used
four repeated diagnostic questions and an increased 2048 input cap after a real
1024-cap overflow; neither diagnostic is a prefix of the proposed experiment.
Production training now supports opt-in HF batching on the live policies; its
default remains serial. No vLLM training integration was introduced. See the
cycle report for timing, quality checks, memory and projection limitations.

The fixed GPT-polluter training baseline remains a proposed later extension and
is not implemented. No new runtime promise is inferred from trajectory counts.
