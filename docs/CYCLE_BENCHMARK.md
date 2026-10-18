# Bounded live-cycle benchmark

Purpose: replace component-only estimates with real alternating-update timings
before choosing an experiment that fits approximately ten total GPU-hours,
prioritizing lower compute cost. CPU/API waiting time is separate.

Both conditions have completed; see [results and budget recommendation](../artifacts/CYCLE_BENCHMARK.md).

The benchmark runs on one Modal A100-80GB at a time, split then unified. Each job
has a two-hour hard timeout, no retries, no warm containers, and automatic release
on completion. Model weights are already in the CPU-built image. No OpenAI calls
or new full-corpus generation are part of this benchmark.

## Protocol

- Existing pinned DeepSeek-R1-Distill-Qwen-1.5B, full tuning, FP32 master weights
  with BF16 autocast, opt-in HF rollout batches of up to 16 and inactive
  optimizer-state offload. Both candidate sampling and frozen-agent scoring batch.
- Five agent updates, five polluter updates, then one additional agent update to
  measure returning to a role with existing optimizer state.
- G=16, one context/update. Existing prompt text, 1024-token output caps, reward
  rules, alpha sampling, fresh clean-trace rejection sampling and repair remain.
- **Exploratory deviation:** raise the input cap from 1024 to 2048 in both
  conditions after the original baseline failed on a 1035-token repair prompt.
  This preserves full shared context and does not silently truncate any role.
  This setting is an implementation choice for this exploratory test, not a
  paper replication setting or an explicitly user-selected numeric cap.
- Use the complete 200-update LR/guidance schedule (100 updates/role); stopping
  after 11 updates does not compress or restart that schedule.
- Both conditions use the same four previously K4-filtered MATH questions from
  the saved A100 vLLM probe. Repetition is a timing diagnostic. This is not a
  resumable prefix of the proposed 200-distinct-question experiment.
- Time initial and final evaluation using one existing paid GPT corruption and
  its clean question. This is a latency sample, not a meaningful accuracy score.
  The saved API request is checked against its corresponding saved context.
- At update ten, write and hash an actual production checkpoint, then restore it
  before update eleven. Checkpoint contents are diagnostic and not downloaded;
  timing, size and checksum metadata are retained. GPU teardown removes the
  temporary filesystem, including any partial checkpoint on failure.
- Time the normal return of optimizer states **before** checkpoint/resume.
  Resume moves both moment sets to CPU, so a transfer measured only afterward
  includes loading but does not include eviction of the outgoing role's states.
- Compare opponent parameter hashes before/after each block to check that the
  inactive policy remains frozen. The diagnostic hashing time is reported
  separately from training.

## Measurement and limitations

`scripts/probe_cycle.py` wraps existing methods to time generation by purpose,
GRPO backward, optimizer steps, role transfers, snapshots, checkpoint/resume and
whole updates. Wrappers add synchronization and logging without changing RNG,
samples or losses. Opt-in batching does change RNG consumption relative to serial
sampling; it preserves the sampling parameters, live model identity and rewards,
not bitwise trajectories. Old log probabilities are computed on each unpadded
sample from the same live model before its update. Backward remains per sample.
Generation timings include old-log-probability calculation
where the production generator requests it. Nested phase times must not be
summed with their enclosing update time.

All phase completions stream into the local console log. Failures retain available
logs and telemetry and stop the paired launch. No prompt truncation, shortened
completion cap, cached clean-trace substitution or selection of easy alpha values
is used to force a successful run.

The small pool, one initial cycle, one evaluation example and initial-policy
output lengths limit extrapolation. Counts of informative reward groups, clean
retries, malformed pollutions and empty repairs must be considered alongside
speed when choosing the subsequent experiment.

Local validation before the first launch: 98 passed, three CUDA skips. The
remote worker repeats the suite on CUDA before training. The instrumentation test
checks that a real tiny-model update has identical contexts and resulting weights
with and without the wrappers.

Command: `PYTHONPATH=src .venv/bin/python scripts/run_modal_cycle.py`

## First baseline attempt

`artifacts/gasp-cycle-20261017-223457` used the unchanged serial implementation
and 1024 input cap. CUDA tests passed 101/101. The first split update took 45.26 s;
all 16 agent completions were immediate EOS and all rewards zero, so only repair
guidance contributed a gradient. Update two generated its candidate and agent
completions but failed before backward because the repair prompt was 1035 tokens.
The worker returned diagnostics after 278.16 s; the paired launcher stopped before
unified. A subsequent Modal container listing returned an empty list.

The first worker was A100-SXM4-80GB (400 W), unlike the prior Modal component
benchmark's A100 PCIe (300 W). Record actual hardware for each attempt.

## Batched run instrumentation caveat

The archived run `artifacts/gasp-cycle-20261017-224321` predates the explicit
`steady_return_optimizer_transfer` phase. It measured the first split eviction
and a post-resume reload separately. Their sum provides a component estimate of
a steady full exchange, not a directly measured full exchange. Use the outer
phase events for checkpoint cost: the production checkpoint stores counters
before its own save finishes, and resume restores that older checkpoint counter.
The archived source ZIP preserves exactly the code used for the measurements.

The live-cycle batch change passed 100 local tests with four CUDA skips; the
GPU workers passed all 104 tests. No training objective, reward, optimizer or
output-token budget changed. The 2048 input cap remains a declared exploratory
protocol change.
