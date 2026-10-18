# Completed alternating-cycle benchmark — 2026-10-17

Both conditions completed on Modal NVIDIA A100-SXM4-80GB GPUs, sequentially.
All benchmark GPU containers were released; the final container listing was `[]`.
No full training campaign or new GPT API calls were launched.

## Measured results

One cycle is five agent updates followed by five polluter updates. Each condition
also completed a checkpoint, restore and an additional agent update.

| Measurement | Unified | Split |
|---|---:|---:|
| Five agent updates | 7.61 min | 8.72 min |
| Five polluter updates | 8.34 min | 7.67 min |
| Full ten-update cycle | **15.95 min** | **16.39 min** |
| Checkpoint write and checksum | 41.96 s | 57.92 s |
| Checkpoint size, decimal GB | 28.43 | 42.65 |
| Restore | 33.80 s | 47.79 s |
| Additional agent update after restore | 105.14 s | 44.48 s |
| Entire diagnostic worker | 21.69 min | 21.72 min |

Worker time includes startup, CUDA tests, diagnostic hashes, tiny evaluation,
checkpoint/restore and the extra update. The two successful workers used about
0.72 GPU-hours; the preceding failed serial attempt used another 0.08. These are
observed worker durations, not a provider billing reconciliation, and are separate
from the future-run projections below.

The split cycle was 2.8% slower in this sample. One cycle with varying trace
lengths and retry counts cannot establish a stable throughput difference.

## Projection for the ten-total-GPU-hour budget

These totals sum unified and split compute; parallel execution would not reduce
them. They include one checkpoint per cycle and a small estimated adjustment for
steady split optimizer transfers. They exclude data preparation, full evaluation,
startup, interruptions and API work. The 20% allowance is a planning choice, not
a measured confidence bound.

| Cycles per condition | Updates per role per condition | Distinct questions for one pass | Paired training + checkpoints | With 20% allowance |
|---|---:|---:|---:|---:|
| **10, recommended** | **50** | **100** | **5.68 GPU-h** | **6.82 GPU-h** |
| 12 | 60 | 120 | 6.82 GPU-h | 8.19 GPU-h |
| 15 | 75 | 150 | 8.53 GPU-h | 10.23 GPU-h |
| 20, original request | 100 | 200 | 11.37 GPU-h | 13.64 GPU-h |

Recommend keeping the same backbone, full tuning, G=16, five-update role blocks,
one context/update and one paired seed, while reducing to **50 updates per role,
10 cycles and 100 distinct retained training questions**. Each condition visits
the pool once, with 50 active contexts per role and 1,600 trained GRPO trajectories
plus repair guidance. Use a fresh 100-update learning-rate/guidance schedule.
This recommendation is not yet an approved or launched full run.

Subsequent independent paper review supports this as an exploratory comparison
with additional analysis and corpus-quality controls. See the
[review-informed plan](../docs/EXPLORATORY_PLAN.md) and
[full review](../docs/EXPLORATORY_PAPER_REVIEW.md). The review does not turn these
one-cycle runtime projections into a verified end-to-end budget.

Preserve the user-selected **100 retained evaluation questions / 400 GPT-6.1 Sol
corruptions**, with initial and endpoint evaluation. Reserve roughly three
GPU-hours for shared preparation, evaluation and other overhead. That reservation
is a target, not a measurement. If the latest diagnostic's roughly 0.8 GPU-hours
must also fit inside the same ten-hour cap, only about 2.4 hours remain after the
6.82-hour training allocation.

The current preparation/evaluation path does **not** yet justify a guaranteed
ten-hour full pipeline: preparation needs a bounded retained-question target and
cross-question batching, and evaluation needs batching. Do not launch the default
7,500-candidate preparation sweep. Measure these stages and reduce the common
training budget before launch if their projected total exceeds the allocation.
GPT corpus generation belongs on CPU, with GPU workers released while waiting;
its API latency adds elapsed time but no GPU-hours. The fixed-GPT-polluter third
training condition is not implemented and is excluded from these estimates.

## Protocol and validity

- Pinned DeepSeek-R1-Distill-Qwen-1.5B, full tuning, FP32 master weights with
  BF16 autocast, inactive optimizer moments offloaded to CPU.
- Opt-in HF rollout batch size 16 for live-policy candidates and frozen-agent
  scoring. No vLLM training engine or stale inference-weight copy was introduced.
  Old log probabilities use the actual pre-update policy on unpadded samples.
- G=16, one context/update, original rewards and repair objective, 1024 output
  token caps. Batching changes RNG consumption and floating-point shapes;
  it does not promise bitwise identity with serial generation.
- **Declared departure:** both conditions used a 2048 input cap. The first
  unchanged serial attempt failed on a 1035-token repair prompt under the 1024
  cap. No silent truncation or shorter output cap was used to pass the test.
- Four previously K4-filtered MATH questions were repeated for timing, using
  fresh on-policy clean traces. These diagnostics cannot be resumed as the
  100- or 200-distinct-question experiment. The diagnostic retained the original
  full 200-update schedule rather than compressing it to eleven updates.
- Initial weights, paired configuration, source, tokenizer, data and evaluation
  hashes matched; question/alpha schedules and actual GPU model matched.
  Inactive opponent parameter hashes stayed unchanged through each block.
  Generated trajectories were not identical between conditions.
- Local suite: 100 passed, four CUDA skips. Both GPU workers: 104 passed.
  Three probe tests passed after the subsequent transfer-instrumentation fix.

Both conditions had mixed rewards in **nine of ten groups** (four of five agent
groups and all five polluter groups). Thus the benchmark exercised useful GRPO
gradients rather than only constant-reward groups. This does not demonstrate
learning or successful convergence. Split generated six empty polluter-training
windows out of 80; unified generated two. Exact pollution tags appeared in 27/80
and 41/80 respectively. Missing tags indicate format noncompliance, not by
themselves an invalid attack. Four of five agent updates in each condition had
nonempty repair targets. Full evaluation used only one saved corruption and its
clean question as a latency sample; no accuracy conclusion is justified.

## Measurement limitations

One initial cycle on four repeated questions is a weak estimate of later-cycle
runtime. Clean-trace retries and completion lengths can change with training;
the reduced plan also uses a shorter schedule than this diagnostic. Full-corpus
preparation/evaluation and a complete multi-cycle training run remain unmeasured.

The archived run measured the initial split optimizer eviction (3.09 s) and a
post-restore reload (1.35 s). Their 4.44-second sum is a component estimate of a
full exchange, not a directly observed steady exchange. Projection adjusts split
cycle time to 989.33 seconds to allow for two steady exchanges. A later probe
instrumentation fix measures the full exchange before checkpoint/restore; the
archived source ZIP preserves the version actually run. This small transfer cost
is not the dominant runtime component.

Checkpoint counters inside saved trainer state predate completion of that save;
restoring them loses the save's elapsed time. Calculations use the measured outer
checkpoint phase (42/58 seconds), not that restored counter.

Ordinary split updates peaked at about **37.10 GiB allocated / 39.80 GiB reserved**.
After checkpoint restore in the same process, the extra update reached
**62.72 GiB allocated / 64.77 GiB reserved**. The cause of this higher restore peak
has not been resolved. These results do not establish that a full resumable run
fits on a 40 GiB GPU; they also do not mean ordinary training required 63 GiB.

## Reproducible evidence

- [Computed measurements and projections](gasp-cycle-20261017-224321/summary.json)
- [Analysis script](gasp-cycle-20261017-224321/analyze.py)
- [Split report](gasp-cycle-20261017-224321/split/split/report.json)
- [Unified report](gasp-cycle-20261017-224321/unified/unified/report.json)
- [GPU teardown verification](gasp-cycle-20261017-224321/containers-after.json)
- [Protocol and implementation notes](../docs/CYCLE_BENCHMARK.md)

The artifact directory also retains exact source, manifests, per-phase timings,
training histories and console logs. Large diagnostic checkpoints were removed
after timing and verification; sizes and checksums remain.
