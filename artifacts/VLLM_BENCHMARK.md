# A100 vLLM benchmark and runtime estimate — 2026-10-17

The bounded benchmark completed successfully. Its A100 was released; a subsequent
Colab server query reported no active sessions. No production corpus was generated,
no full experiment was launched, and no new OpenAI API calls were made.

Evidence: [worker report](gasp-vllm-20261017-053328/results/worker.json),
[inference measurements](gasp-vllm-20261017-053328/results/vllm.json),
[training component measurements](gasp-vllm-20261017-053328/results/backward.json),
[calculated estimates](gasp-vllm-20261017-053328/runtime-estimate.json).
The source archive, executed worker, package freeze, logs and full result bundle
are preserved in the same directory. The first attempt failed during dependency
resolution because its default Python was 3.13; it was also released automatically.
The successful run explicitly selected Python 3.12.

## Measured results

Hardware: one NVIDIA A100-SXM4-40GB. Pinned DeepSeek-R1-Distill-Qwen-1.5B revision
`ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`; vLLM 0.10.1.1, PyTorch 2.7.1,
Transformers 4.57.6. vLLM uses BF16 inference, prefix caching, CUDA graphs, and at
most 128 concurrent sequences. Startup took 81.75 seconds.

| Work | Measured time / rate |
|---|---:|
| HF: four solutions on each of four questions | 46.89 seconds; 10,179 output tokens |
| vLLM: same questions and sample counts | 7.93 seconds; 9,874 output tokens |
| Wall-time improvement on that workload | 5.92x |
| GSM8K filter: 64 questions, K=4 | 18.32 seconds generation + 2.07 seconds verification; 7,911 output tokens/s |
| MATH filter: 64 questions, K=4 | 24.61 seconds generation + 0.38 seconds verification; 9,963 output tokens/s |
| Agent-shaped generation, four contexts/G=64 | 29.29 seconds, including fresh traces, pollution and repair |
| Agent log probabilities + backward + optimizer | 70.52 seconds |
| Polluter-shaped generation, four contexts/G=64 | 49.44 seconds, including fresh traces and agent scoring |
| Polluter log probabilities + backward + optimizer | 73.06 seconds |

Both role timings contain 256 active-policy trajectories. Agent trajectories
averaged 193 tokens and polluter trajectories 817 tokens. The component probe
performed actual clipped-GRPO backwards, visible-repair backwards, clipping and
AdamW updates on disposable models. Both gradient norms were finite and nonzero.
It did not advance a complete self-play training loop.

The existing suite passed **94 tests** in the GPU environment. The four 64-token
greedy checks matched HF exactly. Diagnostic assertions checked tokenized prompt
identity, grouping/sample counts, EOS position, nonempty outputs, finish reasons,
and configured generation caps. This is a bounded smoke check, not proof of
distributional or full-sequence equality between inference engines. Stochastic
samples and BF16 inference can differ from HF FP32 masters/BF16 autocast.

The inference process printed NCCL/engine teardown warnings after saving its
successful report. It returned zero, released its GPU allocations before the
backward process ran, and the entire runtime was subsequently terminated.

## Extrapolation and assumptions

The main evaluation target is **100 retained questions / 400 corruptions**.
The small vLLM filter retained 5/64 questions in each dataset. At that observed
GSM8K yield, selecting 100 questions requires about 1,280 candidates. The sample
is too small to guarantee that the 1,319-question source yields 100. Do not loosen
the K=4 selection rule if fewer qualify.

Measured component extrapolations:

- Evaluation clean-context preparation: **0.22 hours** including engine startup,
  using the observed filter yield and batched fresh-trace retries.
- MATH training selection: **0.84 hours** for 7,500 candidates.
- 400 serial GPT corruptions: **0.67 hours** at the one historical response's
  six-second server duration. This is a weak latency estimate, not a new API
  benchmark; network time, rate limits and failures are not measured here.
- Agent optimizer update: **99.80 seconds** including generation components.
- Polluter optimizer update: **122.50 seconds** including generation components.
- 1,000 updates of each role: **61.75 hours** before overhead. Applying the prior
  measured full optimizer exchange of 12.79 seconds once per five steps adds
  about 1.42 hours in split: **63.17 hours**. Unified snapshot refresh remains
  additional overhead.
- 21 scheduled evaluations of 500 answers each: a simple shape-based estimate
  is 0.20 hours/run. This used repeated MATH-shaped prompts, not the final held-out
  GPT corpus; treat it as a rough component estimate.

The following **planning ranges are judgmental allowances, not confidence
intervals**. They allow for longer/different contexts, clean retries, checkpoint
I/O, inference/training synchronization, provisioning and restarts. The benchmark
does not measure those costs sufficiently to guarantee an upper bound.

| Scope, sequentially on one A100 | Estimated hours |
|---|---:|
| Shared training/evaluation preparation, including GPT calls | 2–4 |
| One full unified or split run | 80–130 |
| Unified + split, one seed, including preparation | 162–264 |
| Unified + split, three seeds, including preparation | 482–784 |
| Additional fixed GPT-polluter run, one seed | 40–70 |
| All three conditions, one seed, including preparation | 202–334 |
| All three conditions, three seeds, including preparation | 602–994 |

For the third baseline, assume 1,000 agent updates, G=64, four contexts/update,
one GPT pollution per context (4,000 training API calls), and self-generated repair.
Its raw component projection is 32.21 hours, replacing local polluter generation
with four six-second API calls per update and removing polluter optimizer updates.
This baseline is not implemented, and its API latency and concurrency are not
validated. Training API calls would add to paid GPU wall time if awaited inline.

## What these estimates do not establish

Production training still uses serial Hugging Face generation. **The training
figures above assume a future validated vLLM integration**; the current trainer
will not automatically achieve them. The inference engine and trainer were run
in separate processes. Their simultaneous memory use and synchronization of
updated policy weights were not tested. The current trainer's 512,000 separate
GRPO backwards per full run are retained in the projection.

Only four initial-checkpoint MATH contexts at alpha=0 were used for training
components. Later policies, other alpha values and prompt overflow can change
runtime or stop training. The existing fail-fast 1024-token prompt behavior remains
unresolved for some long contexts. Checkpoint serialization (200 checkpoints/run),
hashing, uploading, exact vLLM resume and live-policy correctness still need an
integrated pilot. Current gradient checks do not validate that integration.

Human corpus review and implementation/debugging time are excluded from the hour
totals. The 1,000-updates-per-role and four-context accumulation settings are
documented local assumptions, not explicit author-specified budgets. See
[runtime plan](../docs/RUNTIME_PLAN.md) for the generation inventory and scope.
