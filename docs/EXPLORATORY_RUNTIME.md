# Exploratory runtime projection — 2026-10-17

**Historical component projection, superseded for training by the
[completed cycle measurements](../artifacts/CYCLE_BENCHMARK.md).** The measured
HF-batched pair projects to 11.37 GPU-hours for 100 updates/role/condition including
one checkpoint/cycle, before preparation and evaluation. The earlier conditional
5–10-hour estimate below was too optimistic for that update count. For the user's
subsequent ten-total-GPU-hour constraint, recommend 50 updates/role and ten cycles:
5.68 GPU-hours raw, 6.82 with a judgmental 20% allowance, plus preparation/evaluation.
The full bounded, batched preparation/evaluation path is not yet timed or complete.

The original text below is a planning projection from saved component measurements,
not a timed end-to-end run. No GPU or API work was launched to produce that original
projection. Its statements about missing integrated HF timings are historical.

## Scope and assumptions

- Unified and split, one seed each, run sequentially on Modal A100-80GB.
- Each condition: 100 updates per role, 20 cycles, G=16, accumulation=1.
- Shared pool of 200 retained MATH training questions; 100 retained GSM8K
  evaluation questions and 400 GPT-6.1 Sol corruptions.
- Evaluation before and after each condition: four passes of 500 answers.
  Set eval_every=200; inheriting the existing value of 100 would add two passes.
- Same backbone, token limits, full tuning and optimizer-state offload.
- Preparation stops at the retained target. That early-stop feature still needs
  implementation; the current default scans 7,500 raw training candidates.
- No third, fixed-GPT-polluter training condition; it is not implemented.
- No failures, preemption, protocol changes or manual review time included.
  Engineering time is separate. GPU jobs must not wait idle for API generation.

## Current serial HF path

The saved Colab A100 serial diagnostic generated 2,048 tokens in 79.284 seconds:
25.83 tokens/second. Modal's batched HF test was approximately 1.95 times faster
than the corresponding Colab batched test (24.099 versus 46.895 seconds for the
same 10,179 generated tokens). Use **25–50 serial tokens/second** as an explicitly
heuristic planning range; the upper endpoint is not measured Modal serial speed.

The Modal A100 initial-policy sample projects approximately 2.63 million generated
tokens per condition at the smaller training budget, including observed fresh
clean retries. Calculation: 1,600 times each of the agent, polluter and scoring
mean lengths, plus 100 single corruptions and repairs, plus 200 fresh contexts.
This gives **14.6–29.2 generation hours per condition**. Backward/old-log-probability
work scales to roughly ten minutes per condition from the fixed fixture, before
extra optimizer steps, transfers, verification, snapshots and checkpoint I/O.
Budget **15–32 hours per condition** as a rough range.

The A100 MATH filter retained five of 64 candidates. At that yield, finding 200
retained questions requires about 2,560 candidates, four traces each. The measured
mean was 242,225/256 = 946 tokens/trace: approximately **9.69 million tokens**,
or **54–108 hours** of serial generation. Use **55–115 hours** for this stage.
This yield comes from a tiny sample; it is not a guarantee of sufficient questions.

Existing batched HF evaluation preparation retained 32 questions in 2,628 seconds
on Colab. Allow **1–3 hours** for 100 retained questions plus fresh traces/startup
on Modal, with substantial yield uncertainty. Allow **0.5–2 hours** for 400 serial
GPT calls: the only saved API latency observation was six seconds, giving a
40-minute raw extrapolation. This API estimate has very weak evidence.

For evaluation, assume 300–600 output tokens per answer, not a measured full-corpus
average. Four passes at 500 answers each take **3.3–13.3 hours** at the serial rate;
budget **4–14 hours**. Reserve **1–4 additional hours** for loading, large checkpoints,
and result handling; this is an unmeasured allowance.

Together these assumptions give approximately **90–200 hours**, or **4–8 days**,
for the sequential pair from an unprepared corpus. Changing filter yield, output
lengths, checkpoint frequency or decoding speed can move the result outside this
range. It is a warning about the serial implementation, not a GPU performance limit.

## Conditional batched-generation target

With validated batching for preparation, training rollouts and evaluation, a
reasonable exploratory planning target is **5–10 elapsed hours** for the same
sequential pair, excluding engineering. This is lower-confidence than the serial
token calculation and must not be advertised as an achieved runtime.

The A100 vLLM probe filtered 64 MATH candidates (256 traces) in 21.123 seconds:
40 such batches would be roughly 14 minutes of generation at the observed yield.
For training, retaining the measured large-batch stage latencies as conservative
proxies for smaller groups, rather than dividing them linearly by 16, gives roughly
1.5–2 hours per condition before synchronization, I/O and other overhead. Budget
2–3.5 hours per condition, plus shared preparation/API work and evaluation.

The probe ran inference and backward in separate processes. Live weight updates,
frozen opponents, exact old-policy log probabilities, memory coexistence and full
resume correctness remain unvalidated. HF batching is another possible route,
but no integrated timing exists for it either. Neither the tiny smoke tests nor
these component probes establish full-cycle throughput. A real five-agent /
five-polluter cycle, including return to the first role, should replace these
projections before committing to the remainder.

## Evidence

- `artifacts/generation-throughput-20261017.json`
- `artifacts/gasp-traces-20261017-043237/contexts.log`
- `artifacts/gasp-modal-20261017-061436/A100-80GB/results/hf.json`
- `artifacts/gasp-modal-20261017-061436/A100-80GB/results/vllm-generated.json`
- `artifacts/gasp-modal-20261017-061436/A100-80GB/results/backward-unified.json`
- `artifacts/gasp-modal-20261017-061436/A100-80GB/results/backward-split.json`
- `data/full-v3/held-out-corruptions.jsonl.responses/000002.json`
- `src/gasp/models.py`, `src/gasp/data.py`, `src/gasp/evaluation.py`
