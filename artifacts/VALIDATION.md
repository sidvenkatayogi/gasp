# Generation throughput and batched preparation — 2026-10-17 UTC

The serial preparation run was deliberately interrupted and its GPU released
after saving a checkpoint through 32 questions (four retained). Its downloaded
results and source archive are preserved; it is not a completed corpus.

On the same A100, an equal-work benchmark (16 sequences x 128 generated tokens)
measured 25.83 tokens/s at batch 1, 103.99 at batch 4, and 408.08 at batch 16.
The 128-token minimum/maximum was only a diagnostic workload. A separate real
K=4 test with EOS and the configured 1024-token maximum took 11.01s for generation
and 0.29s for verification, with four correct answers. This establishes a
generation bottleneck, not an end-to-end 16x speedup guarantee. The corpus worker
was temporarily SIGSTOPed during measurement and resumed in the probe's finally
block; it was subsequently interrupted cleanly to switch implementations.

Evidence: [benchmark result](generation-throughput-20261017.json),
[executed probe](probe_generation_throughput.py). The probe field named
`total_allocated_bytes_including_paused_job` actually reports PyTorch allocation
for the benchmark process only; do not use it as total GPU memory. A separate
nvidia-smi snapshot showed 28% utilization for serial generation.

The new evaluation preparation batches four questions, with four samples each.
Tests cover mixed-length greedy generation versus individual prompts, grouping,
sampling settings, EOS/padding handling, and exact checkpoint resume after batch
interruption. **91 local tests passed, three CUDA cases skipped; all 94 passed
on the A100 in 38.12s** ([CUDA log](gasp-traces-20261017-043237/tests.log)). The training
generation and optimizer paths are unchanged. New preparation uses `data/full-v4`;
serial-v3 contexts and the one GPT preflight response remain historical artifacts
and are not silently reused with newly sampled traces. The new runtime
`gasp-traces-20261017-043237` has started actual batched context preparation;
its first two batches processed eight questions in 222.55s (76.96s and 145.59s),
retaining two questions. The first completed batch was also verified in the local
backup. This is about 27.8s per question over a very small sample, versus the prior
roughly 1–2 minutes per question. Fresh correct-trace rejection sampling still
runs individually, so the fixed-length generation benchmark's 16x throughput
gain is not the end-to-end speedup. A reliable full-set ETA needs more batches.

---

# Optimizer offload and trace preparation — 2026-10-17 UTC

**86 tests passed in 37.09s on an A100-SXM4-40GB**, including the two CUDA/bf16
offload equivalence cases and the answer-unit parsing regressions. The local run
passed 84 tests, with the two CUDA cases skipped.
The offloaded and resident paths produce bit-identical model/snapshot weights,
AdamW states and RNG state through role switches and checkpoint resume in the
tiny-model GRPO tests. Within-block placement preserves moment storage pointers.
Both full and pilot profiles now enable inactive-state offload. Active moments
remain on GPU across the five-update block; model weights are not offloaded and
no dtype, optimizer, reward, schedule or loss calculation was changed.

The full 1,777,088,000-parameter model also completed three synthetic split updates
(agent, polluter, agent), using 1024 prompt and 1024 completion tokens and two
accumulated backward passes per update. The second pass exercises the peak with
gradients already present. Inactive model hashes stayed unchanged. Peak allocation
was **39,685,512,704 bytes (36.96 GiB)**, with **41,762,684,928 bytes (38.89 GiB)**
reserved. The first eviction took 9.20s; the first complete exchange of the two
existing moment sets took **12.79s**, or approximately **2.56s per update amortized
over five-update blocks**.
These are observed pageable-memory transfers on this Colab host, slower than the
earlier bandwidth-based estimate. Further switches can vary. This is not a
complete G=64, four-context training update or a scientific experiment result.
Full pilot generation, end-to-end memory peaks and training throughput remain to
be checked. The earlier single-backward probe in `gasp-traces-20261017-025753/`
had a lower 34.86 GiB allocation peak and is superseded by this accumulation probe.

Evidence: [CUDA tests](gasp-traces-20261017-031507/tests.log),
[full-model memory probe](gasp-traces-20261017-031507/memory-probe.json),
[source manifest](gasp-traces-20261017-031507/source-manifest.json).

All 7,500 MATH train answers and 1,319 GSM8K test answers were validated locally
before provisioning. Two MATH solutions use valid unbraced single-token TeX
answers (`\\boxed 2`, `\\boxed 9`); the extractor now accepts these and tests
still reject incomplete/multi-token unbraced answers. No dataset rows were dropped.
The pinned DeepSeek revision is `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.

Context preparation now checkpoints each completed question, including RNG state.
Tests verify uninterrupted versus resumed context bytes, audit, counts and RNG
equality, preserve interrupted audit tails, and reject mismatched resume inputs.
The Colab owner downloads results and stops in a finally block. The remote worker
also has a hard lifetime limit and a five-minute result-download grace period
before self-unassignment. Completed-question backups are periodically downloaded.
The initial context job was stopped and its GPU released after two questions:
inspection found that `<answer>3 bolts</answer>` was incorrectly interpreted as
a symbolic product by the wrapper around Math-Verify. Outputs and the checkpoint
are preserved in `gasp-traces-20261017-025753/interrupted-contexts.*`. The fix uses
Math-Verify's plain-expression parsing for numeric answers with unit words while
retaining symbolic-math parsing; regressions cover both kinds of answers and wrong
values. Corrected trace preparation on `gasp-traces-20261017-031507` logged 13/1,319
questions processed and one question retained, then its local controller treated
a 180-second log-download timeout as fatal and released the GPU. There was no
completed result archive. The latest intact local backup covers eight questions;
its config, data and source hashes were verified before recovery. The other five
logged questions must be recomputed from the saved RNG state.

The controller now treats best-effort monitoring timeouts as nonfatal, uses
30-second monitoring timeouts, backs up approximately once per minute and tries
a final checkpoint download on failure. All five controller tests pass. A new
A100 session, `gasp-traces-20261017-034910`, was provisioned at 03:49 UTC with the
recovered checkpoint. It will skip the already-completed tests/memory probe,
enforce the existing resume identity checks, and continue context preparation.
The coordinator will download completed contexts, stop the GPU, then generate
GPT-6.1 Sol corruptions locally. No training campaign has been started yet.
Observed sequential preparation throughput was roughly 1–2 minutes
per question; a full-set estimate of about 30 hours is preliminary.

At 04:13 UTC the resumed job had processed 20 questions, retained two, and saved
a valid backup through question 20. At 04:16 UTC its log had reached 22 questions.
Several further monitoring timeouts were tolerated without terminating the job.

A live GPT-6.1 Sol preflight used committed context index 2 from the question-20
checkpoint with the exact final generator payload. It completed with 767 input
tokens and 154 output tokens (110 reasoning tokens). The raw response is saved
in the final response directory as `000002.json`, for exact request/context-ID
validation and reuse by final generation. The final corpus has not been published.
Manual inspection of this response confirmed a local factual corruption: it
claims three bolts of white fiber where half of two implies one. The actual
DeepSeek tokenizer gives 110 clean-prompt tokens and 292 corrupt-prompt tokens,
both within the unchanged 1024-token cap. This checks one response only and does
not establish quality for the whole corpus. See the [preflight report](gpt-preflight-20261017.json)
and [executed request script](gpt-preflight-20261017.py). The API key remained local.

---

# Live Colab CPU smoke — 2026-10-16

**62 tests passed in 26.72s** on an actual Colab CPU runtime, followed by successful
unified and split smoke runs: eight optimizer updates each, four per role.
The test suite includes exact checkpoint resume and inactive-model isolation.

Environment: Linux, Python 3.13.16, PyTorch 2.7.1+cpu, Transformers 4.57.6,
NumPy 2.2.6, PEFT 0.18.0 and Math-Verify 0.8.0, in an isolated virtual environment.
Setup, tests and smoke took **106.33s** for the successful runner invocation;
the smoke CLI itself took **8.23s** including startup and paired aggregation.

Results: [run report](colab-smoke-20261016/report.json),
[test log](colab-smoke-20261016/tests.log),
[paired summary](colab-smoke-20261016/run/comparison/summary.json).
The downloaded archive, source manifest, exact runner, package versions, training
logs and checkpoints are retained in `colab-smoke-20261016/`. Archive integrity,
source hash, all test outcomes, final role counts and checkpoint hashes were checked
locally before terminating the runtime. Colab confirmed termination.

Two initial setup attempts exposed runner issues: uploaded paths are relative to
the filesystem root, and Colab's system Python omits working ensurepip support.
`scripts/colab_smoke.py` now uses the correct upload path and bootstraps the isolated
environment with system pip's `--python` option. No training source changes were
needed. Colab CLI 0.6.0 can return exit zero after remote Python exceptions, so
success was established from the saved report and artifacts, not its exit code.

This validates the synthetic tiny-model pipeline on Colab CPU. It does not validate
1.5B GPU memory fit, real-data behavior, corruption quality or scientific performance.
No OpenAI API calls or GPU runtime were used.

---

# Corpus generation validation — 2026-10-16

**62 tests passed in 11.78s** with `.venv/bin/python -m pytest -q`, including 12
new corpus tests. The environment remains CPU-only. API requests in tests were
mocked; no real GPT-6.1 Sol call or corpus generation was performed.

- Clean-context preparation checks all four filter samples, obtains a fresh correct
  trace, shares it across four alpha cuts, records rejected questions/attempts,
  and refuses training-split or unresolved-checkpoint inputs.
- Generation uses `gpt-6.1-sol` and the Appendix B polluter prompt without the gold
  answer. Saved responses resume without repeated requests; changed settings and
  invalid/incomplete/unchanged/wrong-model responses fail explicitly.
- Generated Sol corpora import through the existing evaluation path. Their source
  hash is verified; synthetic smoke contexts cannot trigger API generation.
- All five regenerated notebooks pass nbformat validation and code-cell compilation.
  Corpus generation is disabled by default in notebook 01. CLI help exposes the
  new preparation/generation commands and supported Sol reasoning efforts.
- The wheel and Colab archive were rebuilt, with no additional dependency.

The paper's construction details and our explicit choices are in
[CORPUS.md](../docs/CORPUS.md). Actual generation requires prepared GSM8K/initial
model traces, a CUDA environment for the configured model, and `OPENAI_API_KEY`.
Neither live API behavior, semantic corruption quality, nor target-model GPU
training has been verified by these offline tests.

---

# Independent audit validation before corpus generation — 2026-10-16

**50 tests passed in 12.91s** with `.venv/bin/python -m pytest -q`.
CPU environment: Python 3.13.5, PyTorch 2.7.1, Transformers 4.57.6.
See [the audit](../docs/AUDIT.md) for findings and remaining fidelity limits.

Additional checks completed:

- Regression coverage for chat-template-opened unfinished reasoning, exclusion of
  privileged scratchpads from guidance, verbatim word/number boundaries, and prompt
  overflow rejection. Existing isolation/resume/reward tests remain green.
- Ordinary autoregressive `model.generate` and unrestricted log-probability losses
  run on a tiny Qwen2 model, alongside the synthetic binary smoke path.
- Six checks using the actual DeepSeek-R1-Distill-Qwen-1.5B tokenizer passed;
  tokenizer hashes and check descriptions are in `audit-tokenizer-checks.json`.
  No 1.5B weights were downloaded or trained.
- Rebuilt and locally installed the wheel. Its CLI completed both unified and split
  smoke runs, eight optimizer updates each, including evaluation, checkpointing
  and paired aggregation. Outputs: `runs/audit-v3/` (ignored local run artifacts).
- Regenerated all five notebooks with fresh v3 data/run paths. nbformat validation
  and compilation of every code cell passed. No live Colab execution was performed.
- Rebuilt the Colab archive, including the audit and tokenizer check record. Every
  archived file matches its manifest hash; the installed wheel matches every
  current source module.

The 1024-token limit is unchanged, but oversized role prompts now stop explicitly
instead of altering shared context. Real-data overflow frequency, 1.5B CUDA/bf16
memory fit, mathematical quality of repairs and scientific performance remain
unverified. Previous cache/checkpoint fingerprints do not match this source.

---

# Historical validation before the independent audit — 2026-10-16

**40 tests passed in 10.98s** in the executed offline notebook. Environment:
macOS ARM64, Python 3.13.5, PyTorch 2.7.1, Transformers 4.57.6, PEFT 0.18.0,
Math-Verify 0.8.0; CPU only. The package is version 0.2.0.

## Checked

- Real tiny-Qwen2 gradients and AdamW updates in unified and split conditions.
- Independent split parameters/moments, unchanged inactive parameters, one moment
  set for shared weights, common global LR schedule, and zero split snapshot bytes.
- Exact checkpoint resume inside and between blocks, including optimizer, scheduler,
  RNG, role counts, opponent state and data order. Old checkpoint schemas are rejected.
- LoRA adapter isolation and checkpointing on tiny Qwen2, retained as optional support;
  the shipped 1.5B full/pilot profiles both use full tuning.
- Full-group GRPO and repair guidance remain active for unformatted corruptions.
  No semantic, sentence-count, boxed-answer or extra length validity masks remain.
- Fresh clean-trace rejection sampling, early success, and exhaustion without an
  optimizer/scheduler update. Cached traces cannot be substituted.
- Actual non-smoke prompt construction preserves prefix/window boundaries, tested
  with a lossless text tokenizer; visible wrappers and empty targets are handled.
- Actual Math-Verify integration: fraction/decimal equivalence, symbolic algebra,
  wrong answers and unfinished outputs. This dependency is now installed locally.
- External held-out GPT-5 import preserves windows, rejects missing corruptions and
  substitute generators, and blocks synthetic evaluation for real-model runs.
- Data filtering, cache tampering/overlap, paired aggregation and benchmark adapters.
- All five regenerated notebooks pass nbformat validation; every code cell compiles.
- Notebook 00's offline cells completed the full test suite, both smoke conditions,
  and mid-block checkpoint resume. Setup/network cells were skipped. This is code-cell
  execution, not a running Jupyter kernel or live Colab session. Outputs are in
  `00_plan_and_smoke.executed.ipynb`.
- The rebuilt Colab archive was extracted into a fresh temporary directory; its
  own source completed both smoke conditions, with zero split snapshot bytes.
- A normal wheel was built and installed locally with no new package dependency
  beyond the already-declared Math-Verify requirement.

## Measured offline smoke

Run: `artifacts/notebook-smoke-vtyxec55/`. Both conditions completed four updates per
role, with 32 agent GRPO tokens, 32 polluter GRPO tokens and three guidance tokens.
Each generated 121 training-related tokens; this agreement is incidental, not an
assumption enforced for clean retries.

| Quantity | Unified | Split |
|---|---:|---:|
| Live role parameter bytes | 45,696 | 91,392 |
| Optimizer bytes | 91,452 | 182,904 |
| Opponent snapshot bytes | 45,696 | 0 |
| Training wall seconds | 0.444 | 0.417 |

These tiny CPU timings and synthetic accuracies do not support a scientific conclusion.

## Still requires external execution/input

- Real 1.5B model, CUDA/bf16 training and GPU memory measurement.
- Actual MATH/GSM8K preparation and an externally supplied held-out GPT-5 corpus.
  Import fixtures validate mechanics, not mathematical validity of unseen corpora.
- Full multi-seed campaign, live Colab run and benchmark results.
- Author code/artifacts to resolve the unspecified settings in `docs/EXPERIMENT.md`.

No paid GPU or API work was launched. Pre-fix caches/checkpoints are incompatible
with the revised protocol; use fresh v2 output directories.
