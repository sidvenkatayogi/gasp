# Modal comparison — 2026-10-17

Both bounded benchmarks completed successfully; all Modal apps were subsequently
confirmed stopped with zero tasks. Authentication is configured. No production
corpus, GPT calls, or full training campaign was started.

**A100-80GB is the better-value choice for the current implementation.** H100 is
faster, especially for inference, but the measured training speedup does not cover
its higher configured hourly rate. This is one diagnostic run per GPU, not a
statistically precise hardware ranking.

## Measurements

| Component | A100 80GB PCIe | H100 80GB SXM |
|---|---:|---:|
| GPU + requested 4 CPU cores / 64 GiB RAM, $/hour | 3.20 | 4.65 |
| Fixed inference, 128 × 512 output tokens | 12,901 tokens/s | 19,776 tokens/s |
| Unified agent update compute | 38.30 s | 37.78 s |
| Unified polluter update compute | 57.40 s | 38.74 s |
| Split agent update compute | 38.57 s | 39.98 s |
| Split polluter update compute | 57.51 s | 44.12 s |
| Complete worker duration, including tests/startup | 8.16 min | 6.74 min |

Update compute includes old-policy log probabilities, GRPO backward, repair
backward when applicable, clipping and AdamW. It excludes generation, model
loading, snapshot refresh and optimizer transfers. Each role processes the same
256 saved trajectories on both GPUs: 49,515 agent tokens and 209,189 polluter
tokens across four contexts. Unified shares parameters/moments and creates its
frozen opponent snapshot; split retains separate policies and optimizer moments.
Inactive split optimizer moments still move to CPU.

H100 improves fixed large-batch inference by 1.53x, unified update compute by
1.25x, and split update compute by 1.14x. It costs 1.45x as much at the requested
resource levels. The first split eviction took 4.63 s on A100 and 4.77 s on H100;
these are not full steady-state exchanges because the second role had no moments
yet. Unified snapshot copies took 0.04–0.07 s here.

Both GPUs passed **98 tests**, including CUDA optimizer-offload equivalence and
resume tests. Local validation passed 95 tests with three CUDA skips. The four
64-token HF/vLLM greedy comparisons matched 4/4 on A100 and 3/4 on H100; the H100
mismatch starts at token 32. These short comparisons do not establish identical
policies or sample distributions across inference engines or hardware. Training
gradient norms were finite; they also differ slightly across hardware.

Native inference selected four MATH contexts on A100 and three on H100, with
different output lengths. Do not compare their native generation totals or whole
worker times as equal workloads. Backward measurements replay an identical
SHA-256-checked four-context fixture; fixed-length generation uses equal workloads.

## Provisional training estimates

These estimates **require a completed, validated vLLM training integration**.
Production training still uses HF generation. The benchmark ran inference and
training in separate processes and did not validate live-policy synchronization,
co-resident memory, full self-play, or complete checkpoint/resume on Modal.

Budget: 1,000 updates per role, four contexts/update, G=64, one seed, full tuning.
The two update-budget settings are existing local assumptions rather than explicit
paper specifications. The evaluation target remains 100 questions/400 corruptions.

| One full condition/seed | A100 80GB | H100 |
|---|---:|---:|
| Unified planning hours | 55–95 | 40–70 |
| Split planning hours | 55–95 | 40–75 |
| Unified compute cost at requested resources | $176–304 | $186–325 |
| Split compute cost at requested resources | $176–304 | $186–349 |

Method: scale the original Colab generation workload using each GPU's measured
fixed-length small/large-batch speed ratios, then add measured update compute.
This gives raw totals of 43.5/43.6 hours on A100 and 32.5/34.6 hours on H100 for
unified/split. Apply a judgmental 1.3–2.1 allowance, rounded outward to five hours.
This is a heuristic projection, not a measured full-run duration or confidence
interval. Context lengths, retries, transfer/synchronization, checkpoint I/O and
preemption remain uncertain; long-context prompt overflow is still unresolved.
Shared corpus/data preparation, GPT fees and engineering time are additional.
The fixed GPT-polluter training baseline remains unimplemented and is not measured.

Two independent jobs could overlap unified and split, reducing calendar time to
roughly one run's duration while paying for both. The current trainer uses one
device; simply allocating two GPUs to one job will not parallelize it. Correct
batched backward computation is a more promising next optimization to test than
paying for another GPU tier: the current loop still handles 256 trajectories
separately per update. This is an implementation-based inference, not a measured
speedup. H200 remains an unmeasured option if additional memory proves useful.

Modal functions have a 24-hour maximum execution timeout and GPU functions can be
preempted, so full runs need durable checkpoints and bounded resumable chunks.

## Cost, setup and reproducibility

Successful worker execution corresponds to approximately **$0.96 combined** at
requested resource rates, before credits. This excludes loading outside the
worker timer, CPU image builds, failed setup attempts and any CPU/RAM use above
the requests; it is not an invoice. Public list prices were checked on 2026-10-17.

Setup found and fixed a missing image-build helper import and an upload-filter
mistake that copied partial local `.egg-info` metadata. The latter caused package
logging/test setup failures; generated metadata is now excluded with a regression
test. Failed GPU workers returned after about one second and 26 seconds. Two
additional metadata diagnostics used CPU only. Production `src/gasp` is unchanged,
preserving the previous source identity.

The inference processes printed NCCL/worker teardown warnings after writing
successful results, returned zero, and released memory before backward began.

Evidence: [comparison JSON](gasp-modal-20261017-061436/comparison.json),
[A100 worker](gasp-modal-20261017-061436/A100-80GB/results/worker.json),
[H100 worker](gasp-modal-20261017-061436/H100/results/worker.json),
[source hashes](gasp-modal-20261017-061436/source-sha256.json),
[assumptions and commands](../docs/MODAL_EXPERIMENTS.md).
Source/result archives, package manifests, hardware records, all samples,
phase logs, console log and calculation script are retained with the results.

Sources: [Modal pricing](https://modal.com/pricing),
[GPU selection](https://modal.com/docs/guide/gpu),
[timeouts](https://modal.com/docs/guide/timeouts),
[preemption](https://modal.com/docs/guide/preemption).
