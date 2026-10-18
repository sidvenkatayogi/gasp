# Modal experiment plan and assumptions

Completed results: [Modal benchmark report](../artifacts/MODAL_BENCHMARK.md).
Both GPU jobs passed; all Modal apps were confirmed stopped afterward.

## Scope

This is a bounded hardware comparison, followed by a decision about full training.
The selected evaluation target remains 100 retained GSM8K questions and four GPT
corruptions per question. The benchmark neither creates that corpus nor calls GPT.
The fixed GPT-polluter training baseline is still unimplemented.

First compare A100-80GB and H100 SXM using the existing pinned software and model.
Both run the inference diagnostic and actual backward/AdamW components for unified
and split policies. These are disposable component updates, not scientific results
from complete self-play experiments. Production `src/gasp` is unchanged.

## Explicit assumptions and controls

1. Use Python 3.12, PyTorch 2.7.1, Transformers 4.57.6 and vLLM 0.10.1.1, with
   DeepSeek-R1-Distill-Qwen-1.5B revision
   `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
2. Preserve BF16 inference/compute, FP32 training master weights, G=64, four
   contexts/update, 1024-token limits, prompts, rewards and optimizer settings.
   Inactive split optimizer moments continue to move to CPU. Greater VRAM alone
   does not change the memory policy in this first comparison.
3. Replay exactly the same saved four alpha=0 MATH contexts, completion tokens,
   rewards and repair targets for backward timing on both GPUs. The fixture is
   `artifacts/gasp-vllm-20261017-053328/results/vllm.json`; record its SHA-256 in
   each worker report. Native inference output is separately preserved as
   `vllm-generated.json`. Stochastic inference output lengths may differ across
   hardware and must be considered when comparing generation wall time.
4. Unified uses one parameter set and one shared optimizer state, plus the frozen
   block opponent snapshot. Split uses two parameter sets with independent
   optimizer moments. Time snapshot refresh and optimizer placement separately.
   The diagnostic switches role after one update rather than after a five-update
   block, so its raw transfer/snapshot overhead is not a steady-state step cost.
5. Saved trajectories are timing fixtures, not fresh on-policy trajectories after
   an update. This diagnostic cannot validate a complete training run or vLLM
   policy synchronization. The existing production trainer still uses HF rollouts.
6. Request `H100!` to prevent an automatic H200 upgrade from confounding the
   comparison. Record actual hardware using `nvidia-smi`.
7. Prepare dependencies and download public model weights during CPU image build.
   Explicitly allowlist uploaded source/data; never upload `.env` or credentials.
   GPU workers use the offline model cache. No GPU is used for API calls.
8. One GPU job at a time: max one container, no minimum warm containers, single
   use, zero configured failure retries, a 25-minute internal phase deadline and
   a 30-minute Modal execution timeout. Startup has a separate 15-minute timeout.
   Modal may restart a preempted function independently of ordinary failure retry
   settings; these timeouts are per attempt, not an account spending cap.
9. Both GPU jobs request four physical CPU cores and 64 GiB host RAM, with a
   96 GiB RAM limit. Modal bills the larger of requested and actual CPU/RAM usage.
   GPU loading and execution are billed; no persistent deployment is created.

## Current resource options

Public list prices checked 2026-10-17, excluding CPU/RAM and any credits:

| GPU | GPU $/hour | Purpose |
|---|---:|---|
| A100 40GB | 2.0988 | Existing memory-saving approach fits tightly |
| A100 80GB | 2.4984 | First larger-memory baseline |
| H100 SXM 80GB | 3.9492 | First speed comparison |
| H200 SXM 141GB | 4.5396 | Additional room/bandwidth; test if H100 results justify it |
| B200 | 6.2496 | Requires separately validated compatible software stack |

Four requested CPU cores plus 64 GiB RAM add at least $0.700128/hour while the
function is running. Thus the first two worker configurations start around
$3.20/hour and $4.65/hour. H100 must finish the same work about 1.45x faster than
A100-80GB to break even on these combined rates. This is not a bill estimate for
the image build, extra actual CPU/RAM consumption, loading, or account credits.

## Running and inspecting

Authenticate once using `.venv/bin/python -m modal token new`.

```sh
.venv/bin/python scripts/run_modal_benchmark.py --prepare-only
.venv/bin/python scripts/run_modal_benchmark.py --gpus A100-80GB 'H100!'
.venv/bin/python -m modal app list
```

Each run preserves `source.zip`, per-file hashes, and each GPU's result bundle in
`artifacts/gasp-modal-<UTC timestamp>/`. A failed remote report raises locally;
its logs remain available. The runner's app context exits after each GPU, and the
single-use worker shuts down after returning its bundle.

## Options after the benchmark

- Run independent conditions/seeds concurrently to reduce calendar time. This
  does not by itself reduce total GPU-hours and does not change either algorithm.
- Use a faster measured GPU if throughput improves enough to justify its rate.
  Moving to Modal alone does not accelerate the same kernels on similar hardware.
- Validate batched backward computation and live-policy vLLM integration before
  relying on the earlier optimized runtime estimates. Current training processes
  256 separate trajectories per update; faster hardware may leave that overhead.
- Consider retaining split Adam states on GPU on 80GB+ hardware only as a separate
  execution comparison. Keep the existing tested offload path for the first run.
- Full jobs must use durable checkpoints and bounded chunks: Modal functions have
  a maximum 24-hour execution timeout and GPU functions can be preempted. Preserve
  optimizer, scheduler, RNG, data cursor and role/block state when resuming. Do
  not split a long job by silently resetting the learning schedule or policies.

Sources: [pricing](https://modal.com/pricing),
[GPU choices and upgrades](https://modal.com/docs/guide/gpu),
[timeouts](https://modal.com/docs/guide/timeouts),
[preemption](https://modal.com/docs/guide/preemption),
[weight caching](https://modal.com/docs/guide/model-weights).
