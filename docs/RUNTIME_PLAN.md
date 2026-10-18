# Runtime reevaluation — 2026-10-17

The user stopped the previous corpus job and requested a bounded vLLM benchmark
before restarting large jobs. No full experiment or bulk GPT generation is
authorized by this benchmark launch itself.

The main estimate uses **100 retained GSM8K questions, four corruptions each**,
as selected by the user. This target is for the next corpus version; the old
partial v4 corpus has not been relabeled, reduced, or restarted. Shuffle candidate
questions using a recorded seed before selection, apply the same initial-model
K=4 all-correct rule, and freeze the first 100 retained questions. Do not select
them using trained-model scores. Treat each question's four corruptions as a
cluster when estimating uncertainty.
If the source yields fewer than 100 eligible questions, report the actual count;
do not relax the selection rule or synthesize extra independent questions.

## Generation inventory

- Evaluation preparation: K=4 clean solutions per candidate, plus a fresh correct
  trace for each retained question under our existing interpretation. The paper
  requires a correct trace but does not explicitly require a fifth independent
  sample. Reusing a selection trace is a possible separately documented revision;
  it is not silently enabled in this benchmark.
- Evaluation corruptions: 400 GPT-6.1 Sol calls, made once and shared by all runs.
- Training selection: 7,500 MATH candidates times four initial-model solutions,
  made once and shared by all runs.
- Each full unified/split run: 1,000 agent and 1,000 polluter optimizer updates,
  four contexts per update, G=64. This requires at least 784,000 generated
  trajectories, 512,000 GRPO backward passes, and up to 4,000 repair backward
  passes in the current implementation. Clean-trace retries add variable work.
- Each scheduled evaluation: 100 clean answers and 400 corrupted answers. The
  current schedule evaluates initially and every 100 global steps, including the
  final step: 21 evaluations, or 10,500 generated answers per full run.
- Checkpoints: every ten steps, 200 per run. Serialization, hashing, transfer,
  model synchronization and runtime restarts must not be treated as free.

## Benchmark scope and limits

`scripts/probe_vllm.py` tests the pinned DeepSeek revision with vLLM 0.10.1.1,
PyTorch 2.7.1 and Transformers 4.57.6. Python 3.12 is selected explicitly because
that vLLM release excludes Python 3.13. The first setup attempt exposed this
constraint and was stopped automatically before model execution.

The benchmark preserves prompt token IDs, K=4/G=64, temperature .7, top-k 50,
top-p 1, verifier, EOS handling and 1024-token generation limits. The forced-length
throughput cases are diagnostic only. vLLM uses BF16 inference weights, which can
differ numerically from the current HF FP32 weights/BF16 autocast. Greedy agreement
is measured; stochastic outputs are not expected to match bit for bit.

Training-shaped generation and backward components use initial-checkpoint MATH
examples at alpha=0. They measure costs, not learning quality or full self-play
correctness. A multi-update vLLM training integration, synchronization of changing
weights, memory sharing, all alpha values, and exact resume are not established by
this diagnostic. Production training is unchanged. Report any resulting training
speed estimate as conditional on completing and validating that integration.

The configured campaign contains unified and split, with seeds 42, 52, 62. A third
fixed GPT-polluter training baseline was discussed but is not implemented. Its
time must be reported separately, with its own assumptions, rather than silently
included as a measured completed implementation.

The Colab benchmark owns one A100, releases it in a local finally block, and enforces a
remote 55-minute deadline plus a three-minute result-download grace period.

The follow-up Modal hardware comparison has separate documented assumptions,
resource limits, and execution commands in [MODAL_EXPERIMENTS.md](MODAL_EXPERIMENTS.md).
