# Shared assumptions and unified-versus-split comparison

Recorded 2026-10-16, before any real-model GPU comparison.

## Question and comparison

The experiment asks: **under one implementation of GASP, what changes when the
agent and polluter have independent parameters instead of sharing parameters?**
Both conditions run this repository's code. The baseline is our unified run.
The paper motivates the algorithm and supplies the stated settings; its published
scores are background context, not the numerical baseline for this comparison.

The intended treatment is parameter sharing and its consequences:

| Property | Unified | Split |
|---|---|---|
| Initialization | One copy of the resolved initial checkpoint | Two identical independent copies of that checkpoint |
| Trainable parameters | Same parameters serve both roles | Separate agent and polluter parameters |
| AdamW state | One set of moments per shared parameter | Independent moments per model's parameters |
| Opponent during a five-step block | Frozen copy taken at block entry | Inactive model used directly under no-grad inference |
| Effect of a role's update | Changes weights used by both roles | Changes only that role's weights |
| Agent exported/evaluated | Shared model used as the agent | Agent model alone |

Split therefore also has more total trainable parameters and optimizer state.
Unified weights receive both roles' updates; each split model receives only its
own role's updates. These are consequences of the requested split, and we retain
them rather than altering update budgets to compensate. This experiment evaluates
the complete shared-policy versus independent-policy design, not a separation of
weight sharing from AdamW moment sharing.

## Shared training assumptions

The following choices fill in details not fully specified in the paper or select
an implementation of them. They apply identically to both conditions and are not
asserted to be the authors' exact settings. Values refer to `configs/full.json`
and the current source; `configs/pilot.json` is described separately below.

| ID | Choice | Assumption used in both conditions |
|---|---|---|
| T1 | Training duration | 1,000 agent optimizer steps and 1,000 polluter optimizer steps: 2,000 global steps. |
| T2 | Contexts per optimizer step | Four contexts with accumulated gradients, averaged equally. Each has its own G=64 reward-normalized group. |
| T3 | Window construction | Tokenize the extracted reasoning; start at `floor(alpha * length)`; width `min(128, max(1, floor(.1 * length)))`. The 128-token cap and rounding are local choices. |
| T4 | Guidance decay | Keep lambda=.07 until the final 20% of agent updates, then linearly reduce it to zero on the last agent update. The paper specifies decay toward the end but not its onset. |
| T5 | AdamW details | Betas (.9, .999), epsilon=1e-8, one optimizer containing every unique trainable parameter once. Inactive parameters have `grad=None`. |
| T6 | Learning-rate clock | One global cosine schedule over 2,000 optimizer steps, with .1 warmup, advanced once per step. Split does not restart or independently advance schedules for each model. |
| T7 | Old policy and updates per rollout | Store detached log probabilities at generation, then take one accumulated optimizer update for fresh groups. No additional PPO/GRPO epochs reuse those groups. |
| T8 | Sampling probabilities in the loss | Generate at T=.7/top-k=50/top-p=1, but calculate importance ratios with raw model log probabilities. No temperature rescaling or top-k renormalization in the loss. |
| T9 | Group normalization | Population standard deviation plus epsilon=1e-8. Constant-reward groups have zero advantages; every candidate remains in the group. |
| T10 | Episode sampling counts | One pollution and one self-repair per agent context; one agent completion scores each polluter candidate. G=64 is used for the active role's GRPO group. |
| T11 | Correct clean trace | Rejection-sample the current designated agent, up to 32 attempts per context. Abort the step on exhaustion; do not substitute a cached trace or apply an optimizer update to a partial accumulation. |
| T12 | Text serialization | Native tokenizer chat template with the Appendix B training prompts, normalized only for template whitespace. Concatenate prefix and extracted pollution verbatim. |
| T13 | Prompt overflow | Retain the 1024-token prompt cap. Raise an error if any role prompt exceeds it; do not shorten a role's shared prefix. This is a fail-fast choice, not a demonstrated solution for long episodes. |
| T14 | Visible output | Remove internal reasoning using both the prompt's open `<think>` and the generated closing marker. Unfinished scratchpads yield no visible repair/pollution. Empty repairs have zero guidance loss. |
| T15 | Malformed or capped generations | Missing pollution tags fall back to visible generated text. Empty pollution and capped completions still receive the outcome reward; no extra format, mathematical-validity or length reward mask is used. |
| T16 | Final-answer verifier | Math-Verify 0.8.0 with the repository's final-answer extraction (answer tags, balanced boxed answers, `####`, or standalone numeric output). Plain numeric answers followed by unit words use its plain-expression parser; symbolic expressions retain math delimiters. No arbitrary last-number fallback outside an explicit final answer. |
| T17 | Training preparation randomness | Shuffle the candidate pool and perform initial filtering with seed 42 once. Reuse the resulting immutable training cache for both conditions and all training seeds. Oversized initial question prompts are excluded and recorded in the preparation audit. |
| T18 | Precision and execution | fp32 trainable weights/AdamW state with bf16 autocast; gradient checkpointing; dropout modules disabled; synchronous Hugging Face/PyTorch generation with SDPA. The paper's vLLM/distributed runtime is not reproduced. |
| T19 | Reporting intervals | Evaluate initially, every 100 global steps and finally; checkpoint every 10 global steps and at a requested stop. Evaluation saves/restores training RNG state. |
| T20 | Optimizer storage on Colab | Use `optimizer_offload=true` for both conditions on the 40 GiB A100. Keep the active policy's AdamW moments on CUDA throughout its five-update block and the inactive policy's moments in CPU RAM; evict old moments before loading the next role's moments. Unified has only one shared set, which stays on CUDA. Both models' weights stay on CUDA. No quantization, precision change, model sharing, or optimizer-state reset. Report CPU memory and transfer-inclusive runtime as well as GPU memory. This is an execution choice, not a claim about the paper's runtime. |
| T21 | Evaluation preparation batching | `preparation_batch_questions=4` batches four questions with four samples per question (16 sequences) using Hugging Face generation and left-padding. Fresh correct traces are still rejection-sampled separately, and all four alpha cuts are retained. K, temperature, top-k/top-p, precision, prompts, verifier and token limits are unchanged. Batching changes RNG consumption and may change floating-point results; `data/full-v4` restarts from the initial seed and is not mixed with serial-v3 traces. Checkpoints commit whole batches, including RNG; interrupted batches replay. This setting affects evaluation preparation, not training updates. |

T20 checkpoint exception: restoration temporarily moves model weights to CPU so
PyTorch's optimizer loader cannot eagerly allocate both moment sets on CUDA. It
then returns the same parameters to their original devices. Ordinary training
updates and role changes move only optimizer moments. Exact-resume tests cover
this path, including parameter identities and RNG state.

Paper-stated settings retained include the 1.5B backbone, full tuning, five agent
then five polluter updates with a fixed opponent, K=4/T=.7 initial solvability
filtering, alpha in {0,.25,.5,.75}, G=64, 1024-token prompt/completion limits,
GRPO clip .2, no explicit KL, lambda .07, AdamW peak LR=3e-6, cosine decay,
.1 warmup, weight decay 0, gradient clip 1 and seeds 42/52/62. See
[EXPERIMENT.md](EXPERIMENT.md) for the source mapping and corpus schema.

The default rank-16 LoRA field is unused by these full-tuning runs. A different
backbone or LoRA experiment requires a separately declared configuration.

## Shared evaluation assumptions

| ID | Choice | Assumption used in both conditions |
|---|---|---|
| E1 | Corruption corpus | One independent held-out GPT-6.1 Sol corpus, per user request, replacing the paper's GPT-5 generator. Imported once with provenance, identical prompts, question IDs and cache hash for both conditions. Its mathematical validity needs inspection; provenance declarations alone are not proof. |
| E2 | Primary denominator | One fixed clean-solved subset chosen before the comparison. For a newly constructed corpus, use the same initial checkpoint and K=4 clean samples at T=.7, retaining all-correct questions. Record checkpoint and selection details in provenance. Do not reselect the subset separately for trained models. |
| E3 | Existing external corpus | If using author/external artifacts with a different documented selection procedure, freeze that procedure for both conditions and label the protocol accordingly. It is not automatically equivalent to E2. |
| E4 | Decoding | Greedy pass@1 (`eval_temperature=0`), 1024 completion tokens, evaluation seed 10001, and the same final-answer verifier. |
| E5 | Metrics | Primary: `corrupted_initial_clean_solved`; secondary: `corrupted_full` and `clean_full` over the supplied corpus. The historical primary field name means the fixed imported subset. It is not a per-checkpoint clean-solved rate. |
| E6 | Aggregation | For each of seeds 42,52,62, compute split minus unified on identical evaluation rows. Report the paired differences and their mean/sample standard deviation across seeds, alongside the individual conditions. |
| E7 | Held-out use | Use the fixed endpoint budget for the primary result. Do not choose the best checkpoint, decay schedule or other settings by the final held-out scores. Development checks need separate data. |

The fixed test corpus and denominator are intentional controls for our comparison.
The [corpus procedure](CORPUS.md) records all construction choices, including the
fresh trace, all four alpha cuts, Appendix B prompt reuse, Sol API settings,
response validation, and the lack of an immutable dated Sol snapshot.
They may differ from the paper's exact evaluation construction. That limits direct
comparison with published scores; it does not itself invalidate the comparison
between our two conditions. Likewise, shared numerical assumptions can influence
the size or direction of the measured split effect, so conclusions apply to this
documented setup. Sharing a bug would not make it harmless; the correctness tests
and GPU pilot remain necessary.

## Controls and permissible differences

For every seed pair, hold fixed the source version, initial model revision and
weight hash, tokenizer, packages, GPU type, training/evaluation caches, question
order, alpha sampling, prompts, rewards, verifier, guidance rules, optimizer
hyperparameters, global schedule, role-update counts and allocated rollout/token
budgets. The two config objects differ only in `condition`. Run in separate
processes, sequentially on the same GPU setup to avoid contention. Verify equal
initial evaluation predictions before interpreting a final delta.

Do not force the trained models to produce identical traces, pollutions or
repairs. Those are policy-dependent outcomes. Paired seeds do not imply identical
later random draws: differing clean-retry counts and completion lengths can
advance RNG state differently. Record actual tokens, retries, memory and runtime
as measured costs; matched rollout caps and update counts do not promise equal
FLOPs, wall time or memory. Do not use each model's own trained polluter as its
primary test adversary.

The current aggregator checks paired config/source/tokenizer/package/GPU/cache
identity, evaluation row order and GRPO/repair allocations. Initial weight hashes
and initial evaluation outputs should also be inspected in the existing run
artifacts before accepting a comparison.

## GPU run sequence and decisions before execution

1. Use this codebase for both conditions, on the same hardware/software setup.
   Resolve the initial checkpoint revision; prepare one shared training cache and
   one shared held-out corpus. Record the actual corpus provenance, not placeholders.
2. Run `configs/pilot.json` for both conditions at seed 42. It has 10 steps per
   role, evaluates every 10 global steps and checkpoints every 5. All algorithm,
   precision, group-size and token-limit settings match the full profile.
3. Check role isolation, real gradient updates, useful visible corruptions/repairs,
   retry exhaustion, prompt overflow, memory and checkpoint/resume behavior.
   Pilot scores are diagnostics, not the full comparison result.
4. If an operational problem requires a protocol change, document it, apply it to
   both conditions and rerun both under the same version. A shared implementation
   correction does not require matching the paper's reported scores first.
5. Run both full conditions for seeds 42,52,62. Report the paired agent-quality
   differences and measured resource costs. State the assumptions with the results.

The command pairs are in [README.md](../README.md#train-and-compare), and notebook
02 already loops over both conditions and all configured seeds. Author artifacts
would improve fidelity claims but are not a prerequisite to this controlled
comparison using an independently constructed, inspected evaluation corpus.

Execution update (2026-10-17 UTC): an A100-SXM4-40GB passed the CUDA/bf16
offload-equivalence tests and a full-size accumulated-gradient memory probe. Raw MATH
and GSM8K data are validated; actual initial-model context preparation is running.
One real GPT-6.1 Sol corruption was generated and inspected using a committed
context while DeepSeek preparation continued. Its saved response will be reused
by final corpus generation after exact request validation. The key stayed local.
Filtered MATH training data, the completed corruption corpus and full paired
training runs remain pending. See [validation](../artifacts/VALIDATION.md) for
measurements and the limits of the memory probe.
