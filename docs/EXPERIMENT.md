# Fidelity notes after review

Reference: Li et al., *Learning Robust Reasoning through Guided Adversarial Self-Play*,
[arXiv:2602.00173v1](https://arxiv.org/html/2602.00173v1). The repository PDF is the
20-page version with SHA-256 `a606082c825a0234403fa7f623a503b710f6740ca5da92ed435f78ed91380cd5`.
Appendix B prompts are transcribed with whitespace normalized, under CC BY 4.0.

## What matches the stated method

- Initial MATH train filtering: 7,500 candidates, K=4, T=.7, all answers correct.
- Fresh correct current-agent clean traces; prefix fractions {0,.25,.5,.75}; next
  ~10% of tokens as the clean window, with a fixed cap.
- GRPO: population-standard-deviation group advantages, token-level clipping .2,
  mean over generated tokens then trajectories, G=64, no explicit KL penalty.
- Agent reward is binary final-answer correctness. Polluter reward is exactly
  1 minus that correctness. No gradients pass through the opponent or verifier.
- Current-agent repair generation sees the clean and polluted windows. Imitation
  sees only question, prefix and pollution and sums repair-token NLL.
- AdamW, peak LR 3e-6, cosine decay, warmup .1, weight decay 0, gradient clip 1,
  bf16 computation, gradient checkpointing, T=.7/top-k=50/top-p=1.
- Five agent then five polluter updates; lambda .07 for this 1.5B configuration,
  decaying to zero near the end; seeds 42,52,62; full tuning for the 1.5B model.

Split initializes two independent copies of the same checkpoint. Unified has one
parameter object shared by both roles. Split never copies one trained policy into
the other. Its inactive policy serves directly as the fixed opponent. Unified needs
a frozen opponent copy for each block because its live weights serve both roles.

One optimizer owns each unique parameter exactly once. Thus unified shares AdamW
moments with its weights, while split has independent moments for independent
parameters. The scheduler advances once per global optimizer update in both cases.
Inactive split parameters have grad=None and AdamW does not advance their state.

## Corrections to the earlier implementation

Removed semantic/format validity masks from both losses and group normalization.
The pollution parser only unwraps the visible tag block, preserving whitespace; if
tags are absent, it uses the visible text and records that fact. It does not score
mathematical plausibility. The generated polluter trajectory, including any internal
reasoning, still receives the outcome-based GRPO loss. The repair parser extracts
visible text after `</think>`; it does not impose an extra token cap, four-sentence
filter or final-answer ban. An unfinished `<think>` block has no visible repair,
including when the opening marker was supplied by the chat template rather than
generated. The same rule excludes unfinished scratchpad text from the injected
pollution; its full generated trajectory still receives the outcome reward, even
if the extracted window is empty. Empty repair targets contribute zero guidance.
Generation caps and prompt constraints still apply.

Removed cached initial-trace fallback. Clean generation now rejection-samples the
current agent until a correct nonempty trace is available. A 32-attempt watchdog
stops the run before an optimizer update if generation cannot supply one. Investigate
exhaustion and resume a saved checkpoint; it is not a zero-signal training step.
Costs vary with current-agent competence, so aggregation matches GRPO/repair
allocations but reports clean-retry costs rather than requiring them to be equal.

Removed deterministic corruption edits and the frozen-base evaluation generator.
Non-smoke recoverability imports a fixed independent held-out corpus. It can now
be constructed with GPT-6.1 Sol as requested; see [CORPUS.md](CORPUS.md).
The synthetic smoke remains a fixture and is never accepted for real-model runs.
Removed the redundant split snapshot and separate shared-parameter moment sets.
Prefix/window joining now concatenates the generated replacement verbatim, without
inserting spaces inside token-split words or numbers. Role prompts reject overflow
instead of silently trimming different amounts of the shared reasoning prefix.
This can stop real training on long contexts; the paper does not give a truncation
policy that would justify changing those contexts. Full settings are the notebook default. The pilot
now changes only update count and evaluation/checkpoint intervals.

## Assumptions the paper does not resolve

The complete registry is [ASSUMPTIONS.md](ASSUMPTIONS.md): T1–T21 cover training and
E1–E7 cover evaluation. It records the exact shared choices, the intended treatment,
the paired GPU run sequence and the conclusions the comparison can support.
These are declared settings for our experiment, not verified author defaults.

Both unified and split run our implementation with these assumptions held fixed.
A departure shared by both conditions limits reproduction claims without itself
creating a mismatch between the two conditions. The fixed evaluation corpus and
denominator are deliberate controls for this comparison. Our unified score is the
baseline; the paper's reported score is not.

The global optimizer schedule is the simpler conventional interpretation of one
shared model; the paper does not prove it is the authors' exact implementation.
Do not claim bit-for-bit reproduction. Full tuning uses fp32 parameters/moments
with bf16 autocast. The shipped settings target 1.5B; changing to Qwen3-8B also
requires the paper's rank-64 LoRA and lambda .04, not just changing the model ID.

## Evaluation input

Use the independently generated and inspected GPT-6.1 Sol corpus described in
[CORPUS.md](CORPUS.md), or import author artifacts with their actual provenance.
Sol replaces the paper's GPT-5 evaluator by user request. Neither corpus is bundled.
Each JSONL row must supply:

```json
{"id":"gsm8k-test-0-a0","question_id":"gsm8k-test-0","question":"...","answer":"...","prefix":"...","polluted":"...","initial_clean_solved":true,"alpha":0.0}
```

`polluted` is the edited window itself, not an instruction or replacement answer.
`initial_clean_solved` is the historical field name for the fixed corpus subset
mask. Describe exactly which checkpoint selected it. For a newly constructed paired
benchmark use K=4 at T=.7, retaining all-correct instances; author artifacts may use
a different model-conditioned selection that must be documented explicitly.
Preserve any supplied per-row provenance and keep a different protocol/version
for a reconstructed corpus. A source declaration does not prove semantic validity:
inspect the actual corruptions and retain generator outputs.

The companion provenance JSON requires:

```json
{
  "protocol":"reconstructed-sol-fixed-initial-subset-v1",
  "generator_model":"gpt-6.1-sol",
  "generator_revision":"gpt-6.1-sol",
  "held_out":true,
  "source_url":"SOURCE_ARTIFACT_OR_DATASET_LOCATION",
  "generation_settings":{"reasoning":{"effort":"low"},"max_output_tokens":4096,"store":false},
  "subset_selection":"Exact checkpoint and K=4, T=.7 all-correct filtering procedure"
}
```

Use actual values, not these placeholders. The generator writes the full provenance
automatically, including a note that the Sol identifier is not a dated snapshot.
`generator_model` accepts Sol, GPT-5, or a dated GPT-5 model ID. The importer
preserves supplied windows and subset flags and builds
prompt tokens once. Missing/empty windows, mixed generator identities and training
split rows fail validation; no rows are repaired or replaced automatically. Both
conditions must consume the same corpus hash. `generate-eval` makes paid API calls;
import and training do not call OpenAI.

Primary metric: `corrupted_initial_clean_solved`. Also report `corrupted_full` and
`clean_full` over the supplied corpus, not as claims of full GSM8K coverage unless
that corpus actually includes every question. The fixed subset avoids changing the
paired denominator; its equivalence to the paper's exact selection remains unverified
until author artifacts are available.

## Validation and limitations

See [the current validation report](../artifacts/VALIDATION.md). CPU tests exercise
real gradients and symbolic Math-Verify rewards. They do not establish that the
1.5B CUDA path fits memory, that generated repairs are useful, or that split beats
unified. The full campaign and actual corpus generation remain unexecuted.

Optional MR-GSM8K/MHPP and RUPBench adapters retain their separately labeled local
protocol. ACCreason needs explanation judgements in addition to first-error steps;
the adapters do not fabricate judge scores. MHPP clean-solved coding verification
is not implemented. Self-revision uses the model's own failed initial solutions.

Old checkpoints used a different algorithm and schema. Rebuild caches and start
fresh v3 runs; do not silently resume, relabel or pool historical outputs.
