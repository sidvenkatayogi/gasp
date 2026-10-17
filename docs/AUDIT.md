# Independent paper-fidelity audit — 2026-10-16

Scope: compare this implementation with Li et al., *Learning Robust Reasoning
through Guided Adversarial Self-Play*, arXiv:2602.00173v1, allowing independent
agent and polluter parameters as the intended experimental change. Reviewed the
local PDF, equations, Appendix B prompts, Appendix E settings, source, configs,
tests and notebook entry points. Existing fidelity notes were checked rather than
treated as evidence of correctness.

**Verdict:** the two-model split and core training objectives are implemented
correctly at the source/mechanics level. Three context/target handling defects
were corrected in this audit. This is a reconstruction of the described method,
not a verified reproduction of the authors' implementation or numerical results.
The primary experiment compares our unified and split runs, using the same code
and shared assumptions. The evaluation protocol includes choices that may differ
from the paper, but these do not differ between our two conditions. Exact paper
reproduction is a separate question from this controlled comparison. See the
[explicit assumption registry](ASSUMPTIONS.md). The target-model training path
still needs a real GPU pilot.

## Confirmed against the paper

| Paper requirement | Implementation and result |
|---|---|
| Same initial policy, distinct trained policies in the requested split | `Roles` deep-copies the initial model once. No later weight synchronization. Unified aliases both roles. |
| Five agent steps, then five polluter steps, opposing role held fixed (Appendix E) | Block schedule matches. Unified snapshots the opponent at block entry; split uses the inactive model directly. No redundant third split model. |
| Correct current-agent trace, prefix fractions 0/.25/.5/.75, next ~10% clean window (§3.2/4.1) | Current agent rejection sampling, token-based cuts and a fixed cap. No cached initial trace fallback. |
| Initial K=4, T=.7 all-correct filter over 7,500 MATH candidates (§4.1) | Preparation uses that filter. Shorter datasets yield fewer candidates and are recorded. |
| Agent binary correctness, polluter 1-correctness (§3.3/3.5) | Correct reward signs; no extra semantic/format reward gates. Opponent and verifier are outside autograd. |
| GRPO Eq. (1) | Group normalization, clipped token ratios, mean over completion tokens and group members, no prompt-token loss. No extra KL. |
| Self-generated repair guidance (§3.4, Appendix A/B) | Current agent sees clean and polluted windows; visible repair is cloned under the same polluted prompt as agent GRPO, with sequence-sum NLL. No teacher or ground-truth final answer is substituted. |
| Appendix B prompts | The three training templates match the paper apart from whitespace normalization. No extra agent robustness instruction. |
| Appendix E settings for 1.5B | Full tuning, G=64, T=.7, top-k=50, top-p=1, 1024-token limits, AdamW, LR=3e-6, cosine/.1 warmup, weight decay 0, grad clip 1, GRPO clip .2, lambda .07. |
| Independent updates and checkpoint continuity | Tests check actual gradients, weights, optimizer moments, opponent freezing and exact CPU resume inside/between blocks. |

The single AdamW object is not a shared learning state between split models:
each distinct parameter has its own moments and update counter. Parameters with
`grad=None` are untouched. This is a simple valid implementation of independent
updates. The global LR schedule is identical between conditions.

## Defects corrected

1. **Privileged scratchpad could become the repair target.** DeepSeek's native
   template opens `<think>` in the prompt. If generation reaches its token cap
   before `</think>`, the continuation contains no opening marker. The previous
   parser incorrectly treated this text as a visible repair and could clone
   reasoning that discusses the privileged CLEAN window. Parsing now accounts for
   the prompt's open reasoning block. The same fix excludes unfinished polluter
   scratchpads from the injected window. No candidate is excluded from GRPO:
   an empty extracted window still receives the downstream outcome reward.
2. **Injected spaces changed the corruption.** The joiner inserted a space when
   both boundaries were non-whitespace. Token cuts can split numbers or words:
   `10` + `24` became `10 24`. It now concatenates the sampled replacement verbatim.
3. **Role-specific truncation changed the shared context.** Each prompt previously
   dropped a different amount of prefix to fit its own instructions. With the
   actual DeepSeek tokenizer and a 120-clause prefix, the agent kept 120 clauses,
   repair kept 106, and polluter kept 73. Prompts now preserve the supplied prefix
   and raise on overflow. This removes the silent mismatch; it does **not** prove
   that every real training episode fits the 1024-token budget. An author overflow
   policy remains unavailable, and increasing the cap is a protocol change.

Prompt/source fingerprints reject prior caches/checkpoints. Use fresh `v3`
directories and rebuild caches; do not combine pre-audit and post-audit runs.

## Remaining assumptions and limitations

- **Evaluation is a shared experimental choice.** The paper's §4.2 describes
  K=4/T=.7 clean-solved selection and corruption of a correct solution. This harness
  imports fixed contexts and subset flags, then evaluates greedily. It never
  recomputes that selection for the evaluated checkpoint. This can support a
  controlled sharing comparison, but its scores must not be presented as direct
  reproductions of Table 1 without matching author artifacts and selection rules.
  The optional secondary adapters are also explicitly modified protocols.
- **Unspecified numerical defaults remain assumptions:** 1,000 steps per role,
  four accumulated contexts, 128-token clean-window cap, final-20% guidance decay,
  AdamW betas/epsilon, global optimizer schedule and the 32-attempt clean-trace
  watchdog. These are documented, not confirmed author defaults. The paper states
  blocked opponent freezing; the snapshot is an implementation of that requirement.
- **Sampling/log-probability convention is unverified against author code.**
  Sampling uses T=.7/top-k=50, while the loss uses raw model log probabilities.
  This is shared across conditions, but the paper does not specify whether its
  modified TRL uses this exact convention. Each fresh group receives one update,
  so clipping is normally inactive until parameters change; this is not evidence
  of a sign or isolation bug.
- **Runtime differs.** Synchronous Hugging Face generation replaces the paper's
  vLLM/distributed implementation. This simplifies the code and avoids stale
  worker weights, but serial rollouts may be substantially slower. Full 1.5B
  CUDA/bf16 training, memory fit and useful generated repairs remain unverified.
- **No training data or evaluation corpus is bundled.** The importer checks
  structure/provenance, not whether the claimed generator actually produced
  mathematically valid corruptions. No external teacher is used for training.

## Engineering scope

The core is reasonably small: two role objects, one optimizer, one alternating
loop and two losses. There is no need to replace it with an agent framework,
extra reward model, curriculum controller or model synchronization service.
Checksums, resume state and paired aggregation are reproducibility support, not
additional learning objectives. The optional benchmark/notebook machinery is
outside the core method. Requiring an evaluation corpus to construct a trainer
is an avoidable coupling, but does not change its weight updates.

## Validation

Run `.venv/bin/python -m pytest -q` from the repository. Tests exercise real tiny
Qwen2 gradients and include the ordinary autoregressive `model.generate` path,
as well as the binary smoke path. Actual target-tokenizer checks are recorded in
`artifacts/audit-tokenizer-checks.json`; no target model weights were downloaded.
See `artifacts/VALIDATION.md` for the final test and packaging results. CPU tests
and tokenizer checks do not establish training convergence or scientific gains.

Primary source: https://arxiv.org/html/2602.00173v1 (especially §§2–4 and Appendices
A, B, E). Target tokenizer: the official
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` repository on Hugging Face.
