# Implementation contract after fidelity review

This supersedes the original plan's custom masking, cached-trace fallback, local
evaluation generator and role-specific optimizer states for shared parameters.
Reference: [Li et al., arXiv:2602.00173v1](https://arxiv.org/html/2602.00173v1),
Sections 2–4 and Appendices A, B and E. See EXPERIMENT.md for unresolved details.

The primary comparison runs both unified and split conditions in this repository.
All [documented assumptions](ASSUMPTIONS.md) apply to both; the comparison baseline
is our unified model, not the paper's published score. Exact author settings are
not required to compare the two conditions under our declared protocol.

1. Filter 7,500 MATH training candidates using four initial-model solutions at
   temperature 0.7; retain questions only when all final answers are correct.
2. Initialize the agent and polluter from the same checkpoint. Unified aliases
   their parameters; split copies them once and never resynchronizes them.
3. Alternate five agent and five polluter optimizer updates. Unified freezes a
   copy of the opposing role at block entry. Split uses the inactive live model
   under no-grad/eval and allocates no third model.
4. Draw a correct current-agent clean trace with rejection sampling. Stop after
   the configured retry limit if none succeeds; never fall back to an old trace.
   Choose alpha from {0, .25, .5, .75} and take the next ~10% of reasoning tokens,
   subject to the clean-window cap. Concatenate the generated replacement verbatim;
   a token boundary can occur inside a word or number. Never silently shorten the
   shared prefix separately for different roles; reject oversized role prompts.
5. Use group-normalized binary correctness for agent GRPO and 1-correctness for
   polluter GRPO. Include every candidate in normalization. Prompt constraints
   do not become extra reward masks, arithmetic rules, or sentence-count filters.
6. Generate the repair with the current agent using both clean and polluted
   windows. Clone the visible repair under the same deployment prompt as agent
   GRPO. Account for `<think>` opened in the chat template when extracting visible
   text. Use the token-sum NLL; an empty visible repair contributes an empty sum.
   Do not reject repairs for length, symbolic content, or a boxed final answer.
7. Use one AdamW state per unique parameter and a global cosine/warmup schedule.
   All inactive split gradients stay None, so their weights and moments are not
   updated. Save enough state for exact mid-block CPU resume.
8. Construct one held-out GPT-6.1 Sol recoverability corpus with recorded provenance,
   as requested, following [CORPUS.md](CORPUS.md). The paper uses GPT-5; record this
   evaluation deviation. No local-model or deterministic-edit fallback. Use identical fixed contexts
   for paired runs, report clean-solved recoverability as primary, and disclose
   subset-selection details. Report full-cache metrics separately.
9. Test real gradient updates, role isolation, resume, clean retries/exhaustion,
   unformatted corruption learning, text boundaries, symbolic rewards and corpus
   import. Keep the offline smoke explicitly separate from target-model evidence.

The full and short-run pilot profiles use full tuning, G=64 and 1024-token limits.
The pilot only reduces update count and reporting intervals. Notebooks default to
full settings and never provision infrastructure. Existing pre-fix outputs are
historical; start new runs and rebuild caches under the revised code.
