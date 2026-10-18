# Independent review of the reduced GASP comparison

Review date: 2026-10-17. Scope: the proposed roughly ten-total-GPU-hour,
unified-versus-split experiment. No GPU/API jobs were launched and no training
code was changed for this review.

**Verdict: retain the reduced design, but do not launch it as currently specified.**
The proposed 50 optimizer updates per role, G=16, one context per update, 100
retained MATH questions and one paired seed can produce a useful **fixed-budget
exploratory comparison**. It is not yet a fully costed experiment, and its current
reporting does not quantify uncertainty over evaluation questions. Make the four
minimum revisions below before launch. There is no basis to promise a detectable
learning gain or a significant unified/split difference after 50 agent updates.

I read the complete extracted paper, including methods, results and appendices
A–F, in [references/paper.txt](../references/paper.txt), before reading the plan or
implementation. I cross-checked the training/evaluation methods and Appendix E
against the [actual arXiv v1 paper](https://arxiv.org/html/2602.00173v1), then read
the cycle report, exploratory plan, assumptions, experiment/corpus documentation,
and the training, preparation, configuration and evaluation source. Paper page
numbers below refer to the 20-page v1 PDF. This is an independent assessment of
the proposed experiment, not a certification of every implementation detail.

**What is preserved, and what the paper does not justify**

The central experimental question is sound: compare the agent produced by a
shared policy against the agent produced by separate policies, under the same
implementation and allocated update/rollout budget. Separate optimizer histories,
twice the total parameter capacity, and the shared agent receiving polluter
updates are consequences of that treatment. This is already correctly stated in
[ASSUMPTIONS.md, lines 7–29](ASSUMPTIONS.md). It does not isolate weight sharing
from optimizer sharing, and it is not a comparison at equal total parameters or
exactly equal realized FLOPs.

| Paper evidence | Consequence for this experiment |
|---|---|
| §3.2–3.5, pp.3–5: correct current-agent traces, local replacement windows, outcome rewards, self-generated repair imitation, alternating shared parameters. Extracted lines 153–175, 184–250, 258–315. | Keep these mechanisms identical in both conditions. A split architecture is the intended experimental intervention. |
| §4.1, p.5: 7,500 MATH candidates; initial K=4 all-correct filtering. Extracted lines 292–306. | A bounded target of 100 retained questions is a substantial data reduction, but preserves the competence filter. It is not 100 raw candidates. |
| §4.2, p.6: clean-solved evaluation, held-out GPT-5 corruption, additional diagnosis/reliability/revision tests; Appendix A.9, p.16: full-benchmark results. Extracted lines 363–391 and 948–981. | The proposed fixed initial subset and GPT-6.1 Sol corpus are an explicitly reconstructed recovery test. They cannot substantiate the paper's broader robustness claims. |
| Appendix E, pp.19–20: G=64, specified optimizer/decoding settings, 1024-token input/output caps, five-update role blocks, three seeds. Extracted lines 1182–1211. | G=16, one seed, 2048 input tokens and the HF runtime are declared departures. The 1024 output cap and role blocks remain useful controls. |

The paper's Appendix E does **not** state the repository's 1,000 updates per role
or four contexts per optimizer update. Those are local assumptions T1/T2, not
paper-established requirements. Do not describe the reduced run's trajectory
count as a measured fraction of the authors' training budget. Likewise, the
paper's large final scores are not an acceptance target: its main clean-solved
results and Appendix A.9 full-benchmark results use different denominators, and
our reconstruction changes both data scale and evaluation generator. The proper
numerical baseline is this implementation's own initial and unified models.

**Minimum revision 1 — make the entire budget executable and bounded**

The measured 5.68 GPU-hours for paired training/checkpoints, or 6.82 with the
judgmental 20% allowance, is a reasonable planning estimate. It is not a measured
full-pipeline runtime. The benchmark used four repeated questions, an older
200-global-update schedule, and only one corruption for evaluation. Fresh trace
retries, lengths and repair/pollution quality can change with the new pool and
training. See [the cycle report, lines 32–68 and 94–109](../artifacts/CYCLE_BENCHMARK.md).

Before committing to ten cycles, implement and time the necessary bounded,
batched paths on representative data:

- Training preparation must stop at the first 100 eligible questions in a
  recorded shuffled order, with explicit candidate/time limits and an audit of
  rejected candidates. Current `prepare_train` is a serial candidate sweep, not
  a retained-target operation (`src/gasp/data.py:115–144`).
- Evaluation preparation must likewise use a recorded shuffled candidate order
  and retained target. Current `prepare_eval_contexts` batches filtering, but
  takes an ordered candidate prefix and separately generates fresh traces;
  it has no 100-retained stopping rule (`src/gasp/eval_corpus.py:21–145`).
- Batch evaluation across questions and all alpha values, preserving prompt
  tokens, greedy decoding, answer verification and exact row mapping. Current
  `evaluate` generates rows serially (`src/gasp/evaluation.py:13–40`). Check the
  changed batching path on a small set before relying on its measurements.
- Account for all allocated GPU-worker time, including preparation, startup,
  checkpointing and evaluation. Release GPUs during GPT corpus generation and
  human/CPU review. State whether the already spent approximately 0.8 GPU-hours
  of cycle diagnostics is inside the ten-hour limit; conservatively include it
  unless the accounting scope is otherwise established.

A useful saving is **one shared complete initial evaluation plus the two trained
endpoints**: 1,500 answers instead of 2,000 (100 clean + 400 corrupted per pass).
This is defensible because both conditions start with the same agent weights and
greedy protocol. Verify initial weight/tokenizer/config identity, do a small
prediction-agreement spot-check in each condition, and explicitly reference the
shared baseline artifact in both reports. It must not masquerade as two separately
executed complete evaluations.

Measure preparation acceptance rate and representative evaluation throughput
before setting the final common cycle count. Preserve the 100-question test
before spending the remainder on more training. If the total projection does not
fit, reduce both conditions by whole cycles **before training**, and recompute
their common schedule; do not change one condition's budget after seeing scores.
Avoid opportunistic extensions or early endpoint selection based on held-out
accuracy. A resource watchdog may stop a run, but an unfinished pair is an
interrupted experiment, not the planned endpoint result.

**Minimum revision 2 — freeze an interpretable test, including its quality audit**

Keep 100 distinct held-out GSM8K questions and all four fixed corruptions per
question. Use the initial checkpoint's K=4 selection once, with training/test
question disjointness and one immutable corpus hash. Never reselect questions
for a trained model or use either condition's own learned polluter as its primary
test adversary. The existing fixed-subset design is appropriate for the paired
comparison, even though the paper does not unambiguously specify that same
cross-model selection policy.

Freeze the question IDs, prefix/window bytes, generator responses, decoding,
verifier and semantic-audit record before viewing trained-model scores. Validate
all 400 rows structurally. At minimum, inspect a seeded sample of 40 short
clean/polluted pairs covering all four alphas; audit all 400 if practical. Record
whether the original local step is sound, the edit introduces an identifiable
mathematical error, the window remains coherent, and the output avoids an explicit
final-answer giveaway or a meta-instruction. An all-correct final-answer filter
does not prove that every intermediate step in a trace is correct. Expand review
if the sample reveals a systematic defect. A sample audit is evidence about
quality, not proof that every remaining row is valid.

The current importer checks fields/provenance, and the generator checks tags,
nonemptiness and changed text; neither establishes mathematical invalidity of the
new step (`src/gasp/data.py:160–193`, `src/gasp/eval_corpus.py:165–176,235–252`).
Missing tags in *training* are likewise not by themselves proof of an invalid
attack. Do not add an unplanned reward mask or quietly drop examples to improve
scores. Resolve corpus construction defects under a documented policy before
locking it; preserve all attempts and exclusions. Do not resample corruptions
until they fool the initial or trained agent.

The already-budgeted clean evaluation on these same 100 questions is mandatory.
Call it **clean accuracy on the fixed initial clean-solved subset**.
`clean_full` currently means only unique questions in the supplied corpus
(`src/gasp/evaluation.py:18–40`); it is not full GSM8K accuracy or broad retention.
With this corpus, `corrupted_full` and `corrupted_initial_clean_solved` are also
the same score, not independent evidence.

A random 50-question **unfiltered** clean sentinel set is optional if broader
math-competence claims are desired: one shared baseline plus two endpoints adds
150 answers, no K=4 selection and no GPT calls. Keep it disjoint from the final
recovery questions and development data. It remains a small diagnostic, not a
replacement for full GSM8K. If omitted, simply restrict the retention claim to
the selected subset; this omission need not block the core recovery comparison.

**Minimum revision 3 — report paired uncertainty over questions and actual learning**

For condition c and question q, define R(c,q) as the mean correctness of its four
alpha contexts. The primary endpoint effect is the mean over 100 questions of
R(split,q) − R(unified,q). Report that difference in percentage points, each
condition's score, and each condition's paired change from the shared initial
model. A relative winner that merely degrades less must not be described as a
successful robustification result.

Add a CPU-only paired question-cluster bootstrap, for example 10,000 resamples
with a recorded analysis seed. Resample question IDs and carry all four alpha
rows, both conditions and the baseline together. Report a 95% interval for the
primary difference and pre/post changes. Analyze clean accuracy once per unique
question. Report per-alpha scores descriptively, without selecting whichever
alpha gives a favorable result. Current `aggregate` reports means and standard
deviations across training seeds only; at one seed the latter is unavailable
(`src/gasp/evaluation.py:60–131`). It does not implement question-level intervals.

The independent sampling units are approximately 100 questions, not 400
unrelated attacks. Four prefixes from the same question and clean trace can be
strongly correlated. With 100 questions, differences of a few percentage points
may remain unresolved; pairing can help, but its benefit depends on the observed
question-level disagreements. Do not advertise a minimum detectable effect
without calculating it from justified assumptions.

These intervals describe question-sampling uncertainty conditional on this
trained pair and reconstruction procedure. They do not estimate training-seed
variance or robustness to a different corruption generator. The score on the
exact frozen finite corpus is itself known once evaluated. One paired seed is
acceptable for exploration, but even an interval excluding zero is not evidence
that the ordering will persist across training seeds. An interval containing zero
is not equivalence. A preservation claim likewise requires a predeclared tolerated
clean-accuracy loss and an interval sufficiently narrow to exclude larger damage;
“no significant drop” alone is insufficient.

**Minimum revision 4 — separate useful training exposure from successful learning**

Keep G=16 and the current matched loss/optimizer design for this first comparison.
The 100-question sampler visits each question once, so each role has just 50
active contexts and 800 trained GRPO trajectories per condition, plus at most
50 agent repair targets. The agent's scoring of polluter candidates is additional
inference, not another 800 agent-training trajectories. This is a small learning
budget, and increasing a nominal pool beyond the number of visits would not help.
See `src/gasp/trainer.py:134–140,191–259` and the cycle report, lines 47–51.

The paper explicitly motivates guidance by rare successful recoveries (§3.4,
p.4; Appendix A.1 and A.7, pp.11,14–15). For independent binary-reward rollouts,
the probability of a mixed group is 1 − p^G − (1−p)^G. At recovery probability
0.01 this is about 15% for G=16 versus 47% for G=64; at 0.05 it is about 56%
versus 96%. These are illustrations, not estimates of this experiment's recovery
rate. The observed 9/10 mixed diagnostic groups on four repeated questions do not
establish adequate signal over the new training pool, or learning.

Record, separately by role and block, mixed-group counts, success counts within
groups, all-zero/all-one groups, clean-trace retries, capped completions, empty
pollutions, nonempty repair targets and their lengths, gradient norms/clipping,
and actual generated/trained tokens. Preserve a deterministic, time-spaced sample
of pollution and repair texts from each condition for offline semantic inspection.
The logs already retain much of this information. Nonempty repairs need to correct
the local mistake; they are not automatically useful supervision. The sequence-sum
repair loss can also be large for long targets; inspect lengths and contributions
without silently changing the declared objective (`trainer.py:202–219`).

Use a **fresh 100-global-update cosine schedule**, ten warmup updates and the
declared final-ten-agent-update guidance decay. Do not resume the timing run or
carry its 200-update schedule into this experiment (`trainer.py:35–41,71–72`;
`config.py:89–95`). The resulting cumulative LR multipliers are approximately
25.503 for agent updates and 25.497 for polluter updates, so there is no material
aggregate LR imbalance from the five-step blocks. This does not establish that
either role has reached a useful plateau.

The endpoint at global update 100 follows a polluter block. That is a legitimate,
predeclared end-of-cycle measurement, but the shared agent has just received
polluter gradients whereas the split agent has not changed since update 95.
The final five LR multipliers are only approximately .00760, .00487, .00274,
.00122 and .00030 of the 3e-6 peak. Consequently a phase diagnostic is **optional**,
not worth sacrificing the primary corpus or paired training budget. If desired,
predeclare 20 question IDs and evaluate their 80 corrupted contexts at update 95
for unified, then reuse the corresponding update-100 outputs; the split agent's
unchanged weights can be hash-verified. This adds only 80 answers and no new GPT
calls. It is a secondary phase-sensitivity check and must not select the endpoint.

**Acceptance and interpretation**

There are three distinct standards; none guarantees the next one.

1. **Valid execution:** the same frozen data/config/source, initialization,
   question/alpha schedule, finite updates, role isolation and allocated budget
   are verified; the corpus audit is recorded; both fixed endpoints and the
   baseline are complete. The existing CUDA checks support this standard but do
   not replace corpus validation or full-run accounting.
2. **Evidence that the intended training mechanism ran:** report the actual
   informative groups and useful repair examples across the run. If a role has
   essentially no informative reward groups, guidance is absent/mostly unusable,
   or corruptions are systematically invalid, diagnose that limitation before
   interpreting an architecture comparison. There is no paper-backed universal
   count of informative groups that guarantees learning.
3. **Evidence of recovery improvement or a condition difference:** use the
   predeclared paired endpoint and pre/post changes with their question-level
   intervals, supported by valid examples and the clean-subset check. Distinguish
   clear improvement, suggestive but uncertain improvement, and no demonstrated
   improvement. Do not make statistical significance a condition for reporting
   the experiment, and do not extend it until significance appears.

An inconclusive result remains useful if it accurately bounds the observed
finite-budget effect and identifies training-signal limitations. If neither
condition improves and intervals are wide, the result is primarily evidence
about this low-budget regime and implementation feasibility; it cannot decide
whether policy splitting helps a successfully trained GASP system.

The initial checkpoint and unified condition are the necessary baselines for
the stated question. Adding an unguided ASP run, a fixed learned polluter, a
GPT training polluter or additional backbones is **not required** within this
budget. Those controls would be needed for separate causal claims about repair
guidance, adaptive curriculum or cross-model generality; the paper addresses
such questions with additional baselines (§4.1 and Appendix A.9). They should not
consume the present experiment's evaluation reserve.

After the four minimum revisions, I would proceed with the reduced paired run
if the measured complete pipeline fits the budget. The appropriate result is
“under this one-seed, fixed-budget implementation and frozen recovery test,
split changed agent accuracy by X percentage points, with this question-level
uncertainty and these pre/post/clean results.” It cannot establish replication
of the paper, convergence, equivalence, a seed-stable winner, generalized
diagnosis/reliability improvements, or the isolated effect of weight sharing
independent of optimizer history and capacity.
