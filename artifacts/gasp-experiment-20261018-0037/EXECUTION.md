# Approved exploratory execution

Source and budget are frozen in `experiment.json` and `source.zip`. User approved
the paper-reviewed plan, about ten total GPU-hours with cost priority. Target is
50 updates per role / ten cycles in each condition, subject to measured shared
preparation/evaluation costs before freezing the final paired schedule.

Local validation: 133 passed, four CUDA-only skips. No secrets are uploaded.
Initial preparation attempt was preempted before generation; its recorded worker
time was 9.07 seconds. Saved budget was recovered and the same source retried.

The first bounded preparation completed 768 candidates / 58 retained training
questions in 3,375.19 worker seconds, then released its GPU. A continuation landed
on A100 PCIe rather than the checkpoint's A100-SXM4 and was rejected by the existing
runtime identity check after 6.18 seconds, before generation. An attempted AWS
placement was rejected at Function creation (`Pinning cloud aws not supported`),
before any GPU allocation; its launch record is preserved as creation-rejected
with zero GPU charge. Continuation uses normal allocation and retains the hardware
identity guard. Frozen code, checkpoint state and data selection are unchanged.

Modal Volume: `gasp-exploratory-artifacts`, experiment directory
`gasp-experiment-20261018-0037`. Only JSON/JSONL/log outputs download locally;
preparation RNG checkpoints and training checkpoints persist remotely.

Order:

1. Prepare 100 retained MATH and 100 retained GSM8K questions with K=4 filtering;
   16 questions / 64 samples per offline batch, explicit candidate/time limits.
2. With GPU released, run local `generate_corpus.py`, initially four requests,
   then reuse those responses when completing all 400. No automatic paid retries.
3. Apply `corpus-audit-policy.json` before trained-model scores exist; write the
   semantic review with the exact corpus hash and reviewed IDs.
4. Freeze equal whole-cycle training allocation using measured preparation cost
   and a conservative evaluation reserve before launching the shared baseline;
   the baseline stage binds that schedule. Use its measured throughput to check
   the remaining training/evaluation reservations. Baseline plus both endpoints
   and shape-matched initial spot-checks total 1,564 generated answers.
5. Train unified and split sequentially from the same initial checkpoint and
   source. Persist checkpoints, stream updates, evaluate the fixed endpoint.
6. Run `scripts/analyze_exploratory.py` on CPU and report paired question-level
   intervals, gains over initialization, selected-subset clean retention, signal
   quality and actual resource use. Confirm no GPU containers remain.

Use the durable `gpu-budget.json` and attempt reports for actual progress/spend;
this file describes the protocol, not a claim of completed stages.

## Network interruption recovery

The attempted in-place detach did not change the app lifecycle. A cloud
heartbeat returned success, but did not prevent cancellation after the local
client stopped heartbeating. The worker received cancellation at 02:25:43 UTC,
committed its report/ledger at 02:25:44, and the app was marked stopped at
02:28:41. The earlier claim of disconnect protection was incorrect.

Recovered: all 100 training questions and 22 evaluation questions / 176 evaluated
candidates. The interrupted evaluation batch is replayed from its prior RNG
checkpoint under the existing resume protocol. Worker time was 2939.195 seconds;
charged worker time plus the existing lifecycle allowance is 2999.195 seconds.
Keep an additional conservative 180-second planning reserve for the uncertain
cancellation-to-app-stop tail; this is an allowance, not measured GPU activity.

Subsequent stages use `run_detached.py`, a local launcher adapter that passes
`detach=True` to Modal App.run at creation. It leaves the frozen GPU worker,
algorithm, data, time limits and budget code unchanged. Verify detached lifecycle
by server read-back. Do not use another heartbeat workaround.

## Source quality correction before baseline

All 100 clean traces were audited before any model evaluation. Eleven definite
intermediate errors and one ambiguous claim triggered the documented fresh-trace
redraw policy. Original caches and API responses are preserved. A helper import
failed before its worker began; app ap-R3lGJEB47psVKf3qFaeepI was stopped and the
full 660-second reservation conservatively charged. The local/remote import path
was corrected and checked before retrying. Both training arms are reduced to
35 updates per role / seven cycles before scores; see the revised budget record.

Corpus construction and audit complete: primary100 questions/400 corruptions;
strict quality sensitivity95 questions / 380 corruptions. Six disclosed instruction
misses remain after capped same-prompt retries. All 400 have an identified math
error. See corpus-v4/corpus-review.json and final-corpus-adjudication-policy.json.
The sensitivity helper checks full-set agreement with the tested primary analysis
before applying the frozen exclusions. Saved API usage:427 unique responses,
330,414 input tokens,29,858 output tokens; one earlier in-flight response may be unsaved.
Only preparation has used GPUs so far; all are released before the baseline.
