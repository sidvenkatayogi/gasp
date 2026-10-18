# Short network interruption

The user requested protection for a five-minute internet outage. The active Modal
app is `ap-BKIIccBJPEzpfwUfQCKuiX`, container
`ta-01M56AES2N54795GXQVH6X3BRR`, function
`fu-NVv49OxoQ4q39BGjuL0ofQ`. Existing preparation remains bounded by its original
6,000-second worker timeout and experiment ledger. No new GPU was provisioned.

AppPublish with DETACHED and unchanged function IDs returned successfully but
read-back remained EPHEMERAL. Do not claim that the application was detached.
The saved `disconnect-protection.json` records that unsuccessful state change.

Instead, a cloud heartbeat was started inside the existing container using its
own Modal client identity. Initial and subsequent successful heartbeats are in
`/persist/gasp-experiment-20261018-0037/disconnect-heartbeat.log`. The helper is
`/tmp/gasp-offline-heartbeat.py`, remote PID 22, maximum duration 3,600 seconds from
2026-10-18 02:24:00 UTC. The existing GPU worker's earlier hard deadline still
applies; its single-use container is released when the stage ends. The helper
has no new GPU allocation and no user API credentials were copied to it.

Local Modal runner PID **40752** was suspended with SIGSTOP, not terminated,
after the cloud heartbeat succeeded. Local exec session is **46284**. On return:

1. Read Modal app/container status and the cloud heartbeat log.
2. Resume the local client with `kill -CONT 40752` (after checking PID command).
3. Poll session 46284 / the existing `prepare-continue2-console.log`.
4. Once the local heartbeat/log stream works, stop remote helper PID 22 (only
   after verifying its command) or let it disappear with the single-use worker.
5. If the client cannot recover, retrieve the finished worker report, ledger and
   JSON/JSONL/log artifacts from the durable Volume. Do not rerun a live stage.

Training cache is complete and validated: 100 distinct MATH questions from 1,296
candidates, 89.07 selection minutes. Held-out evaluation context preparation was
still running when the short disconnect was arranged. GPT generation, baseline,
and paired training remain pending. Continue the existing experiment after the
user returns, preserving source/config/budget and corpus-audit requirements.

## Superseded on return, 03:03 UTC

The workaround failed: Modal canceled the input at 02:25:43 despite successful
cloud heartbeat calls. The local client was resumed and exited; old helper and
GPU are gone. Recovery preserved 100 training and 22 evaluation questions.
Do not repeat the heartbeat workaround.

Current supported detached app: `ap-wA0G0TZpWitDaZWCIo5MLf`, started 03:02:48 UTC.
Modal app-list read-back confirms `ephemeral (detached)` and one task. Local
session 52499 runs `run_detached.py`; console is `prepare-detached3-console.log`.
Original frozen worker code is unchanged, with an 4800-second hard limit.
A prior detached allocation landed on PCIe and was rejected before generation;
current GPU is matching NVIDIA A100-SXM4-80GB.
