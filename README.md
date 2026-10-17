# GASP: shared versus separate policy parameters

Implementation of [Learning Robust Reasoning through Guided Adversarial Self-Play](https://arxiv.org/html/2602.00173v1).
The split condition initializes two identical policies and trains them independently.
The unified condition shares the same parameters between roles. Both use the same
training code, prompts, rewards and stated paper hyperparameters.

The primary experiment compares **our unified run with our split run** on the
same GPU setup, shared data and evaluation protocol. It measures the effect of
separating the policies within this implementation. The paper's published scores
are background context; they are not the baseline being subtracted from our results.

Read [the implementation contract](docs/PLAN.md), [fidelity details](docs/EXPERIMENT.md),
the [shared assumptions and comparison protocol](docs/ASSUMPTIONS.md),
the [independent implementation audit](docs/AUDIT.md), and
[validation results](artifacts/VALIDATION.md). Several numerical settings are
absent from the paper; they remain explicit assumptions, not verified author defaults.

## Run the offline smoke

Python 3.11–3.13. The smoke uses a tiny random Qwen2 and binary tasks on CPU; it
checks training mechanics, not reasoning performance.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-colab.txt
python -m pip install --no-deps .
python -m pytest -q
python -m gasp smoke --output runs/smoke-v3
```

## Prepare the experiment

The default profile is `configs/full.json`: full tuning of DeepSeek-R1-Distill-Qwen-1.5B,
G=64, 1024-token prompts/completions, five agent then five polluter updates.
`configs/pilot.json` only shortens the run and reporting intervals; it uses the same
algorithm, full tuning, group size, and generation limits. It is not a low-memory
LoRA substitute. Both profiles offload the inactive policy's AdamW moments to CPU
RAM, retaining active moments on GPU across a five-update block. Both models'
weights remain on GPU. A full-model synthetic probe with accumulated gradients
peaked at 36.96 GiB allocated on a 40 GiB A100 without reducing precision. An
end-to-end pilot must still establish training peaks and throughput before a
campaign; see [validation results](artifacts/VALIDATION.md).

```bash
python -m gasp download --output data/raw
python -m gasp prepare-train --config configs/full.json \
  --source data/raw/math_raw.jsonl --output data/full-v4/train.jsonl --candidates 7500
```

Preparation resolves the model revision and keeps MATH train questions with four
correct initial-checkpoint samples at T=0.7. Training generates fresh correct
current-agent traces by rejection sampling. After 32 unsuccessful attempts it stops
without updating weights; it never substitutes cached initial traces. Retry costs
are recorded and can differ between conditions.

Role prompts preserve the shared reasoning prefix and the generated replacement
verbatim. If any role exceeds 1024 prompt tokens, training stops with an explicit
error. The paper does not specify how to handle this overflow; silently trimming
each role differently changes the episode, and raising the cap changes its settings.

Recoverability evaluation uses one **held-out GPT-6.1 Sol corruption corpus**, as
requested. This is a declared change from the paper's GPT-5 evaluator, shared by
both conditions. Follow [the corpus construction procedure](docs/CORPUS.md) to
select clean-solved GSM8K questions on the GPU and generate corruptions through the
API. After generation and inspection, import it:

```bash
python -m gasp import-eval --config data/full-v4/resolved_config.json \
  --source data/full-v4/held-out-corruptions.jsonl \
  --provenance data/full-v4/held-out-corruptions.jsonl.provenance.json \
  --output data/full-v4/eval.jsonl
```

`prepare-eval` is an alias for this import and also requires `--provenance`.
The [corpus format](docs/EXPERIMENT.md#evaluation-input) specifies required fields.
Only `generate-eval` makes paid API calls and requires `OPENAI_API_KEY`.
Missing or empty corruptions produce an error. Both conditions use the same imported
contexts and clean-solved subset; its exact selection must be recorded.

## Train and compare

```bash
python -m gasp train --config data/full-v4/resolved_config.json --condition unified --seed 42 \
  --train-cache data/full-v4/train.jsonl --eval-cache data/full-v4/eval.jsonl --output runs/full-v3/unified-42
python -m gasp train --config data/full-v4/resolved_config.json --condition split --seed 42 \
  --train-cache data/full-v4/train.jsonl --eval-cache data/full-v4/eval.jsonl --output runs/full-v3/split-42
```

Repeat for seeds 52 and 62, then pass all six run paths to `python -m gasp aggregate
--runs ... --output runs/full-v3/comparison`. For the short pilot, use its resolved
config and add `--allow-pilot` when aggregating the paired seed-42 runs.

Resume with the same config and total budget by adding `--resume PATH/checkpoint.pt`.
`--stop-after N` stops at an absolute global step without altering schedules.
Checkpoints include policy/optimizer/scheduler states, the unified opponent snapshot,
RNG state and data position. **Pre-fix checkpoints and caches must not be resumed or
mixed with new results.** Use fresh `v3` directories; the checkpoint schema and
prompt-source checks reject the old protocol.

```bash
python -m gasp evaluate --run runs/full-v3/split-42 \
  --train-cache data/full-v4/train.jsonl --eval-cache data/full-v4/eval.jsonl
python -m gasp export-agent --run runs/full-v3/split-42 --output runs/deployed-agent
```

The optimizer holds one AdamW state per unique parameter. Shared weights therefore
share moments; separate weights have separate moments. A global cosine/warmup
schedule is common to both conditions. Only unified needs an opponent snapshot;
split uses its inactive policy directly under no-grad inference.

## Colab / Jupyter

For a CPU smoke through the authenticated Colab CLI:

```bash
python scripts/package_colab.py
colab --auth=oauth2 new -s gasp-smoke
colab --auth=oauth2 upload -s gasp-smoke dist/gasp-colab.zip gasp-colab.zip
colab --auth=oauth2 exec -s gasp-smoke -f scripts/colab_smoke.py --timeout 1200
colab --auth=oauth2 download -s gasp-smoke content/gasp-smoke-results.zip artifacts/colab-smoke-results.zip
colab --auth=oauth2 stop -s gasp-smoke
```

The runner installs CPU dependencies in an isolated environment, runs the tests,
and trains both tiny smoke conditions. Inspect `report.json` inside the downloaded
archive for `success: true`; Colab CLI 0.6.0 can exit zero after a remote Python error.
Stop the runtime after collecting results, including when setup or testing fails.

Open `notebooks/00_plan_and_smoke.ipynb` and upload the rebuilt `dist/gasp-colab.zip`
when prompted. Continue with notebooks 01–03 for preparation, training and results.
The notebooks default to the full profile and display training commands before the
run flag is enabled. Notebook 01 supports Sol corpus generation with an explicit
run flag, or imports existing held-out examples. Notebook 04 contains optional
secondary benchmark adapters, whose protocol limits are documented separately.

The local tests do not establish target-model GPU memory fit or scientific results.
No full campaign has been launched and no real evaluation corpus is bundled.
