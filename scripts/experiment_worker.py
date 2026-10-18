"""The three bounded GPU stages of the exploratory paired experiment.

API generation and semantic corpus review happen between workers on CPU.
Checkpoints and preparation progress live on the mounted Modal Volume.
"""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from gasp.config import Config
from gasp.data import import_eval, prepare_train, read_cache, sha256, write_json
from gasp.eval_corpus import prepare_eval_contexts
from gasp.models import model_digest
from gasp.trainer import Trainer


def experiment_config(base, condition="unified", role_steps=50):
    return base.with_changes(condition=condition, label="gasp-exploratory-fixed-budget",
        role_steps=role_steps, block_steps=5, accumulation=1, group_size=16,
        preparation_batch_questions=16, rollout_batch_size=16, prompt_tokens=2048,
        completion_tokens=1024, corruption_tokens=1024, repair_tokens=1024,
        eval_every=2 * role_steps, checkpoint_every=10, optimizer_offload=True)


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def corpus_review(path, corpus):
    review = json.loads(Path(path).read_text())
    if review.get("corpus_sha256") != sha256(corpus):
        raise ValueError("Semantic review belongs to a different corpus")
    if review.get("verdict") != "pass":
        raise ValueError("Corpus review must record verdict=pass before GPU evaluation/training")
    if not review.get("scope") or not review.get("sample_ids"):
        raise ValueError("Corpus review must record scope and reviewed sample_ids")
    return review


def cached_status(path, kind):
    path = Path(path)
    if path.exists():
        _, manifest = read_cache(path, kind)
        return {"complete": True, **manifest}
    return None


def timed(label, function, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.monotonic()
    try:
        return function(*args, **kwargs)
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        emit("phase", phase=label, seconds=time.monotonic() - started)


def prepare(root, cfg, deadline, commit, resume):
    directory = root / "prepare"
    directory.mkdir(exist_ok=True)
    cfg.save(directory / "resolved_config.json")
    results = {}
    for name, kind, source, function, cap in (
        ("train", "train", "math_raw", prepare_train, 2000),
        ("eval-contexts", "eval_contexts", "gsm_raw", prepare_eval_contexts, 1000),
    ):
        path = directory / (name + ".jsonl")
        existing = cached_status(path, kind)
        if existing is not None:
            results[name] = existing
            continue
        remaining = deadline - time.monotonic() - 180
        if remaining <= 0:
            results[name] = {"complete": False, "reason": "stage_budget"}
            break
        has_progress = Path(str(path) + ".progress.pt").exists()
        if has_progress and not resume:
            raise ValueError("Preparation progress exists; pass --resume")
        result = timed("prepare_" + name, function, cfg, "data/raw/" + source + ".jsonl",
            path, retained_target=100, max_candidates=cap, max_seconds=remaining,
            resume=has_progress)
        results[name] = result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        commit()
        emit("preparation_status", dataset=name, result=result)
        if result.get("complete") is False:
            break
    return {"complete": len(results) == 2 and all(r.get("complete", True) for r in results.values()),
            "datasets": results}


def paired_plan(root, cfg, freeze=False):
    path = root / "paired-plan.json"
    value = {"config": asdict(cfg), "paired_config_sha256": cfg.paired_hash(),
             "endpoint": 2 * cfg.role_steps, "analysis_seed": 10002,
             "bootstrap_samples": 10000, "training_seed": cfg.seed}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != value:
            raise ValueError("Paired schedule is already frozen; both conditions require identical allocation")
    elif freeze:
        write_json(path, value)
    else:
        raise FileNotFoundError("Run the shared baseline to freeze the paired schedule first")
    return value


def baseline(root, cfg, inputs, deadline, commit):
    paired_plan(root, cfg, freeze=True)
    directory = root / "baseline"
    if (directory / "eval/step-000000/metrics.json").exists():
        raise FileExistsError("The shared baseline is already complete")
    for name, kind in (("train", "train"), ("eval-contexts", "eval_contexts")):
        read_cache(root / "prepare" / (name + ".jsonl"), kind)
    corpus = inputs / "held-out-corruptions.jsonl"
    review = corpus_review(inputs / "corpus-review.json", corpus)
    contexts, context_manifest = read_cache(root / "prepare/eval-contexts.jsonl", "eval_contexts")
    raw = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    if [r["id"] for r in raw] != [r["id"] for r in contexts]:
        raise ValueError("GPT corpus IDs/order differ from the frozen evaluation contexts")
    for actual, expected in zip(raw, contexts):
        for key in ("question_id", "question", "answer", "split", "alpha", "prefix", "clean",
                    "initial_clean_solved"):
            if actual.get(key) != expected.get(key):
                raise ValueError("GPT corpus changed a frozen context field: " + key)
    provenance = json.loads((inputs / "held-out-corruptions.jsonl.provenance.json").read_text())
    if provenance.get("contexts_sha256") != context_manifest["sha256"]:
        raise ValueError("GPT corpus provenance differs from the prepared contexts")
    cache = root / "prepare/eval.jsonl"
    if not cache.exists():
        timed("import_eval", import_eval, cfg, corpus, cache,
              inputs / "held-out-corruptions.jsonl.provenance.json")
    else:
        _, manifest = read_cache(cache, "eval")
        if manifest["provenance"]["input_sha256"] != sha256(corpus):
            raise ValueError("Imported evaluation cache belongs to a different corpus")
    write_json(root / "corpus-review.json", review)
    commit()
    trainer = timed("initialize_baseline", Trainer, cfg, root / "prepare/train.jsonl", cache, directory)
    if time.monotonic() >= deadline - 180:
        raise TimeoutError("Stage has no evaluation budget left")
    result = timed("shared_initial_evaluation", trainer.evaluate)
    write_json(directory / "telemetry.json", trainer.telemetry())
    commit()
    return {"complete": True, "evaluation": result,
            "initial_model_sha256": trainer.run_manifest["initial_model_sha256"]}


def train(root, cfg, deadline, commit, resume):
    from gasp.evaluation import verify_shared_baseline
    paired_plan(root, cfg.with_changes(condition="unified"))
    directory = root / "train" / cfg.condition
    checkpoint = directory / "checkpoint.pt"
    if (directory / "run.json").exists() and not resume:
        raise FileExistsError("Condition already has output; pass --resume to restore its checkpoint")
    if resume and not checkpoint.exists():
        raise FileNotFoundError("No durable checkpoint exists for this condition")
    trainer = timed("initialize_trainer", Trainer, cfg, root / "prepare/train.jsonl",
                    root / "prepare/eval.jsonl", directory)
    verification = timed("verify_shared_baseline", verify_shared_baseline,
        trainer.roles.agent, trainer.tokenizer, cfg, trainer.eval_rows, trainer.eval_manifest,
        trainer.run_manifest, root / "baseline/eval/step-000000", directory,
        spotcheck_batches=1)
    if resume:
        timed("resume", trainer.resume, checkpoint)
    else:
        trainer.last_evaluated = 0
    commit()
    longest_update = 180.
    endpoint = cfg.role_steps * 2
    last_saved = trainer.global_step if resume else -1
    frozen = None
    while trainer.global_step < endpoint:
        # Leave time to save a recoverable checkpoint. The outer Modal timeout
        # remains the hard watchdog if an individual generation stalls.
        if deadline - time.monotonic() < 2 * longest_update + 180:
            if last_saved != trainer.global_step:
                timed("checkpoint", trainer.save)
                commit()
            return {"complete": False, "reason": "stage_budget", "step": trainer.global_step,
                    "endpoint": endpoint, "telemetry": trainer.telemetry()}
        role = trainer.role
        if trainer.block_position == 0 or frozen is None:
            other = "polluter" if role == "agent" else "agent"
            # On resume inside a block, the stored snapshot is authoritative.
            fixed = trainer.roles.opponent if trainer.block_position else trainer.roles.active(other)
            frozen = model_digest(fixed)
        started = time.monotonic()
        entry = trainer.step()
        seconds = time.monotonic() - started
        longest_update = max(longest_update, seconds)
        emit("training_update", condition=cfg.condition, step=entry["step"], role=role,
             seconds=seconds, lr=entry["lr"], grad_norm=entry["grad_norm"],
             question_ids=[c["question_id"] for c in entry["contexts"]],
             alphas=[c["alpha"] for c in entry["contexts"]],
             informative_samples=[c["informative_samples"] for c in entry["contexts"]],
             clean_attempts=[c["clean_attempts"] for c in entry["contexts"]])
        if trainer.block_position == 0:
            if model_digest(trainer.roles.opponent) != frozen:
                raise AssertionError("Frozen opponent changed during its block")
            emit("role_isolation", step=trainer.global_step, role=role, unchanged=True)
        if trainer.global_step % cfg.checkpoint_every == 0 or trainer.global_step == endpoint:
            timed("checkpoint", trainer.save)
            last_saved = trainer.global_step
            commit()
            emit("checkpoint_committed", step=trainer.global_step, path=str(checkpoint),
                 bytes=checkpoint.stat().st_size)
    if deadline - time.monotonic() < 180:
        return {"complete": False, "reason": "endpoint_evaluation_budget", "step": endpoint}
    evaluation = timed("endpoint_evaluation", trainer.evaluate)
    write_json(directory / "telemetry.json", trainer.telemetry())
    schedule = [{"step": e["step"], "role": e["role"],
                 "question_ids": [c["question_id"] for c in e["contexts"]],
                 "alphas": [c["alpha"] for c in e["contexts"]]} for e in trainer.history]
    schedule_hash = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    summary = {"complete": True, "step": endpoint, "role_counts": trainer.role_counts,
               "schedule_sha256": schedule_hash, "evaluation": evaluation,
               "telemetry": trainer.telemetry(), "baseline_verification": verification}
    write_json(directory / "completion.json", summary)
    other = "split" if cfg.condition == "unified" else "unified"
    other_completion = root / "train" / other / "completion.json"
    if other_completion.exists():
        if json.loads(other_completion.read_text())["schedule_sha256"] != schedule_hash:
            raise AssertionError("Paired question/alpha schedules differ")
    commit()
    return summary


def run(stage, condition, root, inputs, max_seconds, commit, role_steps=50, resume=False):
    root = Path(root)
    deadline = time.monotonic() + max_seconds
    base = Config.load("data/full-v4/resolved_config.json")
    cfg = experiment_config(base, condition or "unified", role_steps)
    if stage == "prepare":
        return prepare(root, cfg, deadline, commit, resume)
    if stage == "baseline":
        return baseline(root, cfg.with_changes(condition="unified"), Path(inputs), deadline, commit)
    if stage == "train" and condition in ("unified", "split"):
        return train(root, cfg, deadline, commit, resume)
    raise ValueError("Unknown experiment stage or training condition")
