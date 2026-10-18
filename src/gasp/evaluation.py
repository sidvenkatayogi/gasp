"""Agent-only evaluation, RNG isolation, cache-identical comparisons."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from .data import write_json
from .models import rng_state, restore_rng, seed_all, Rollouts, model_digest
from .verification import correct, extract_answer


def _prompt_digest(rows):
    return hashlib.sha256(json.dumps([
        [r["id"], r["question_id"], r["clean_prompt_ids"], r["corrupt_prompt_ids"]]
        for r in rows], separators=(",", ":")).encode()).hexdigest()


def _unique_questions(rows):
    unique = {}
    for row in rows:
        qid = row["question_id"]
        if qid in unique and any(row[k] != unique[qid][k]
                                 for k in ("question", "answer", "clean_prompt_ids")):
            raise ValueError("Inconsistent clean question: " + str(qid))
        unique.setdefault(qid, row)
    return list(unique.values())


def _batched_answers(rollout, model, rows, key, purpose):
    size = rollout.cfg.rollout_batch_size
    for start in range(0, len(rows), size):
        batch = rows[start:start + size]
        started = time.perf_counter()
        groups = rollout.generate_many(model, [r[key] for r in batch], 1,
            rollout.cfg.completion_tokens, purpose, temperature=rollout.cfg.eval_temperature)
        if len(groups) != len(batch) or any(len(group) != 1 for group in groups):
            raise RuntimeError("Evaluation requires exactly one answer per prompt")
        print(json.dumps({"phase": purpose, "completed": start + len(batch),
            "total": len(rows), "batch_seconds": time.perf_counter() - started}), flush=True)
        yield from (group[0] for group in groups)


def evaluate(model, tokenizer, config, rows, manifest, output, step, run_manifest):
    if not rows:
        raise ValueError("Evaluation requires nonempty rows")
    saved_rng = rng_state()
    start = time.perf_counter()
    rollout = Rollouts(tokenizer, config)
    predictions = []
    clean_by_question = {}
    try:
        seed_all(config.eval_seed)
        if config.rollout_batch_size > 1:
            clean_rows = _unique_questions(rows)
            for row, sample in zip(clean_rows, _batched_answers(rollout, model, clean_rows,
                    "clean_prompt_ids", "eval_clean")):
                clean_by_question[row["question_id"]] = {"prediction": sample.text,
                    "correct": correct(sample.text, row["answer"], config.verifier),
                    "parsed_answer": extract_answer(sample.text)}
            corrupted = iter(_batched_answers(rollout, model, rows,
                                             "corrupt_prompt_ids", "eval_corrupted"))
        for row in rows:
            qid = row["question_id"]
            if qid not in clean_by_question:
                s = rollout.generate(model, row["clean_prompt_ids"], 1, config.completion_tokens,
                    "eval_clean", temperature=config.eval_temperature)[0]
                clean_by_question[qid] = {"prediction": s.text,
                    "correct": correct(s.text, row["answer"], config.verifier),
                    "parsed_answer": extract_answer(s.text)}
            s = next(corrupted) if config.rollout_batch_size > 1 else rollout.generate(
                model, row["corrupt_prompt_ids"], 1, config.completion_tokens,
                "eval_corrupted", temperature=config.eval_temperature)[0]
            predictions.append({"id": row["id"], "question_id": qid, "question": row["question"],
                "answer": row["answer"], "prediction": s.text, "parsed_answer": extract_answer(s.text),
                "correct": correct(s.text, row["answer"], config.verifier),
                "clean": clean_by_question[qid], "initial_clean_solved": row["initial_clean_solved"],
                "eval_sha256": manifest["sha256"], "alpha": row.get("alpha"),
                "generator": row.get("generator", "imported")})
        subset = [p["correct"] for p in predictions if p["initial_clean_solved"]]
        metrics = {"corrupted_initial_clean_solved": statistics.mean(subset) if subset else None,
            "corrupted_full": statistics.mean(p["correct"] for p in predictions),
            "clean_full": statistics.mean(p["correct"] for p in clean_by_question.values())}
        result = {"step": step, "condition": config.condition, "seed": config.seed, "label": config.label,
            "metrics": metrics, "primary_metric": "corrupted_initial_clean_solved",
            "n_corrupted": len(predictions), "n_subset": len(subset),
            "n_clean": len(clean_by_question), "eval_sha256": manifest["sha256"],
            "train_sha256": run_manifest["train_sha256"], "protocol": manifest["provenance"]["protocol"],
            "paired_config_sha256": config.paired_hash(), "config": asdict(config),
            "model_sha256": model_digest(model), "source_sha256": run_manifest["source_sha256"],
            "tokenizer_sha256": run_manifest["tokenizer_sha256"], "packages": run_manifest["packages"],
            "gpu": run_manifest["gpu"],
            "prompt_ids_sha256": _prompt_digest(rows),
            "counts": rollout.counts, "wall_seconds": time.perf_counter()-start}
        directory = Path(output)/"eval"/f"step-{step:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        prediction_text = "".join(json.dumps(p)+"\n" for p in predictions)
        (directory/"predictions.jsonl").write_text(prediction_text)
        result["predictions_sha256"] = hashlib.sha256(prediction_text.encode()).hexdigest()
        write_json(directory/"metrics.json", result)
        return result
    finally:
        restore_rng(saved_rng)


def _read_evaluation(directory):
    directory = Path(directory)
    metrics = json.loads((directory/"metrics.json").read_text())
    prediction_bytes = (directory/"predictions.jsonl").read_bytes()
    if hashlib.sha256(prediction_bytes).hexdigest() != metrics["predictions_sha256"]:
        raise ValueError("Prediction artifact hash mismatch")
    predictions = [json.loads(line) for line in prediction_bytes.splitlines()]
    if not predictions or len({p["id"] for p in predictions}) != len(predictions):
        raise ValueError("Empty or duplicate prediction IDs")
    if any(p["eval_sha256"] != metrics["eval_sha256"] for p in predictions):
        raise ValueError("Prediction corpus hash mismatch")
    return metrics, predictions


def verify_shared_baseline(model, tokenizer, config, rows, manifest, run_manifest,
                           baseline_directory, output, spotcheck_batches=1):
    """Verify one reused baseline; generate only matching initial batch shapes.

    This creates a reference/verification record, not a second full evaluation.
    It must run before the first training update, with the final frozen config.
    """
    if config.eval_temperature != 0 or spotcheck_batches < 1:
        raise ValueError("Shared baseline requires greedy decoding and positive spot-check batches")
    started = time.perf_counter()
    baseline, predictions = _read_evaluation(baseline_directory)
    expected = {"step": 0, "paired_config_sha256": config.paired_hash(),
        "eval_sha256": manifest["sha256"], "protocol": manifest["provenance"]["protocol"],
        "prompt_ids_sha256": _prompt_digest(rows), "model_sha256": model_digest(model),
        **{k: run_manifest[k] for k in ("train_sha256", "source_sha256",
                                        "tokenizer_sha256", "packages", "gpu")}}
    for key, value in expected.items():
        if baseline[key] != value:
            raise ValueError("Shared baseline identity mismatch: " + key)
    if expected["model_sha256"] != run_manifest["initial_model_sha256"]:
        raise ValueError("Shared baseline verification must precede training")
    if [p["id"] for p in predictions] != [r["id"] for r in rows]:
        raise ValueError("Shared baseline prediction IDs/order differ")
    saved_rng = rng_state()
    rollout = Rollouts(tokenizer, config)
    limit = config.rollout_batch_size * spotcheck_batches
    clean_rows, corrupt_rows = _unique_questions(rows)[:limit], rows[:limit]
    by_id = {p["id"]: p for p in predictions}
    try:
        seed_all(config.eval_seed)
        for key, purpose, chosen in (("clean_prompt_ids", "eval_clean", clean_rows),
                                     ("corrupt_prompt_ids", "eval_corrupted", corrupt_rows)):
            for row, sample in zip(chosen, _batched_answers(rollout, model, chosen, key, purpose)):
                expected_prediction = by_id[row["id"]]
                if purpose == "eval_clean":
                    expected_prediction = expected_prediction["clean"]
                if sample.text != expected_prediction["prediction"]:
                    raise ValueError("Shared baseline spot-check prediction mismatch: " + row["id"])
    finally:
        restore_rng(saved_rng)
    result = {"shared_baseline": str(baseline_directory),
        "baseline_metrics_sha256": hashlib.sha256(
            (Path(baseline_directory)/"metrics.json").read_bytes()).hexdigest(),
        "predictions_sha256": baseline["predictions_sha256"], "identity": expected,
        "condition": config.condition, "clean_question_ids": [r["question_id"] for r in clean_rows],
        "corrupted_ids": [r["id"] for r in corrupt_rows], "counts": rollout.counts,
        "wall_seconds": time.perf_counter() - started, "passed": True,
        "note": "Shared complete baseline reused; only listed initial batches regenerated."}
    Path(output).mkdir(parents=True, exist_ok=True)
    write_json(Path(output)/"baseline-verification.json", result)
    return result


def analyze_paired(baseline_directory, unified_directory, split_directory, output,
                   samples=10000, seed=10002):
    """One-seed endpoint comparison, paired bootstrap over whole questions.

    Each question carries its four alpha rows, clean answer, and all three
    models through the same resample. Intervals are conditional on this seed
    and frozen corruption corpus, not training-seed confidence intervals.
    """
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    artifacts = {name: _read_evaluation(path) for name, path in (
        ("initial", baseline_directory), ("unified", unified_directory), ("split", split_directory))}
    baseline, reference = artifacts["initial"]
    if baseline["step"] != 0:
        raise ValueError("Initial baseline must be at step zero")
    identity_keys = ("paired_config_sha256", "eval_sha256", "train_sha256", "protocol",
                     "source_sha256", "tokenizer_sha256", "packages", "gpu", "prompt_ids_sha256")
    row_keys = ("id", "question_id", "question", "answer", "alpha", "initial_clean_solved", "eval_sha256")
    for name, (metrics, predictions) in artifacts.items():
        for key in identity_keys:
            if metrics[key] != baseline[key]:
                raise ValueError("Unmatched paired results: " + key)
        if metrics["seed"] != baseline["seed"] or {
                k: v for k, v in metrics["config"].items() if k != "condition"} != {
                k: v for k, v in baseline["config"].items() if k != "condition"}:
            raise ValueError("Unmatched paired seed/config")
        if name != "initial" and (metrics["condition"] != name or
                                  metrics["step"] != 2 * metrics["config"]["role_steps"]):
            raise ValueError("Expected the fixed complete " + name + " endpoint")
        if len(predictions) != len(reference) or any(
            any(a[k] != b[k] for k in row_keys) for a, b in zip(reference, predictions)):
            raise ValueError("Evaluation row identities/order differ")
    groups = {}
    for index, row in enumerate(reference):
        if not row["initial_clean_solved"]:
            raise ValueError("Exploratory analysis requires the fixed initial clean-solved subset")
        groups.setdefault(row["question_id"], []).append(index)
    alphas = (0., .25, .5, .75)
    for indices in groups.values():
        if sorted(reference[i]["alpha"] for i in indices) != list(alphas):
            raise ValueError("Each question must carry exactly the four declared alpha rows")
    arrays, scores = {}, {}
    for name, (_, predictions) in artifacts.items():
        clean, corrupted = [], []
        for indices in groups.values():
            if any(predictions[i]["clean"] != predictions[indices[0]]["clean"] for i in indices):
                raise ValueError("Inconsistent repeated clean prediction")
            values = [predictions[i]["correct"] for i in indices]
            clean_value = predictions[indices[0]]["clean"]["correct"]
            if any(value not in (0, 1) for value in values + [clean_value]):
                raise ValueError("Correctness must be binary")
            corrupted.append(statistics.mean(values))
            clean.append(clean_value)
        arrays[name] = {"corrupted_accuracy": np.array(corrupted, dtype=float),
                        "clean_selected_subset_accuracy": np.array(clean, dtype=float)}
        scores[name] = {key: float(value.mean()) for key, value in arrays[name].items()}
        scores[name]["per_alpha_corrupted_accuracy"] = {
            str(alpha): statistics.mean(p["correct"] for p in predictions if p["alpha"] == alpha)
            for alpha in alphas}
    # A local generator leaves training/global analysis RNG untouched. The same
    # question indices are used for every metric and comparison.
    indices = np.random.default_rng(seed).integers(0, len(groups), size=(samples, len(groups)))
    differences = {}
    for left, right in (("split", "unified"), ("unified", "initial"), ("split", "initial")):
        differences[left + "_minus_" + right] = {}
        for metric in arrays[left]:
            deltas = arrays[left][metric] - arrays[right][metric]
            interval = np.quantile(deltas[indices].mean(axis=1), [.025, .975])
            differences[left + "_minus_" + right][metric] = {
                "estimate": float(deltas.mean()), "ci95": interval.tolist()}
    report = {"primary_comparison": "split_minus_unified", "primary_metric": "corrupted_accuracy",
        "scores": scores, "paired_differences": differences, "n_questions": len(groups),
        "n_corruptions": len(reference), "training_seed": baseline["seed"],
        "bootstrap": {"unit": "question", "samples": samples, "seed": seed,
                      "interval": "paired percentile 95%"},
        "units": "accuracy fractions; multiply differences by 100 for percentage points",
        "interpretation": "Question-sampling uncertainty conditional on one training seed and the frozen corpus; "
                          "does not estimate training-seed or regenerated-corruption variability. "
                          "An interval including zero does not establish equivalence.",
        "identity": {key: baseline[key] for key in identity_keys},
        "artifacts": {name: {"model_sha256": metrics["model_sha256"],
                               "predictions_sha256": metrics["predictions_sha256"]}
                      for name, (metrics, _) in artifacts.items()}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"summary.json", report)
    question_scores = [{"question_id": qid, **{name: {key: float(value[index])
        for key, value in metrics.items()} for name, metrics in arrays.items()}}
        for index, qid in enumerate(groups)]
    (output/"question_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in question_scores))
    return report


def aggregate(run_directories, output, expected_seeds=(42,52,62), allow_pilot=False):
    """Never silently aggregate unlike experiments or incomplete seed pairs."""
    results = {}
    for root in map(Path, run_directories):
        run = json.loads((root/"run.json").read_text())
        cfg = run["config"]
        if ("pilot" in cfg["label"] or cfg["tiny"]) and not allow_pilot:
            raise ValueError("Pilot/smoke results require --allow-pilot and cannot establish paper reproduction")
        final = root/"eval"/f"step-{cfg['role_steps']*2:06d}"
        m = json.loads((final/"metrics.json").read_text())
        key = (cfg["seed"],cfg["condition"])
        if key in results:
            raise ValueError("duplicate seed/condition")
        predictions = [json.loads(s) for s in (final/"predictions.jsonl").read_text().splitlines()]
        results[key] = (m, predictions, json.loads((root/"telemetry.json").read_text()))
    seeds = sorted({k[0] for k in results})
    if not allow_pilot and seeds != sorted(expected_seeds):
        raise ValueError(f"Full campaign requires exactly seeds {expected_seeds}")
    shared_protocol = set()
    paired = []
    disagreements = []
    for seed in seeds:
        if (seed,"unified") not in results or (seed,"split") not in results:
            raise ValueError("Missing paired condition for seed " + str(seed))
        u, up, ut = results[(seed,"unified")]
        s, sp, st = results[(seed,"split")]
        for key in ("paired_config_sha256", "eval_sha256", "train_sha256", "protocol", "step", "source_sha256",
                    "tokenizer_sha256", "packages", "gpu"):
            if u[key] != s[key]:
                raise ValueError("Unmatched paired results: " + key)
        # Beyond each seed, budgets/configs may differ only in seed/condition.
        cross = dict(u["config"])
        cross.pop("seed"); cross.pop("condition")
        shared_protocol.add(json.dumps([cross,u["eval_sha256"],u["train_sha256"],u["source_sha256"]],sort_keys=True))
        if [p["id"] for p in up] != [p["id"] for p in sp]:
            raise ValueError("Evaluation IDs/order differ")
        for purpose in set(ut["rollouts"]) | set(st["rollouts"]):
            # Obtaining a correct current-policy trace uses rejection sampling.
            # Its measured cost can differ; GRPO/repair allocations must still match.
            if purpose.endswith("_clean_attempts"):
                continue
            for budget in ("sequences", "allocated_tokens"):
                if ut["rollouts"][purpose][budget] != st["rollouts"][purpose][budget]:
                    raise ValueError(f"Unmatched rollout allocation: {purpose}/{budget}")
        deltas = {k: s["metrics"][k]-v if v is not None and s["metrics"][k] is not None else None
                  for k,v in u["metrics"].items()}
        paired.append({"seed": seed, "unified": u["metrics"], "split": s["metrics"],
                       "split_minus_unified": deltas, "cost_unified": ut, "cost_split": st})
        for a,b in zip(up,sp):
            if a["correct"] != b["correct"]:
                disagreements.append({"seed":seed,"id":a["id"],"question":a["question"],
                    "answer":a["answer"],"unified":a["prediction"],"split":b["prediction"],
                    "split_minus_unified":b["correct"]-a["correct"]})
    if len(shared_protocol) != 1:
        raise ValueError("Cannot pool seeds with different protocols/budgets/data")
    summary = {}
    for metric in paired[0]["unified"]:
        summary[metric] = {}
        for condition in ("unified","split","split_minus_unified"):
            values = [p[condition][metric] for p in paired]
            if any(v is None for v in values):
                summary[metric][condition] = {"n":0,"mean":None,"sample_std":None}
            else:
                summary[metric][condition] = {"n":len(values),"mean":statistics.mean(values),
                    "sample_std":statistics.stdev(values) if len(values)>1 else None}
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    report = {"label":u["label"],"protocol":u["protocol"],"per_seed":paired,"summary":summary,
              "units":"accuracy fractions; multiply by 100 for percentage points",
              "compute_note":"GRPO/repair allocations matched; on-policy clean retries and actual token counts/time may differ."}
    write_json(output/"summary.json",report)
    (output/"disagreements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in disagreements))
    return report
