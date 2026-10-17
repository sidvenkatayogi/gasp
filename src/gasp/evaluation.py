"""Agent-only evaluation, RNG isolation, cache-identical comparisons."""
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

from .data import write_json
from .models import rng_state, restore_rng, seed_all, Rollouts, model_digest
from .verification import correct, extract_answer


def evaluate(model, tokenizer, config, rows, manifest, output, step, run_manifest):
    saved_rng = rng_state()
    start = time.perf_counter()
    rollout = Rollouts(tokenizer, config)
    predictions = []
    clean_by_question = {}
    try:
        seed_all(config.eval_seed)
        for row in rows:
            qid = row["question_id"]
            if qid not in clean_by_question:
                s = rollout.generate(model, row["clean_prompt_ids"], 1, config.completion_tokens,
                    "eval_clean", temperature=config.eval_temperature)[0]
                clean_by_question[qid] = {"prediction": s.text,
                    "correct": correct(s.text, row["answer"], config.verifier),
                    "parsed_answer": extract_answer(s.text)}
            s = rollout.generate(model, row["corrupt_prompt_ids"], 1, config.completion_tokens,
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
            "counts": rollout.counts, "wall_seconds": time.perf_counter()-start}
        directory = Path(output)/"eval"/f"step-{step:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory/"predictions.jsonl").write_text("".join(json.dumps(p)+"\n" for p in predictions))
        write_json(directory/"metrics.json", result)
        return result
    finally:
        restore_rng(saved_rng)


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
