"""Summarize archived cycle measurements; never sum nested phase timers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def summarize(condition):
    root = ROOT / condition / condition
    report = json.loads((root / "report.json").read_text())
    if not report["success"] or report["completed_updates"] != 11:
        raise ValueError("A complete successful cycle plus return update is required")
    events = report["phase_events"]
    def phase(name, step=None):
        return sum(e["seconds"] for e in events if e["phase"] == name
                   and (step is None or e["step"] == step))
    steps = [e for e in events if e["phase"] == "training_update" and e["step"] <= 10]
    history = [json.loads(line) for line in (root / "run/training.jsonl").read_text().splitlines()]
    manifest = json.loads((root / "run/run.json").read_text())
    quality = {}
    for role in ("agent", "polluter"):
        contexts = [c for r in history if r["step"] <= 10 and r["role"] == role
                    for c in r["contexts"]]
        quality[role] = {
            "groups": len(contexts),
            "mixed_reward_groups": sum(c["informative_samples"] > 0 for c in contexts),
            "fresh_clean_attempts": sum(c["clean_attempts"] for c in contexts),
            "pollutions": sum(len(c["polluted"]) for c in contexts),
            "exact_pollution_tags": sum(sum(c["polluted_tag_present"]) for c in contexts),
            "empty_pollutions": sum(not w.strip() for c in contexts for w in c["polluted"]),
        }
        if role == "agent":
            quality[role]["nonempty_repairs"] = sum(bool(c["repair_target_ids"]) for c in contexts)
    cycle = sum(e["seconds"] for e in steps)
    # Resume places both Adam moment sets on CPU. Step 11 therefore measures
    # reload only, not a full steady exchange. Estimate the latter from separately
    # observed eviction/reload components; do not call it directly measured.
    transfer_estimate = phase("optimizer_transfer", 6) + phase("optimizer_transfer", 11)
    steady_cycle = cycle
    if condition == "split":
        steady_cycle += 2 * transfer_estimate - phase("optimizer_transfer", 6)
    result = {
        "gpu": manifest["gpu"],
        "cycle_seconds": cycle,
        "agent_block_seconds": sum(e["seconds"] for e in steps if e["step"] <= 5),
        "polluter_block_seconds": sum(e["seconds"] for e in steps if e["step"] > 5),
        "checkpoint_seconds": phase("checkpoint"),
        "checkpoint_bytes": report["checkpoint_bytes"],
        "resume_seconds": phase("resume"),
        "return_update_seconds": phase("training_update", 11),
        "steady_cycle_seconds_estimate": steady_cycle,
        "split_exchange_seconds_component_estimate": transfer_estimate if condition == "split" else None,
        "worker_seconds": json.loads((ROOT / condition / "worker.json").read_text())["seconds"],
        "peak_gpu_allocated_bytes": report["telemetry"]["peak_gpu_allocated_bytes"],
        "peak_gpu_reserved_bytes": report["telemetry"]["peak_gpu_reserved_bytes"],
        "peak_process_rss_bytes": report["telemetry"]["peak_process_rss_bytes"],
        "quality_first_cycle": quality,
        "manifest": {k: manifest[k] for k in (
            "initial_model_sha256", "paired_config_sha256", "train_sha256",
            "eval_sha256", "source_sha256", "tokenizer_sha256")},
        "question_alpha_schedule": [(r["role"], c["question_id"], c["alpha"])
                                     for r in history for c in r["contexts"]],
    }
    return result


results = {condition: summarize(condition) for condition in ("unified", "split")}
assert results["unified"]["manifest"] == results["split"]["manifest"]
assert results["unified"]["question_alpha_schedule"] == results["split"]["question_alpha_schedule"]
assert results["unified"]["gpu"] == results["split"]["gpu"]
projection = {}
for cycles in (10, 12, 15, 20):
    raw_hours = cycles * sum(r["steady_cycle_seconds_estimate"] + r["checkpoint_seconds"]
                            for r in results.values()) / 3600
    projection[str(cycles)] = {
        "updates_per_role_per_condition": 5 * cycles,
        "distinct_questions_for_one_pass": 10 * cycles,
        "paired_training_and_checkpoint_hours_raw": raw_hours,
        "paired_training_and_checkpoint_hours_with_20_percent_allowance": 1.2 * raw_hours,
    }
summary = {"conditions": results, "paired_identity_checks_passed": True,
           "projections": projection,
           "limitations": [
               "One cycle per condition on four repeated questions; no statistical runtime confidence interval.",
               "Twenty-percent allowance is judgmental, not a measured bound.",
               "Projections exclude data preparation, API time, full evaluation and engineering.",
               "Projections assume one checkpoint per cycle and no interruptions/resumes.",
               "The diagnostic used a 2048 input cap, a declared departure from the paper's 1024.",
               "Default data preparation/evaluation are not yet optimized for the proposed ten-hour pipeline.",
           ]}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
