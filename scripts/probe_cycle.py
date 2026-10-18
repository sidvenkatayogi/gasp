"""Time real HF self-play updates without changing the production algorithm.

Diagnostic pool: four previously filtered MATH questions, reused across steps.
This is not a prefix of the proposed 200-question production experiment.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from gasp.config import Config
from gasp.data import write_cache, write_json, sha256
from gasp.models import load_model, model_digest
from gasp.prompts import Prompts
from gasp.trainer import Trainer


def cycle_config(condition):
    return Config.load("data/full-v4/resolved_config.json").with_changes(
        condition=condition, label="gasp-exploratory-cycle-diagnostic",
        role_steps=100, block_steps=5, accumulation=1, group_size=16,
        eval_every=200, checkpoint_every=10, optimizer_offload=True,
        rollout_batch_size=16, prompt_tokens=2048)


class Timings:
    def __init__(self, output, synchronize):
        self.output = Path(output)
        self.synchronize = synchronize
        self.events = []
        self.step = 0

    def call(self, label, function, *args, **kwargs):
        self.synchronize()
        started = time.perf_counter()
        ok = False
        try:
            result = function(*args, **kwargs)
            ok = True
            return result
        finally:
            self.synchronize()
            event = {"step": self.step, "phase": label, "success": ok,
                     "seconds": time.perf_counter() - started}
            self.events.append(event)
            with self.output.open("a") as f:
                f.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            return self.call(label, original, *args, **kwargs)
        setattr(obj, name, measured)


def run(condition, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = cycle_config(condition)
    cfg.save(output / "config.json")
    timing = Timings(output / "phases.jsonl", torch.cuda.synchronize)
    report = {"success": False, "condition": condition, "completed_updates": 0,
              "notes": ["Real on-policy HF training, batches up to 16, no cached training traces.",
                        "Exploratory input cap 2048 after measured 1035-token overflow; output caps remain 1024.",
                        "Four prior K4-filtered questions reused for timing; not a 200-question experiment.",
                        "One existing GPT corruption times evaluation only; accuracy is not a study result.",
                        "Ten cycle updates plus one return-to-agent update; full 200-update LR schedule."]}
    started = time.perf_counter()
    trainer = None
    try:
        inputs = json.loads(Path("cycle-inputs.json").read_text())
        model, tok = timing.call("load_model", load_model, cfg)
        prompts = Prompts(tok, cfg)
        train = inputs["train"]
        evaluation = dict(inputs["eval"])
        evaluation["clean_prompt_ids"] = prompts.agent(evaluation["question"])
        evaluation["corrupt_prompt_ids"] = prompts.agent(evaluation["question"],
            evaluation["prefix"], evaluation["polluted"])
        provenance = {"config": vars(cfg), "prompt_sha256": sha256("src/gasp/prompts.py")}
        write_cache(output / "train.jsonl", train, provenance | {
            "kind": "train", "protocol": "prior-vllm-K4-filtered-diagnostic-pool",
            "source_sha256": inputs["training_fixture_sha256"]})
        write_cache(output / "eval.jsonl", [evaluation], provenance | inputs["eval_provenance"])
        trainer = timing.call("initialize_trainer", Trainer, cfg,
            output / "train.jsonl", output / "eval.jsonl", output / "run", model, tok)
        timing.wrap(trainer.roles, "refresh", "refresh_snapshot")
        timing.wrap(trainer, "place_optimizer_moments", "optimizer_transfer")
        timing.wrap(trainer, "backward_group", "grpo_backward")
        timing.wrap(trainer.optimizer, "step", "optimizer_step")
        generate = trainer.rollouts.generate
        def measured_generation(model, prompt, n, limit, purpose, **kwargs):
            return timing.call("generate/" + purpose, generate,
                               model, prompt, n, limit, purpose, **kwargs)
        trainer.rollouts.generate = measured_generation
        timing.wrap(trainer.rollouts, "generate_many", "generate_many")
        timing.call("initial_evaluation", trainer.evaluate)
        frozen = None
        for step in range(1, 12):
            timing.step = step
            role = trainer.role
            if trainer.block_position == 0:
                opposite = "polluter" if role == "agent" else "agent"
                frozen = timing.call("opponent_digest_before", model_digest,
                                     trainer.roles.active(opposite))
            entry = timing.call("training_update", trainer.step)
            report["completed_updates"] = step
            report.setdefault("updates", []).append({
                "step": step, "role": role, "lr": entry["lr"], "grad_norm": entry["grad_norm"],
                "question_ids": [c["question_id"] for c in entry["contexts"]],
                "alphas": [c["alpha"] for c in entry["contexts"]],
                "informative_samples": [c["informative_samples"] for c in entry["contexts"]],
                "clean_attempts": [c["clean_attempts"] for c in entry["contexts"]]})
            print(json.dumps({"update_summary": report["updates"][-1]}), flush=True)
            if step in (5, 10, 11):
                after = timing.call("opponent_digest_after", model_digest, trainer.roles.opponent)
                if after != frozen:
                    raise AssertionError("Frozen opponent changed during an active role's block")
            write_json(output / "report.json", report | {"telemetry": trainer.telemetry()})
            if step == 10:
                # Measure the ordinary return before resume offloads BOTH moment
                # sets to CPU. After step ten, trainer.role already names agent.
                # Clearing grads and moving moments changes neither weights nor RNG.
                trainer.optimizer.zero_grad(set_to_none=True)
                timing.call("steady_return_optimizer_transfer", trainer.place_optimizer_moments)
                checkpoint = timing.call("checkpoint", trainer.save)
                report["checkpoint_bytes"] = checkpoint.stat().st_size
                # Measure actual production hash validation and resume, on the same
                # trainer. It restores all RNG/data/schedule state before step 11.
                timing.call("resume", trainer.resume, checkpoint)
                checkpoint.unlink()
        timing.call("final_evaluation", trainer.evaluate)
        report["success"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["phase_events"] = timing.events
        if trainer is not None:
            report["telemetry"] = trainer.telemetry()
        write_json(output / "report.json", report)
    if not report["success"]:
        raise RuntimeError("Cycle failed; see report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=("unified", "split"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.condition, args.output)
