"""Bounded diagnostic; writes no production corpus or trained checkpoint.

Run each phase in a separate process so inference memory is released before the
training component measurements. BF16 vLLM is an execution change, not a promise
of identical samples to FP32 master weights with HF autocast.
"""
import argparse
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch

from gasp.config import Config
from gasp.data import read_cache, cut_window
from gasp.models import load_model, Rollouts, Sample, Roles, token_logps
from gasp.prompts import Prompts
from gasp.trainer import Trainer
from gasp.verification import correct, reasoning, pollution, repair_target

OUT = Path("artifacts/vllm-probe")


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


def sources():
    cfg = Config.load("data/full-v4/resolved_config.json")
    gsm, _ = read_cache("data/raw/gsm_raw.jsonl", "raw_eval")
    math, _ = read_cache("data/raw/math_raw.jsonl", "raw_train")
    random.Random(10001).shuffle(gsm)
    random.Random(42).shuffle(math)
    return cfg, gsm, math


def hf():
    cfg, gsm, _ = sources()
    model, tok = load_model(cfg)
    prompts = Prompts(tok, cfg)
    rollout = Rollouts(tok, cfg)
    ids = [prompts.agent(r["question"]) for r in gsm[:4]]
    rollout.generate(model, ids[0], 1, 8, "warmup")
    torch.cuda.synchronize()
    start = time.perf_counter()
    groups = rollout.generate_many(model, ids, 4, 1024, "real_K4")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    greedy = rollout.generate_many(model, ids, 1, 64, "greedy", temperature=0.)
    report = {"seconds": elapsed, "counts": rollout.counts,
              "prompt_ids": ids, "question_ids": [r["id"] for r in gsm[:4]],
              "greedy_tokens": [g[0].tokens for g in greedy],
              "scores": [[correct(s.text, r["answer"], cfg.verifier) for s in g]
                         for r, g in zip(gsm[:4], groups)]}
    save("hf", report)
    print(json.dumps(report), flush=True)


def inference():
    from vllm import LLM, SamplingParams
    cfg, gsm, math = sources()
    report = {"config": asdict(cfg), "cases": [], "contexts": [],
              "packages": {p: version(p) for p in ("vllm", "torch", "transformers")},
              "note": "Initial-checkpoint component benchmark, not integrated training",
              "inference_dtype": "bfloat16", "success": False}
    started = time.perf_counter()
    llm = LLM(model=cfg.model, revision=cfg.revision, tokenizer_revision=cfg.revision,
              dtype="bfloat16", max_model_len=2048, max_num_seqs=128,
              gpu_memory_utilization=.75, seed=10001, generation_config="vllm",
              enable_prefix_caching=True, disable_log_stats=True)
    report["engine_start_seconds"] = time.perf_counter() - started
    tok = llm.get_tokenizer()
    prompts = Prompts(tok, cfg)

    def generate(label, ids, n=1, limit=1024, temperature=.7, forced=False):
        assert ids and all(0 < len(p) <= cfg.prompt_tokens for p in ids)
        params = SamplingParams(n=n, temperature=temperature, top_k=cfg.top_k,
                                top_p=cfg.top_p, max_tokens=limit,
                                min_tokens=limit if forced else 0,
                                ignore_eos=forced, repetition_penalty=1.,
                                presence_penalty=0., frequency_penalty=0.,
                                skip_special_tokens=True)
        start = time.perf_counter()
        results = llm.generate([{"prompt_token_ids": p} for p in ids], params,
                               use_tqdm=False)
        seconds = time.perf_counter() - start
        assert len(results) == len(ids)
        groups = []
        for prompt, result in zip(ids, results):
            assert list(result.prompt_token_ids) == prompt
            assert len(result.outputs) == n
            group = []
            for output in result.outputs:
                tokens = list(output.token_ids)
                assert 0 < len(tokens) <= limit
                assert output.finish_reason in ("stop", "length")
                assert tok.eos_token_id not in tokens[:-1]
                group.append({"prompt": prompt, "tokens": tokens, "text": output.text,
                              "finish_reason": output.finish_reason})
            groups.append(group)
        count = sum(len(s["tokens"]) for g in groups for s in g)
        case = {"label": label, "seconds": seconds, "sequences": len(ids)*n,
                "generated_tokens": count, "tokens_per_second": count/seconds,
                "mean_prompt_tokens": sum(map(len, ids))/len(ids),
                "capped": sum(s["finish_reason"] == "length" for g in groups for s in g),
                "n": n, "temperature": temperature, "limit": limit,
                "forced_length_diagnostic": forced}
        report["cases"].append(case)
        save("vllm", report)
        print(json.dumps(case), flush=True)
        return groups

    try:
        ids = [prompts.agent(r["question"]) for r in gsm[:4]]
        generate("warmup", ids[:1], limit=8)
        greedy = generate("greedy", ids, limit=64, temperature=0.)
        baseline = json.loads((OUT / "hf.json").read_text())
        assert baseline["prompt_ids"] == ids
        matches = []
        for expected, group in zip(baseline["greedy_tokens"], greedy):
            actual = group[0]["tokens"]
            prefix = 0
            for a, b in zip(expected, actual):
                if a != b:
                    break
                prefix += 1
            matches.append({"equal": actual == expected, "common_prefix": prefix,
                            "hf_length": len(expected), "vllm_length": len(actual)})
        report["greedy_comparison"] = matches
        generate("real_K4_same_four_questions", ids, n=4)
        for batch, length in ((16, 128), (64, 512), (128, 512)):
            generate(f"fixed_{batch}x{length}", [ids[0]] * batch,
                     limit=length, forced=True)

        filters = {}
        for label, rows in (("gsm_filter", gsm[:64]), ("math_filter", math[:64])):
            valid = []
            for row in rows:
                try:
                    valid.append((row, prompts.agent(row["question"])))
                except ValueError as exc:
                    report.setdefault("prompt_overflows", []).append({"id": row["id"], "error": str(exc)})
            groups = generate(label, [p for _, p in valid], n=4)
            start = time.perf_counter()
            selected = []
            for (row, _), group in zip(valid, groups):
                scores = [correct(s["text"], row["answer"], cfg.verifier) for s in group]
                if all(scores):
                    selected.append(row)
            report[label] = {"candidates": len(valid), "selected": len(selected),
                             "verification_seconds": time.perf_counter()-start}
            filters[label] = selected
            save("vllm", report)

        # Fresh evaluation traces, batched across questions; retries remain per
        # question and bounded. This is diagnostic output, not a final corpus.
        for label in ("gsm_filter", "math_filter"):
            selected = filters[label] if label == "gsm_filter" else filters[label][:4]
            pending = selected.copy()
            clean = {}
            for attempt in range(1, cfg.max_clean_attempts+1):
                if not pending:
                    break
                groups = generate(label+f"_fresh_{attempt}",
                                  [prompts.agent(r["question"]) for r in pending])
                failed = []
                for row, group in zip(pending, groups):
                    text = group[0]["text"]
                    if reasoning(text) and correct(text, row["answer"], cfg.verifier):
                        clean[row["id"]] = group[0]
                    else:
                        failed.append(row)
                pending = failed
            assert not pending, "Fresh-trace watchdog exhausted"
            if label == "gsm_filter":
                report["eval_fresh_questions"] = len(clean)
                continue
            assert len(clean) >= 2, "Insufficient retained MATH questions for component test"
            for i, row in enumerate(selected):
                # Alpha=0 is a declared benchmark stratum, chosen before looking
                # at generated pollutions. No corpus row is silently truncated.
                prefix, window = cut_window(reasoning(clean[row["id"]]["text"]), tok, 0., cfg.window_cap)
                report["contexts"].append({"row": row, "alpha": 0., "prefix": prefix, "clean": window})

        contexts = report["contexts"]
        pp = [prompts.polluter(c["row"]["question"], c["prefix"], c["clean"]) for c in contexts]
        corruptions = generate("agent_corruption", pp)
        deployments, repairs = [], []
        for c, p, group in zip(contexts, pp, corruptions):
            window, tagged = pollution(group[0]["text"], tok.decode(p, skip_special_tokens=False))
            c["corruption"] = group[0]
            c["polluted"] = window
            c["tagged"] = tagged
            deployments.append(prompts.agent(c["row"]["question"], c["prefix"], window))
            repairs.append(prompts.repair(c["row"]["question"], c["prefix"], c["clean"], window))
        agent_groups = generate("agent_G64", deployments, n=64)
        repair_groups = generate("agent_repair", repairs)
        polluter_groups = generate("polluter_G64", pp, n=64)
        score_prompts = []
        for c, prompt, group in zip(contexts, pp, polluter_groups):
            for sample in group:
                window, _ = pollution(sample["text"], tok.decode(prompt, skip_special_tokens=False))
                score_prompts.append(prompts.agent(c["row"]["question"], c["prefix"], window))
        scored = generate("polluter_scoring", score_prompts)
        for i, c in enumerate(contexts):
            c["agent_samples"] = agent_groups[i]
            c["agent_rewards"] = [correct(s["text"], c["row"]["answer"], cfg.verifier) for s in agent_groups[i]]
            c["repair_target"] = repair_target(repair_groups[i][0]["text"], tok,
                                               tok.decode(repairs[i], skip_special_tokens=False))
            c["polluter_samples"] = polluter_groups[i]
            c["polluter_rewards"] = [1-correct(g[0]["text"], c["row"]["answer"], cfg.verifier)
                                       for g in scored[i*64:(i+1)*64]]
        generate("greedy_evaluation_shape", deployments * 16, temperature=0.)
        report["success"] = True
    finally:
        save("vllm", report)


def backward(condition="split", output="backward"):
    cfg, _, _ = sources()
    cfg = replace(cfg, condition=condition)
    source = json.loads((OUT / "vllm.json").read_text())
    assert source["success"]
    model, tok = load_model(cfg)
    roles = Roles(model, condition)
    params = {id(p): p for role in ("agent", "polluter")
              for p in roles.active(role).parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(list(params.values()),
        lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay, foreach=False)
    probe = SimpleNamespace(cfg=cfg, roles=roles, optimizer=optimizer, role="agent",
                            loss_tokens={"agent": 0, "polluter": 0, "repair": 0},
                            forward_train_tokens=0)
    report = {"success": False, "condition": condition, "components": [],
              "contexts": len(source["contexts"]),
              "note": "Disposable component updates on saved initial-checkpoint rollouts; not an integrated self-play run"}
    try:
        for role in ("agent", "polluter"):
            probe.role = role
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            roles.refresh(role)
            torch.cuda.synchronize()
            snapshot_seconds = time.perf_counter()-started
            started = time.perf_counter()
            Trainer.place_optimizer_moments(probe)
            torch.cuda.synchronize()
            result = {"role": role, "transfer_seconds": time.perf_counter()-started,
                      "snapshot_seconds": snapshot_seconds,
                      "old_logps_seconds": 0., "backward_seconds": 0., "repair_seconds": 0.,
                      "sample_tokens": 0, "samples": 0}
            torch.cuda.reset_peak_memory_stats()
            for c in source["contexts"]:
                samples = [Sample(s["prompt"], s["tokens"], s["text"]) for s in c[role+"_samples"]]
                roles.active(role).eval()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad():
                    for sample in samples:
                        sample.old_logps = token_logps(roles.active(role), sample.prompt, sample.tokens, cfg).cpu()
                torch.cuda.synchronize()
                result["old_logps_seconds"] += time.perf_counter()-started
                started = time.perf_counter()
                Trainer.backward_group(probe, samples, c[role+"_rewards"])
                torch.cuda.synchronize()
                result["backward_seconds"] += time.perf_counter()-started
                result["samples"] += len(samples)
                result["sample_tokens"] += sum(len(s.tokens) for s in samples)
                if role == "agent" and c["repair_target"]:
                    started = time.perf_counter()
                    logps = token_logps(roles.agent, samples[0].prompt, c["repair_target"], cfg)
                    (-logps.sum()*cfg.guidance/cfg.accumulation).backward()
                    del logps
                    torch.cuda.synchronize()
                    result["repair_seconds"] += time.perf_counter()-started
            started = time.perf_counter()
            norm = torch.nn.utils.clip_grad_norm_(roles.active(role).parameters(), cfg.grad_clip)
            assert torch.isfinite(norm)
            optimizer.step()
            roles.updated(role)
            torch.cuda.synchronize()
            result["optimizer_seconds"] = time.perf_counter()-started
            result["grad_norm"] = float(norm)
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["components"].append(result)
            save(output, report)
            print(json.dumps(result), flush=True)
        report["success"] = True
    finally:
        save(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("hf", "inference", "backward"))
    parser.add_argument("--condition", choices=("unified", "split"), default="split")
    parser.add_argument("--output", default="backward")
    args = parser.parse_args()
    if args.phase == "backward":
        backward(args.condition, args.output)
    else:
        {"hf": hf, "inference": inference}[args.phase]()
