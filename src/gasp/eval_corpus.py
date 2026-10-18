"""Build a fixed recoverability corpus; see docs/CORPUS.md for paper vs choices."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from .data import read_cache, write_cache, write_json, sha256, cut_window
from .models import load_model, Rollouts, model_digest, rng_state, restore_rng
from .prompts import Prompts, POLLUTER
from .verification import correct, reasoning, pollution


def prepare_eval_contexts(config, source, output, questions=None, resume=False, max_seconds=None,
                          *, retained_target=None, max_candidates=None):
    """K=4 selection, a fresh correct trace, and all four cuts per kept question.

    A retained target or candidate cap opts into recorded seeded shuffling and
    batched fresh traces. A short target returns an incomplete status, never a
    final cache. max_seconds is a per-invocation soft limit checked after whole
    batches; bounded-mode timing includes model loading. Resume preserves the
    candidate cap/target/config but may change the time allowance.
    """
    invocation_started = time.perf_counter()
    bounded = retained_target is not None or max_candidates is not None
    if not config.tiny and config.revision == "main":
        raise ValueError("Use resolved_config.json so both conditions share the initial checkpoint")
    if questions is not None and questions < 1:
        raise ValueError("questions must be positive")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if retained_target is not None and retained_target < 1:
        raise ValueError("retained_target must be positive")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    output = Path(output)
    audit_path = Path(str(output) + ".audit.jsonl")
    progress_path = Path(str(output) + ".progress.pt")
    if any(p.exists() for p in (output, Path(str(output)+".manifest.json"))):
        raise FileExistsError("Evaluation context preparation exists; use a fresh output path")
    if resume and not (progress_path.exists() and audit_path.exists()):
        raise FileNotFoundError("Resume requires both the progress checkpoint and audit")
    if not resume and (audit_path.exists() or progress_path.exists()):
        raise FileExistsError("Preparation exists; inspect it and explicitly resume")
    rows, source_manifest = read_cache(source, "raw_eval")
    if any(r.get("split") != "test" for r in rows):
        raise ValueError("Evaluation contexts require test-split questions")
    config = config.with_changes(seed=config.eval_seed)
    if bounded:
        random.Random(config.eval_seed).shuffle(rows)
    candidate_limit = len(rows) if questions is None else questions
    if max_candidates is not None:
        candidate_limit = min(candidate_limit, max_candidates)
    candidates = rows[:candidate_limit]
    identity = {"config": asdict(config), "source_sha256": source_manifest["sha256"],
                "questions": questions,
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "packages": {name: version(name) for name in ("transformers", "numpy", "math-verify")}},
                "code": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
    if bounded:
        identity.update(retained_target=retained_target, max_candidates=max_candidates,
                        candidate_ids=[row["id"] for row in candidates])
    saved = torch.load(progress_path, map_location="cpu", weights_only=False) if resume else None
    if saved is not None and saved["identity"] != identity:
        raise ValueError("Resume config/source/code differs from the saved preparation")
    order_path = Path(str(output)+".candidate_order.json")
    if bounded:
        order = {"source_sha256": source_manifest["sha256"], "seed": config.eval_seed,
                 "shuffle": True, "candidate_ids": [row["id"] for row in candidates]}
        if resume and (not order_path.exists() or json.loads(order_path.read_text()) != order):
            raise ValueError("Resume candidate order differs")
        if not resume:
            write_json(order_path, order)
    model, tok = load_model(config)
    initial_digest = model_digest(model)
    prompts, rollout = Prompts(tok, config), Rollouts(tok, config)
    retained, cursor, prior_seconds = [], 0, 0.
    output.parent.mkdir(parents=True, exist_ok=True)
    if saved is not None:
        if saved["initial_model_sha256"] != initial_digest:
            raise ValueError("Resume initial model differs")
        if audit_path.stat().st_size < saved["audit_size"]:
            raise ValueError("Resume audit is shorter than the committed checkpoint")
        with audit_path.open("r+b") as audit:
            audit.seek(saved["audit_size"])
            tail = audit.read()
            if tail:
                Path(str(audit_path)+f".interrupted-{time.time_ns()}").write_bytes(tail)
            audit.truncate(saved["audit_size"])
        retained, cursor = saved["retained"], saved["cursor"]
        prior_seconds = saved.get("wall_seconds", 0.)
        rollout.counts = saved["counts"]
        restore_rng(saved["rng"])

    started = invocation_started if bounded else time.perf_counter()

    def status(complete=False, reason="running"):
        elapsed = time.perf_counter()-started
        return {"complete": complete, "reason": reason, "processed": cursor,
                "total": len(candidates), "selected_questions": len(retained)//4,
                "retained_target": retained_target, "counts": rollout.counts,
                "wall_seconds": prior_seconds+elapsed, "invocation_wall_seconds": elapsed}

    def checkpoint(audit):
        audit.flush()
        state = {"identity": identity, "retained": retained, "cursor": cursor,
                 "counts": rollout.counts, "rng": rng_state(),
                 "initial_model_sha256": initial_digest, "audit_size": audit_path.stat().st_size,
                 "wall_seconds": prior_seconds+time.perf_counter()-started}
        temporary = Path(str(progress_path)+".tmp")
        torch.save(state, temporary)
        os.replace(temporary, progress_path)
        if bounded:
            write_json(str(output)+".status.json", status())

    with audit_path.open("a" if resume else "x") as audit:
        checkpoint(audit)
        batch_size = config.preparation_batch_questions
        for start in range(cursor, len(candidates), batch_size):
            if retained_target is not None and len(retained)//4 >= retained_target:
                break
            batch_started = time.perf_counter()
            batch = candidates[start:start+batch_size]
            entry = None
            try:
                batch_prompts = [prompts.agent(row["question"]) for row in batch]
                groups = rollout.generate_many(model, batch_prompts, 4,
                    config.completion_tokens, "eval_filter", temperature=.7)
                if len(groups) != len(batch) or any(len(group) != 4 for group in groups):
                    raise RuntimeError("Preparation requires exactly four samples per question")
                if bounded:
                    entries, pending = [], []
                    remaining = retained_target-len(retained)//4 if retained_target is not None else len(batch)
                    for index, (row, prompt, samples) in enumerate(zip(batch, batch_prompts, groups), start):
                        scores = [correct(s.text, row["answer"], config.verifier) for s in samples]
                        entry = {"question_id": row["id"], "candidate_index": index, "kept": False,
                                 "filter_scores": scores, "filter_outputs": [s.text for s in samples]}
                        entries.append(entry)
                        if all(scores):
                            if remaining > 0:
                                entry["clean_outputs"] = []
                                pending.append((row, prompt, entry))
                                remaining -= 1
                            else:
                                entry["reason"] = "target_already_reached_in_batch"
                    # These are new on-policy samples, distinct from all K=4 filter
                    # outputs. Retry only unfinished questions, retaining their order.
                    completed = {}
                    for attempt in range(1, config.max_clean_attempts + 1):
                        if not pending:
                            break
                        fresh_groups = rollout.generate_many(model, [p for _, p, _ in pending], 1,
                            config.completion_tokens, "eval_clean_trace", temperature=.7)
                        if len(fresh_groups) != len(pending) or any(len(g) != 1 for g in fresh_groups):
                            raise RuntimeError("Preparation requires one fresh trace per pending question")
                        retry = []
                        for (row, prompt, entry), group in zip(pending, fresh_groups):
                            sample = group[0]
                            entry["clean_outputs"].append(sample.text)
                            trace = reasoning(sample.text)
                            if trace and correct(sample.text, row["answer"], config.verifier):
                                completed[row["id"]] = (sample, trace, attempt)
                            else:
                                retry.append((row, prompt, entry))
                        pending = retry
                    if pending:
                        entry = pending[0][2]
                        raise RuntimeError(f"No correct nonempty evaluation trace for {pending[0][0]['id']}")
                    # Keep corpus order tied to shuffled candidates, not to retry success order.
                    for row, entry in zip(batch, entries):
                        if row["id"] in completed:
                            sample, trace, attempt = completed[row["id"]]
                            for alpha in (0., .25, .5, .75):
                                prefix, clean = cut_window(trace, tok, alpha, config.window_cap)
                                retained.append({"id": f"{row['id']}-a{alpha:g}",
                                    "question_id": row["id"], "question": row["question"],
                                    "answer": row["answer"], "split": "test", "alpha": alpha,
                                    "prefix": prefix, "clean": clean, "clean_trace": sample.text,
                                    "filter_scores": entry["filter_scores"], "clean_attempts": attempt,
                                    "initial_clean_solved": True})
                            entry["kept"] = True
                        audit.write(json.dumps(entry)+"\n")
                for row, prompt, samples in (() if bounded else zip(batch, batch_prompts, groups)):
                    entry = {"question_id": row["id"], "kept": False}
                    scores = [correct(s.text, row["answer"], config.verifier) for s in samples]
                    entry.update(filter_scores=scores, filter_outputs=[s.text for s in samples])
                    if all(scores):
                        entry["clean_outputs"] = []
                        for attempt in range(1, config.max_clean_attempts + 1):
                            sample = rollout.generate(model, prompt, 1, config.completion_tokens,
                                                      "eval_clean_trace", temperature=.7)[0]
                            entry["clean_outputs"].append(sample.text)
                            trace = reasoning(sample.text)
                            if trace and correct(sample.text, row["answer"], config.verifier):
                                break
                        else:
                            raise RuntimeError(f"No correct nonempty evaluation trace for {row['id']}")
                        for alpha in (0., .25, .5, .75):
                            prefix, clean = cut_window(trace, tok, alpha, config.window_cap)
                            retained.append({"id": f"{row['id']}-a{alpha:g}",
                                "question_id": row["id"], "question": row["question"],
                                "answer": row["answer"], "split": "test", "alpha": alpha,
                                "prefix": prefix, "clean": clean, "clean_trace": sample.text,
                                "filter_scores": scores, "clean_attempts": attempt,
                                "initial_clean_solved": True})
                        entry["kept"] = True
                    audit.write(json.dumps(entry) + "\n")
            except BaseException as exc:
                audit.write(json.dumps((entry or {}) | {"batch_question_ids": [r["id"] for r in batch],
                                       "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                audit.flush()
                if bounded:
                    write_json(str(output)+".status.json", status(reason="error") |
                               {"error": f"{type(exc).__name__}: {exc}", "uncommitted_batch": True})
                raise
            # Commit the whole batch: RNG has already advanced for every filter
            # sample. An interrupted batch is replayed from the previous state.
            cursor = start + len(batch)
            checkpoint(audit)
            if bounded:
                print(json.dumps({"stage": "prepare_eval_contexts", **status(),
                                  "batch_seconds": time.perf_counter()-batch_started}), flush=True)
            else:
                print(f"contexts: processed={cursor}/{len(candidates)} "
                      f"selected={len(retained)//4} "
                      f"batch_seconds={time.perf_counter()-batch_started:.2f} "
                      f"elapsed_seconds={time.perf_counter()-started:.2f}", flush=True)
            if max_seconds is not None and time.perf_counter()-started >= max_seconds:
                break
    if bounded:
        complete = len(retained)//4 >= retained_target if retained_target is not None else cursor == len(candidates)
        reason = "target_reached" if complete and retained_target is not None else (
            "candidates_complete" if complete else "time_limit" if cursor < len(candidates) else
            "candidate_limit" if candidate_limit < len(rows) else "source_exhausted")
        result = status(complete, reason)
        if not complete:
            write_json(str(output)+".status.json", result)
            return result
    elif cursor < len(candidates):
        return {"complete": False, "processed": cursor, "total": len(candidates)}
    provenance = {"kind": "eval_contexts",
        "protocol": "initial-K4-fresh-trace-four-alphas-v1",
        "config": asdict(config), "source_sha256": source_manifest["sha256"],
        "source_provenance": source_manifest["provenance"],
        "candidate_count": len(rows[:questions]), "selected_questions": len(retained)//4,
        "initial_model_sha256": initial_digest, "counts": rollout.counts,
        "prompt_sha256": sha256(Path(__file__).with_name("prompts.py")),
        "subset_selection": f"Initial {config.model}@{config.revision}; K=4, T=0.7, "
                            f"all correct; seed={config.seed}; one fresh correct trace; all four alphas"}
    if bounded:
        provenance.update(candidate_count=cursor, candidate_limit=len(candidates),
            retained_target=retained_target, candidate_order_sha256=sha256(order_path),
            candidate_shuffle_seed=config.eval_seed, wall_seconds=result["wall_seconds"])
    manifest = write_cache(output, retained, provenance)
    if bounded:
        write_json(str(output)+".status.json", result)
    return manifest | result if bounded else manifest


def request_corruption(payload):
    """One Responses API request. No automatic paid retries or model substitution."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY in the execution environment")
    request = Request("https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=180) as response:
            return json.load(response)
    except HTTPError as exc:
        # Never print the credential-bearing request headers.
        raise RuntimeError(f"Corruption API HTTP {exc.code}; no automatic retry. "
                           "Check account/model access or rate limits and rerun.") from None


def response_pollution(response, expected_model):
    if response.get("model") != expected_model:
        raise ValueError("Response model differs from the requested model")
    if response.get("status") != "completed":
        raise ValueError("Incomplete/failed response; inspect the saved response")
    raw = "".join(c["text"] for item in response.get("output", [])
                  if item.get("type") == "message" and item.get("role") == "assistant"
                  for c in item.get("content", []) if c.get("type") == "output_text")
    window, tagged = pollution(raw)
    if not tagged or not window.strip():
        raise ValueError("Generator must return a nonempty <polluted> block; inspect the saved response")
    return window, raw


def generate_eval_corpus(source, output, model="gpt-6.1-sol", effort="low",
                         max_output_tokens=4096, max_requests=None):
    """Generate resumable external JSONL plus provenance for the existing importer.

    Completed responses are retained before parsing. Invalid responses stop the
    run and are not silently replaced, skipped or billed again on resumption.
    """
    if model != "gpt-6.1-sol":
        raise ValueError("Corpus generation uses the requested gpt-6.1-sol model")
    if effort not in ("low", "medium", "high", "xhigh", "max") or max_output_tokens < 1:
        raise ValueError("Invalid GPT-6.1 Sol reasoning effort or output budget")
    if max_requests is not None and max_requests < 1:
        raise ValueError("max_requests must be positive")
    rows, manifest = read_cache(source, "eval_contexts")
    if manifest["provenance"]["config"]["tiny"]:
        raise ValueError("Synthetic smoke contexts cannot become an evaluation corpus")
    if manifest["provenance"]["prompt_sha256"] != sha256(Path(__file__).with_name("prompts.py")):
        raise ValueError("Context prompt source differs; prepare a new cache")
    if any(r.get("split") != "test" or not r.get("clean", "").strip() for r in rows):
        raise ValueError("Every context needs a test-split question and nonempty clean window")
    output = Path(output)
    provenance_path = Path(str(output) + ".provenance.json")
    if output.exists() or provenance_path.exists():
        raise FileExistsError("Final corpus exists; use a new output path")
    settings = {"model": model, "reasoning": {"effort": effort},
                "max_output_tokens": max_output_tokens, "store": False}
    job = {"contexts_sha256": manifest["sha256"], "settings": settings,
           "polluter_prompt_sha256": hashlib.sha256(POLLUTER.encode()).hexdigest()}
    response_dir = Path(str(output) + ".responses")
    response_dir.mkdir(parents=True, exist_ok=True)
    job_path = response_dir/"job.json"
    if job_path.exists():
        if json.loads(job_path.read_text()) != job:
            raise ValueError("Resume source/prompt/settings differ; use a new output path")
    else:
        write_json(job_path, job)
    generated, sent = [], 0
    for index, row in enumerate(rows):
        payload = settings | {"input": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": POLLUTER.format(q=row["question"],
                prefix=row["prefix"], continuation=row["clean"])}]}
        response_path = response_dir/f"{index:06d}.json"
        if response_path.exists():
            saved = json.loads(response_path.read_text())
            if saved["request"] != payload or saved["context_id"] != row["id"]:
                raise ValueError("Saved response does not match this context/request")
            response = saved["response"]
        else:
            if max_requests is not None and sent >= max_requests:
                return {"complete": False, "saved_contexts": index, "total_contexts": len(rows)}
            response = request_corruption(payload)
            sent += 1
            write_json(response_path, {"context_id": row["id"], "request": payload,
                                       "response": response,
                                       "received_at": datetime.now(timezone.utc).isoformat()})
        window, raw = response_pollution(response, model)
        if window.strip() == row["clean"].strip():
            raise ValueError("Unchanged corruption window; inspect the saved response")
        generated.append(row | {"polluted": window, "generator": model,
            "generator_response_id": response.get("id"), "raw_corruption": raw,
            "response_sha256": sha256(response_path)})
    output.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in generated))
    write_json(provenance_path, {"protocol": "reconstructed-sol-fixed-initial-subset-v1",
        "generator_model": model, "generator_revision": model, "held_out": True,
        "revision_note": "API model identifier, not an immutable dated snapshot; raw responses retained",
        "paper_deviation": "User-selected GPT-6.1 Sol replaces the paper's GPT-5 corruption generator",
        "source_url": Path(source).resolve().as_uri(), "contexts_sha256": manifest["sha256"],
        "context_preparation": manifest["provenance"], "generation_settings": settings,
        "subset_selection": manifest["provenance"]["subset_selection"],
        "polluter_prompt_sha256": job["polluter_prompt_sha256"],
        "corpus_sha256": sha256(output), "semantic_review": "Not performed by the generator",
        "response_directory": str(response_dir.resolve()),
        "scope": "Initial clean-solved questions only; not full GSM8K coverage"})
    return {"complete": True, "contexts": len(generated), "corpus": str(output),
            "provenance": str(provenance_path)}
