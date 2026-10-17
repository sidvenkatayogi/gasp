"""Immutable, provenance-bearing data caches. Training never receives eval rows."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re
import time

from .models import load_model, Rollouts
from .prompts import Prompts
from .verification import correct, extract_answer, reasoning


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def question_hash(q):
    return hashlib.sha256(" ".join(q.lower().split()).encode()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_cache(path, rows, provenance):
    path = Path(path)
    if path.exists() or path.with_suffix(path.suffix + ".manifest.json").exists():
        raise FileExistsError(f"Immutable cache already exists: {path}; use a new version directory")
    if not rows:
        raise ValueError("Refusing an empty data cache")
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row IDs")
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    manifest = {"schema": 1, "sha256": sha256(path), "rows": len(rows),
                "provenance": provenance}
    write_json(str(path) + ".manifest.json", manifest)
    return manifest


def read_cache(path, kind=None):
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if sha256(path) != manifest["sha256"]:
        raise ValueError("Cache hash mismatch: " + str(path))
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != manifest["rows"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Cache count/ID mismatch")
    if kind and manifest["provenance"]["kind"] != kind:
        raise ValueError(f"Expected {kind} cache")
    return rows, manifest


def assert_disjoint(train, evaluation):
    train_questions = {question_hash(r["question"]) for r in train}
    if train_questions & {question_hash(r["question"]) for r in evaluation}:
        raise ValueError("Training/evaluation question overlap")
    if any(r.get("split") != "train" for r in train):
        raise ValueError("Every training row must be marked train")
    if any(r.get("split") != "test" for r in evaluation):
        raise ValueError("Every evaluation row must be marked test")


def cut_window(trace, tokenizer, alpha, cap):
    ids = tokenizer.encode(trace, add_special_tokens=False)
    if not ids:
        raise ValueError("empty clean reasoning")
    start = int(len(ids) * alpha)
    width = min(cap, max(1, int(len(ids) * .1)))
    return (tokenizer.decode(ids[:start], skip_special_tokens=True),
            tokenizer.decode(ids[start:start+width], skip_special_tokens=True))


def resolve_config(config):
    if config.tiny:
        return config
    from huggingface_hub import HfApi
    revision = HfApi().model_info(config.model, revision=config.revision).sha
    return config.with_changes(revision=revision)


def download_sources(directory, math_repo="DigitalLearningGmbH/MATH-lighteval", math_revision="main",
                     gsm_revision="main"):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    mr = api.dataset_info(math_repo, revision=math_revision).sha
    gr = api.dataset_info("openai/gsm8k", revision=gsm_revision).sha
    train = load_dataset(math_repo, revision=mr, split="train")
    test = load_dataset("openai/gsm8k", "main", revision=gr, split="test")
    train_rows = [{"id": f"math-train-{i}", "question": r["problem"],
        "answer": extract_answer(r["solution"]), "reference_trace": reasoning(r["solution"]),
        "split": "train"} for i, r in enumerate(train)]
    if any(r["answer"] is None for r in train_rows):
        raise ValueError("MATH answers must be boxed; dataset schema or extraction needs review")
    test_rows = [{"id": f"gsm8k-test-{i}", "question": r["question"],
        "answer": extract_answer(r["answer"]),
        "reference_trace": re.sub(r"<<.*?>>", "", reasoning(r["answer"])),
        "split": "test"} for i, r in enumerate(test)]
    assert_disjoint(train_rows, test_rows)
    write_cache(Path(directory)/"math_raw.jsonl", train_rows,
        {"kind": "raw_train", "dataset": math_repo, "revision": mr, "split": "train"})
    write_cache(Path(directory)/"gsm_raw.jsonl", test_rows,
        {"kind": "raw_eval", "dataset": "openai/gsm8k", "revision": gr, "split": "test"})


def prepare_train(config, source, output, candidates=7500):
    start = time.perf_counter()
    config = resolve_config(config).with_changes(seed=42)
    rows, manifest = read_cache(source, "raw_train")
    rng = random.Random(42)
    rng.shuffle(rows)
    model, tok = load_model(config)
    prompts, rollout = Prompts(tok, config), Rollouts(tok, config)
    retained, audit = [], []
    for row in rows[:candidates]:
        try:
            p = prompts.agent(row["question"])
            samples = rollout.generate(model, p, 4, config.completion_tokens, "train_filter", temperature=.7)
            scores = [correct(s.text, row["answer"], config.verifier) for s in samples]
            keep = all(scores)
            if keep:
                retained.append({k:v for k,v in row.items() if k != "reference_trace"} |
                                {"filter_scores": scores})
            audit.append({"id": row["id"], "scores": scores, "kept": keep,
                          "outputs": [s.text for s in samples]})
        except ValueError as exc:
            # Shared preparation-time prompt filtering applies once to both conditions.
            audit.append({"id": row["id"], "kept": False, "error": str(exc)})
    write_json(str(output)+".filter_audit.json", audit)
    write_cache(output, retained, {"kind": "train", "protocol": "initial-K4-all-correct",
        "source_sha256": manifest["sha256"], "config": asdict(config),
        "candidate_count": min(len(rows), candidates), "counts": rollout.counts,
        "wall_seconds": time.perf_counter()-start,
        "prompt_sha256": sha256(Path(__file__).with_name("prompts.py"))})
    config.save(Path(output).parent/"resolved_config.json")


def validate_eval_provenance(provenance):
    """Accept the requested Sol generator or paper-style GPT-5 corpus imports."""
    for key in ("protocol", "generator_model", "generator_revision", "source_url",
                "generation_settings", "subset_selection"):
        if not provenance.get(key):
            raise ValueError(f"Missing evaluation provenance: {key}")
    if not re.fullmatch(r"(?:gpt-5(?:-\d{4}-\d{2}-\d{2})?|gpt-6\.1-sol)",
                        provenance["generator_model"], re.I):
        raise ValueError("Recoverability evaluation requires held-out GPT-5 or GPT-6.1 Sol corruptions")
    if provenance.get("held_out") is not True:
        raise ValueError("Evaluation provenance must declare held_out=true")


def import_eval(config, source, output, provenance_path):
    """Import author or other external fixed contexts; require accountable provenance."""
    provenance = json.loads(Path(provenance_path).read_text())
    validate_eval_provenance(provenance)
    if provenance.get("corpus_sha256") and provenance["corpus_sha256"] != sha256(source):
        raise ValueError("Corpus differs from its generation provenance")
    if not config.tiny and config.revision == "main":
        raise ValueError("Use resolved_config.json from training-data preparation")
    from .models import TinyTokenizer
    if config.tiny:
        tok = TinyTokenizer()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    p = Prompts(tok, config)
    rows = [json.loads(s) for s in Path(source).read_text().splitlines()]
    for r in rows:
        for key in ("id", "question_id", "question", "answer", "prefix", "polluted", "initial_clean_solved"):
            if key not in r:
                raise ValueError(f"Missing {key}")
        if r.get("split", "test") != "test":
            raise ValueError("Evaluation rows must belong to the test split")
        if not isinstance(r["polluted"], str) or not r["polluted"].strip():
            raise ValueError("Missing polluted window; no synthetic fallback is available")
        if type(r["initial_clean_solved"]) is not bool:
            raise ValueError("initial_clean_solved must be a boolean subset flag")
        if r.get("generator", provenance["generator_model"]) != provenance["generator_model"]:
            raise ValueError("Row generator differs from held-out corpus provenance")
        r["split"] = "test"
        r["generator"] = provenance["generator_model"]
        r["clean_prompt_ids"] = p.agent(r["question"])
        r["corrupt_prompt_ids"] = p.agent(r["question"], r["prefix"], r["polluted"])
    write_cache(output, rows, provenance | {"kind": "eval", "config": asdict(config),
        "input_sha256": sha256(source), "prompt_sha256": sha256(Path(__file__).with_name("prompts.py"))})


def make_smoke_data(directory, config):
    from .models import TinyTokenizer
    directory = Path(directory)
    tok = TinyTokenizer()
    p = Prompts(tok, config)
    train = [{"id": f"train-{i}", "question": f"train {letter}", "answer": str(i%2),
              "filter_scores": [1]*4, "split": "train"}
             for i, letter in enumerate("abcd")]
    evaluation = [{"id": f"eval-{i}", "question_id": f"eval-{i}", "question": f"test {letter}",
        "answer": str(i%2), "prefix": "", "polluted": str(1-i%2), "split": "test",
        "initial_clean_solved": i < 2, "generator": "synthetic-binary-smoke-fixture",
        "clean_prompt_ids": p.agent(f"test {letter}"),
        "corrupt_prompt_ids": p.agent(f"test {letter}", "", str(1-i%2))}
        for i, letter in enumerate("efgh")]
    write_cache(directory/"train.jsonl", train,
                {"kind": "train", "protocol": "synthetic-smoke-NOT-K4-filtered", "config": asdict(config)})
    write_cache(directory/"eval.jsonl", evaluation,
                {"kind": "eval", "protocol": "synthetic-smoke-NOT-GSM8K", "config": asdict(config)})
