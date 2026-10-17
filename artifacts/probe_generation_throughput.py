"""Diagnostic only: fixed-length generation benchmark, no corpus/training writes."""
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch

sys.path.insert(0, "/content/gasp/src")
from gasp.config import Config
from gasp.data import read_cache
from gasp.models import autocast, load_model, seed_all
from gasp.prompts import Prompts
from gasp.verification import correct

pid = int(sys.argv[1])
report = {"torch_threads": torch.get_num_threads(), "cases": [], "success": False}
output = Path("/content/generation-throughput.json")
os.kill(pid, signal.SIGSTOP)
try:
    cfg = Config.load("/content/gasp/data/full-v3/resolved_config.json")
    model, tok = load_model(cfg)
    model.eval()
    rows, _ = read_cache("/content/gasp/data/raw/gsm_raw.jsonl", "raw_eval")
    prompt = Prompts(tok, cfg).agent(rows[1]["question"])
    ids = torch.tensor([prompt], device="cuda")
    common = {"attention_mask": torch.ones_like(ids), "do_sample": True,
              "temperature": .7, "top_k": cfg.top_k, "top_p": cfg.top_p,
              "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
              "use_cache": True}
    with torch.no_grad(), autocast(cfg):
        model.generate(input_ids=ids, max_new_tokens=16, **common)
    # Equal work: 16 sequences x 128 new tokens. Forced length is diagnostic only.
    for batch in (1, 4, 16):
        seed_all(10001)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            for _ in range(16 // batch):
                with autocast(cfg):
                    model.generate(input_ids=ids, num_return_sequences=batch,
                        min_new_tokens=128, max_new_tokens=128, **common)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        result = {"batch": batch, "seconds": elapsed, "generated_tokens": 2048,
                  "tokens_per_second": 2048/elapsed,
                  "total_allocated_bytes_including_paused_job": torch.cuda.memory_allocated()}
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
    # Real configuration: four solutions, EOS enabled, unchanged 1024-token cap.
    seed_all(10001)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), autocast(cfg):
        generated = model.generate(input_ids=ids, num_return_sequences=4,
                                   max_new_tokens=cfg.completion_tokens, **common)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter()-started
    texts = tok.batch_decode(generated[:, len(prompt):], skip_special_tokens=True)
    started = time.perf_counter()
    scores = [correct(text, rows[1]["answer"], cfg.verifier) for text in texts]
    report["real_k4"] = {"generation_seconds": generation_seconds,
                         "verification_seconds": time.perf_counter()-started,
                         "scores": scores, "outputs": texts}
    print(json.dumps(report["real_k4"]), flush=True)
    report["success"] = True
finally:
    output.write_text(json.dumps(report, indent=2)+"\n")
    os.kill(pid, signal.SIGCONT)
