"""One real corpus request from a committed prefix, reused by final generation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gasp.data import sha256, write_json
from gasp.eval_corpus import request_corruption, response_pollution
from gasp.prompts import POLLUTER

checkpoint = ROOT / "artifacts/gasp-traces-20261017-034910/backup-00038/eval-contexts.jsonl.progress.pt"
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert state["cursor"] == 20
assert state["identity"]["source_sha256"] == sha256(ROOT / "data/raw/gsm_raw.jsonl")
for name, digest in state["identity"]["code"].items():
    assert sha256(ROOT / "src/gasp" / name) == digest, name

index = 2
row = state["retained"][index]
assert row["id"] == "gsm8k-test-1-a0.5"
assert row["split"] == "test" and row["filter_scores"] == [1, 1, 1, 1]
settings = {"model": "gpt-6.1-sol", "reasoning": {"effort": "low"},
            "max_output_tokens": 4096, "store": False}
payload = settings | {"input": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": POLLUTER.format(q=row["question"],
        prefix=row["prefix"], continuation=row["clean"])}]}
path = ROOT / f"data/full-v3/held-out-corruptions.jsonl.responses/{index:06d}.json"
if path.exists():
    saved = json.loads(path.read_text())
    assert saved["request"] == payload and saved["context_id"] == row["id"]
    response = saved["response"]
else:
    if not os.environ.get("OPENAI_API_KEY"):
        for line in (ROOT / ".env").read_text().splitlines():
            name, separator, value = line.strip().removeprefix("export ").lstrip().partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                values = shlex.split(value, comments=True)
                if len(values) == 1:
                    os.environ["OPENAI_API_KEY"] = values[0]
    response = request_corruption(payload)
    write_json(path, {"context_id": row["id"], "request": payload, "response": response,
                      "received_at": datetime.now(timezone.utc).isoformat()})
window, raw = response_pollution(response, settings["model"])
assert window.strip() != row["clean"].strip()
report = {"context_id": row["id"], "checkpoint_sha256": sha256(checkpoint),
          "polluter_prompt_sha256": hashlib.sha256(POLLUTER.encode()).hexdigest(),
          "saved_response": str(path.relative_to(ROOT)), "response_sha256": sha256(path),
          "model": response["model"], "status": response["status"],
          "usage": response.get("usage"), "clean": row["clean"], "polluted": window,
          "note": "Preflight from committed context index 2; final generator revalidates exact request and context ID before reuse."}
write_json(ROOT / "artifacts/gpt-preflight-20261017.json", report)
print(json.dumps(report, indent=2))
