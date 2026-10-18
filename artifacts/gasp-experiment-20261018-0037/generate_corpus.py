"""Local CPU/API stage. Credentials stay local; completed responses are reused."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--max-requests", type=int)
parser.add_argument("--contexts", type=Path, default=HERE / "results/prepare/eval-contexts.jsonl")
parser.add_argument("--output-directory", type=Path, default=HERE / "corpus")
args = parser.parse_args()
with zipfile.ZipFile(HERE / "source.zip") as frozen:
    for name in frozen.namelist():
        if name.startswith("src/gasp/") and name.endswith(".py"):
            if frozen.read(name) != (ROOT / name).read_bytes():
                raise ValueError("Local API source differs from the frozen experiment: " + name)

if not os.environ.get("OPENAI_API_KEY"):
    for line in (ROOT / ".env").read_text().splitlines():
        name, separator, value = line.strip().removeprefix("export ").lstrip().partition("=")
        if separator and name.strip() == "OPENAI_API_KEY":
            values = shlex.split(value, comments=True)
            if len(values) == 1:
                os.environ["OPENAI_API_KEY"] = values[0]
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("No local OpenAI credential is configured")

from gasp import eval_corpus
original_request = eval_corpus.request_corruption
sent = 0

def observed_request(payload):
    global sent
    started = time.monotonic()
    result = original_request(payload)
    sent += 1
    print(json.dumps({"new_response": sent, "seconds": time.monotonic() - started,
                      "model": result.get("model"), "status": result.get("status"),
                      "usage": result.get("usage")}), flush=True)
    return result

eval_corpus.request_corruption = observed_request
directory = args.output_directory
directory.mkdir(exist_ok=True)
result = eval_corpus.generate_eval_corpus(
    args.contexts,
    directory / "held-out-corruptions.jsonl", max_requests=args.max_requests)
(directory / "generation-status.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
