"""Generate portable Jupyter/Colab notebooks with no build-time dependencies."""
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(text).strip()+"\n"}


def code(text,tags=()):
    return {"cell_type":"code","metadata":{"tags":list(tags)},"source":textwrap.dedent(text).strip()+"\n",
            "execution_count":None,"outputs":[]}


BOOTSTRAP=code('''
from pathlib import Path
import os, sys, json, subprocess, zipfile

candidates = [Path(os.environ.get("GASP_ROOT", ".")), Path.cwd(), Path.cwd().parent, Path("/content/gasp")]
ROOT = next((p.resolve() for p in candidates if (p / "src/gasp").is_dir()), None)
if ROOT is None:
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError("Unzip dist/gasp-colab.zip and open this notebook inside the project.")
    print("Select the delivered gasp-colab.zip archive.")
    uploaded = files.upload()
    archives = [p for p in uploaded if p.endswith(".zip")]
    if len(archives) != 1:
        raise ValueError("Select exactly one project archive")
    with zipfile.ZipFile(archives[0]) as z:
        for name in z.namelist():
            if not name.startswith("gasp/") or ".." in Path(name).parts:
                raise ValueError("Unexpected archive member")
        z.extractall("/content")
    ROOT = Path("/content/gasp")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

def command(*args):
    result = subprocess.run([sys.executable, *map(str,args)], cwd=ROOT,
                            env={**os.environ,"PYTHONPATH":str(ROOT/"src")},
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    result.check_returncode()

print("Project:", ROOT)
''')

SETUP=code('''
# Run once per new Colab runtime. Restart the runtime afterward, then rerun bootstrap.
INSTALL_DEPENDENCIES = True
if INSTALL_DEPENDENCIES:
    assert (3,11) <= sys.version_info[:2] < (3,14), "Use Python 3.11–3.13"
    # These optional image/audio packages can conflict with a pinned text-only torch.
    command("-m", "pip", "uninstall", "-y", "torchvision", "torchaudio")
    command("-m", "pip", "install", "-r", "requirements-colab.txt")
    command("-m", "pip", "install", "--no-deps", ".")
    print("Restart runtime if packages were already imported. No training has started.")
''',tags=["setup","requires-network"])

DRIVE=code('''
# Optional persistence. Keep downloaded HF weights in Colab scratch; persist data/runs.
USE_DRIVE = False
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = Path("/content/drive/MyDrive/gasp-experiment")
else:
    WORK = ROOT
WORK.mkdir(parents=True, exist_ok=True)
print("Data and outputs:", WORK)
''',tags=["optional-colab"])


def save(name,cells):
    for i,c in enumerate(cells):c["id"]=f"cell-{i:03d}"
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.12"},"colab":{"name":name,"provenance":[]}},
        "nbformat":4,"nbformat_minor":5}
    (ROOT/"notebooks"/name).write_text(json.dumps(notebook,indent=2)+"\n")


save("00_plan_and_smoke.ipynb",[
    md('''# GASP parameter-sharing experiment: design and local validation
    Start here. This notebook runs a **synthetic smoke test**, not a paper reproduction.
    It uses a real, randomly initialized 11,424-parameter Qwen2 causal transformer and
    real sampling, GRPO, repair NLL, backward passes and AdamW updates. Its binary
    vocabulary makes the complete pipeline affordable on CPU. No weights download
    or paid resource is required after dependencies are installed.

    Open via Colab **File → Upload notebook**, then provide `dist/gasp-colab.zip`
    to the bootstrap cell. Full experiments are never launched from this notebook.
    '''),
    md((ROOT/"docs/PLAN.md").read_text()),
    BOOTSTRAP, SETUP,
    code('''
    import torch, transformers
    torch.set_num_threads(1)
    print({"torch":torch.__version__, "transformers":transformers.__version__,
           "cuda":torch.cuda.is_available()})
    command("-m", "pytest", "-q")
    '''),
    code('''
    import tempfile
    # New output version on every execution; immutable caches cannot be overwritten.
    smoke_root = Path(tempfile.mkdtemp(prefix="notebook-smoke-", dir=ROOT/"artifacts"))
    command("-m", "gasp", "smoke", "--output", smoke_root)
    report = json.loads((smoke_root/"comparison/summary.json").read_text())
    print("Synthetic metrics only:", json.dumps(report["summary"],indent=2))
    print("Saved:", smoke_root)
    '''),
    code('''
    # Explicit resume at an interior block position; total schedule is unchanged.
    from gasp.config import Config
    from gasp.trainer import Trainer
    cfg = Config.load(ROOT/"configs/smoke.json").with_changes(condition="split")
    resumed_dir = smoke_root/"resume-demo"
    t = Trainer(cfg, smoke_root/"data/train.jsonl", smoke_root/"data/eval.jsonl", resumed_dir)
    t.run(stop_after=3)
    checkpoint = t.output/"checkpoint.pt"
    del t
    t = Trainer(cfg, smoke_root/"data/train.jsonl", smoke_root/"data/eval.jsonl", resumed_dir)
    t.resume(checkpoint)
    assert t.global_step == 3 and t.role == "polluter" and t.block_position == 1
    t.run()
    print("Resume complete:", t.role_counts)
    del t
    '''),
    md('''## What this does and does not establish
    Passing verifies parameter ownership, frozen snapshots, detached opponents,
    guidance conditioning, cache identity and resume mechanics. It establishes no
    mathematical reasoning or robustness advantage. Continue to `01_prepare_data`
    for full 1.5B tuning on a sufficiently large bf16-capable GPU (plan for 80GB).
    ''')])

save("01_prepare_data.ipynb",[
    md('''# Prepare training data and held-out evaluation
    Default: paper-stated settings plus the shared assumptions in docs/ASSUMPTIONS.md.
    The short pilot uses the same algorithm
    and memory requirements, with fewer updates. Generate one held-out GPT-6.1 Sol
    corpus (the user-selected change from the paper's GPT-5), or import existing
    held-out examples. Both conditions use the same corpus.
    See `docs/CORPUS.md` for the paper's procedure and explicit construction choices.
    '''),BOOTSTRAP,SETUP,DRIVE,
    code('''
    import torch
    from gasp.config import Config
    PROFILE = "full"  # "pilot" only shortens training; it does not enable LoRA
    assert PROFILE in ("pilot", "full")
    cfg = Config.load(ROOT/f"configs/{PROFILE}.json")
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Use a bf16-capable GPU; notebook00 is CPU-only"
    print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory/2**30, "GiB")
    RAW = WORK/"data/raw"
    DATA = WORK/f"data/{PROFILE}-v4"
    print("Immutable cache version:", DATA)
    '''),
    code('''
    if not (RAW/"math_raw.jsonl").exists():
        command("-m", "gasp", "download", "--output", RAW)
    if not (DATA/"train.jsonl").exists():
        command("-m", "gasp", "prepare-train", "--config", ROOT/f"configs/{PROFILE}.json",
                "--source", RAW/"math_raw.jsonl", "--output", DATA/"train.jsonl",
                "--candidates", 7500)
    print(json.loads(Path(str(DATA/"train.jsonl")+".manifest.json").read_text()))
    '''),
    code('''
    import shlex
    # Enable after setting OPENAI_API_KEY in this runtime. The API stage is paid.
    RUN_CORPUS_GENERATION = False
    MAX_NEW_REQUESTS = 4  # Inspect a first batch, then set None and rerun to finish.
    CONTEXTS = DATA/"eval-contexts.jsonl"
    EXTERNAL = DATA/"held-out-corruptions.jsonl"
    PROVENANCE = Path(str(EXTERNAL)+".provenance.json")
    prepare = ["-m", "gasp", "prepare-eval-contexts", "--config", DATA/"resolved_config.json",
               "--source", RAW/"gsm_raw.jsonl", "--output", CONTEXTS]
    generate = ["-m", "gasp", "generate-eval", "--source", CONTEXTS, "--output", EXTERNAL]
    if MAX_NEW_REQUESTS is not None:
        generate += ["--max-requests", MAX_NEW_REQUESTS]
    for args in (prepare, generate):
        print(shlex.join([sys.executable, *map(str,args)]))
    if RUN_CORPUS_GENERATION and not EXTERNAL.exists():
        if not CONTEXTS.exists():
            command(*prepare)
        command(*generate)
    '''),
    code('''
    # Inspect generated clean/polluted pairs first. For an existing author corpus,
    # set EXTERNAL and PROVENANCE above to its paths instead.
    if not (DATA/"eval.jsonl").exists():
        if not EXTERNAL.exists() or not PROVENANCE.exists():
            raise FileNotFoundError("Finish corpus generation or supply a held-out corpus; see docs/CORPUS.md")
        command("-m", "gasp", "import-eval", "--config", DATA/"resolved_config.json",
                "--source", EXTERNAL, "--provenance", PROVENANCE,
                "--output", DATA/"eval.jsonl")
    from gasp.data import read_cache, assert_disjoint
    train, tm = read_cache(DATA/"train.jsonl", "train")
    evaluation, em = read_cache(DATA/"eval.jsonl", "eval")
    assert_disjoint(train, evaluation)
    print("Training questions:", len(train), "Evaluation contexts:", len(evaluation))
    print("Protocol:", em["provenance"])
    print("Fixed clean-solved contexts:", sum(r["initial_clean_solved"] for r in evaluation))
    ''')])

save("02_train.ipynb",[
    md('''# Train unified and split policies
    Default: full tuning, paper-stated settings and documented shared assumptions.
    Both conditions run this codebase; our unified run is the comparison baseline.
    Run notebook01 first and use
    the same WORK/DATA location. The short pilot only reduces update count.
    Commands are printed until RUN_TRAINING is enabled; no GPU is provisioned.

    Unified shares parameters and optimizer moments. Split has separate parameters
    and moments; the inactive live model supplies opponent rollouts. Only unified
    needs a frozen opponent snapshot. Both use one global learning-rate schedule.
    '''),BOOTSTRAP,SETUP,DRIVE,
    code('''
    PROFILE = "full"
    DATA = WORK/f"data/{PROFILE}-v4"
    RUNS = WORK/f"runs/{PROFILE}-v4"
    from gasp.config import Config
    cfg = Config.load(DATA/"resolved_config.json")
    print(cfg)
    '''),
    code('''
    import shlex
    RUN_TRAINING = False
    SEEDS = [42, 52, 62] if PROFILE == "full" else [42]
    # Separate processes release GPU allocations and reinitialize each seed.
    for seed in SEEDS:
        for condition in ("unified", "split"):
            out = RUNS/f"{condition}-{seed}"
            args=["-m", "gasp", "train", "--config", DATA/"resolved_config.json",
                  "--condition", condition, "--seed", seed,
                  "--train-cache", DATA/"train.jsonl", "--eval-cache", DATA/"eval.jsonl",
                  "--output", out]
            if (out/"checkpoint.pt").exists():
                args += ["--resume", out/"checkpoint.pt"]
            print(shlex.join([sys.executable, *map(str,args)]))
            if RUN_TRAINING:
                command(*args)
    '''),
    code('''
    for seed in SEEDS:
        for condition in ("unified", "split"):
            out=RUNS/f"{condition}-{seed}"
            if not (out/"training.jsonl").exists():
                continue
            history=[json.loads(x) for x in (out/"training.jsonl").read_text().splitlines()]
            contexts=[c for h in history for c in h["contexts"]]
            print(condition, seed, "clean generation attempts:", sum(c["clean_attempts"] for c in contexts),
                  "informative samples:", sum(c["informative_samples"] for c in contexts))
            print(json.dumps(json.loads((out/"telemetry.json").read_text()), indent=2))
    '''),
    md('''Clean generation retries can cost different amounts in each condition.
    A failure to obtain a correct trace stops the run; investigate before resuming
    from the last checkpoint. No cached trace or synthetic corruption is substituted.
    Pre-fix checkpoints use a different protocol and cannot be resumed here.
    ''')])

save("03_results.ipynb",[
    md('''# Paired evaluation, costs and learning curves
    No training is launched. The primary comparison uses the same fixed corrupted
    contexts for every agent. Never compare each agent against its own polluter.
    Report split minus unified from our runs; the paper's score is background context.
    Shared choices are recorded in docs/ASSUMPTIONS.md. Smoke/pilot values are
    diagnostics and cannot establish the full comparison's scientific conclusion.
    '''),BOOTSTRAP,DRIVE,
    code('''
    PROFILE = "full"
    SEEDS = [42] if PROFILE=="pilot" else [42,52,62]
    RUNS = WORK/f"runs/{PROFILE}-v4"
    run_paths=[RUNS/f"{condition}-{seed}" for seed in SEEDS for condition in ("unified","split")]
    from gasp.evaluation import aggregate
    report=aggregate(run_paths,RUNS/"comparison",allow_pilot=PROFILE=="pilot")
    print("Protocol:",report["protocol"])
    print(json.dumps(report["summary"],indent=2))
    '''),
    code('''
    import pandas as pd
    from IPython.display import display
    rows=[]
    for pair in report["per_seed"]:
        for metric in pair["unified"]:
            rows.append({"seed":pair["seed"],"metric":metric,
                **{c:100*pair[c][metric] if pair[c][metric] is not None else None
                   for c in ("unified","split","split_minus_unified")}})
    display(pd.DataFrame(rows))  # percentage points
    costs=[]
    for pair in report["per_seed"]:
        for condition in ("unified","split"):
            t=pair[f"cost_{condition}"]
            costs.append({"seed":pair["seed"],"condition":condition,
                "generated_tokens":sum(v["generated_tokens"] for v in t["rollouts"].values()),
                "training_seconds":t["training_wall_seconds"],
                "gpu_interval_seconds":t["cuda_elapsed_interval_seconds"],
                "peak_gpu_GiB":t["peak_gpu_allocated_bytes"]/2**30,
                "parameter_GiB":t["unique_role_parameter_bytes"]/2**30,
                "optimizer_GiB":t["optimizer_bytes"]/2**30})
    display(pd.DataFrame(costs))
    '''),
    code('''
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(12,4))
    for path in run_paths:
        history=[json.loads(x) for x in (path/"training.jsonl").read_text().splitlines()]
        tokens={0:0,**{h["step"]:sum(v["generated_tokens"] for v in h["telemetry"]["rollouts"].values()) for h in history}}
        metrics=[json.loads(p.read_text()) for p in sorted((path/"eval").glob("step-*/metrics.json"))]
        for metric,style in [("corrupted_initial_clean_solved","-"),("clean_full","--")]:
            y=[100*m["metrics"][metric] if m["metrics"][metric] is not None else float("nan") for m in metrics]
            axes[0].plot([m["step"] for m in metrics],y,style,label=f"{path.name} {metric}")
            axes[1].plot([tokens[m["step"]] for m in metrics],y,style,label=f"{path.name} {metric}")
    axes[0].set_xlabel("Global optimizer updates");axes[1].set_xlabel("Actual training-related generated tokens")
    for ax in axes:ax.set_ylabel("Accuracy (%)");ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"{report['label']} — {report['protocol']}")
    fig.tight_layout();fig.savefig(RUNS/"comparison/learning-curves.png",dpi=160)
    plt.show()
    '''),
    code('''
    # Inspect paired disagreements instead of only looking at means.
    path=RUNS/"comparison/disagreements.jsonl"
    disagreements=[json.loads(x) for x in path.read_text().splitlines()]
    display(pd.DataFrame(disagreements).head(20))
    '''),
    md('''## Interpretation
    Report per-seed scores, means, sample standard deviations and paired differences.
    Three seeds give limited precision. Equal GRPO/repair rollout allocation does not guarantee
    equal actual tokens or GPU time; report both. Preparation costs are shared and
    stored in data manifests, evaluation costs in each metrics file.

    Published R1-1.5B GASP results provide context only: clean70.3±2.4,
    clean-solved recoverability95.6±2.2 (Table1), full-set recoverability78.5±1.2
    (Table2). Author artifacts and target-model runs are still needed before comparing numbers.
    ''')])

save("04_optional_benchmarks.ipynb",[
    md('''# Optional secondary robustness metrics
    These tools reuse public benchmark data schemas. Their local inference protocol
    is labeled modified. No paid scoring API is called. ACCreason requires a correct
    first error **and** a correct explanation: it remains null until you supply
    provenance-bearing human/official-judge decisions. Merely identifying a step
    is not reported as ACCreason.
    '''),BOOTSTRAP,DRIVE,
    code('''
    RUN = WORK/"runs/full-v4/split-42"
    FETCH_PUBLIC_BENCHMARKS = False
    if FETCH_PUBLIC_BENCHMARKS:
        for name,url in [("MR-GSM8K","https://github.com/JIA-Lab-research/MR-GSM8K"),
                         ("Mr-Ben","https://github.com/JIA-Lab-research/Mr-Ben"),
                         ("RUPBench","https://github.com/EternityYW/RUPBench")]:
            dest=WORK/"references"/name
            if not dest.exists():subprocess.run(["git","clone","--depth","1",url,str(dest)],check=True)
            print(name,subprocess.check_output(["git","-C",str(dest),"rev-parse","HEAD"],text=True).strip())
    '''),
    md('''Use the exact upstream file you inspected; record its git commit. File names
    can change. The adapters fail on unexpected schemas instead of guessing labels.
    For MHPP choose only the coding subset, not the entire Mr-Ben collection.
    '''),code('''
    # Set paths and enable only after obtaining the real files.
    PREPARE_SECONDARY = False
    if PREPARE_SECONDARY:
        from gasp.benchmarks import prepare_mr, prepare_rup
        mr_repo=WORK/"references/MR-GSM8K"
        revision=subprocess.check_output(["git","-C",str(mr_repo),"rev-parse","HEAD"],text=True).strip()
        prepare_mr(mr_repo/"dataset/MR-GSM8K.json",WORK/"data/secondary/mr.jsonl",
                   "mr-gsm8k","https://github.com/JIA-Lab-research/MR-GSM8K",revision)
        # prepare_mr(mhpp_file, ..., "mhpp", mr_ben_url, exact_commit)
        # prepare_rup(gsm_csv, ..., "gsm8k", rupbench_url, exact_commit)
        # prepare_rup(cqa_csv, ..., "cqa", rupbench_url, exact_commit)
    '''),code('''
    RUN_SECONDARY = False
    if RUN_SECONDARY:
        command("-m","gasp.benchmarks","evaluate","--run",RUN,
                "--cache",WORK/"data/secondary/mr.jsonl","--output",RUN/"secondary/mr")
        command("-m","gasp.benchmarks","self-revision","--run",RUN,
                "--cache",WORK/"data/full-v4/eval.jsonl","--output",RUN/"secondary/revision")
    '''),
    md('''The diagnostic output includes `judge_template.json`. Supply a descriptive
    `judge_provenance` and binary `explanation_correct` decisions, retaining prediction
    hashes, then score with the command below. Do this for both conditions using the
    same judging process. MHPP initial-clean-solved filtering needs a coding verifier
    and is not provided; report full-set diagnosis separately. Self-revision uses each
    agent's own failed set as a secondary metric, not the primary sharing comparison.
    '''),code('''
    SCORE_EXPLANATIONS = False
    if SCORE_EXPLANATIONS:
        command("-m","gasp.benchmarks","score-diagnosis",
                "--predictions",RUN/"secondary/mr/predictions.jsonl",
                "--judge",RUN/"secondary/mr/judge-completed.json",
                "--output",RUN/"secondary/mr/reason-score.json")
    ''')])

print("Generated",len(list((ROOT/"notebooks").glob("*.ipynb"))),"notebooks")
