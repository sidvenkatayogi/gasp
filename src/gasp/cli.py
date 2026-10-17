import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Controlled GASP experiment; no cloud provisioning")
    sub = parser.add_subparsers(dest="command",required=True)
    def config_arg(p):
        p.add_argument("--config",required=True)
    p=sub.add_parser("download");p.add_argument("--output",required=True)
    p.add_argument("--math-repo",default="DigitalLearningGmbH/MATH-lighteval")
    p=sub.add_parser("prepare-train");config_arg(p)
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--candidates",type=int,default=7500)
    p=sub.add_parser("prepare-eval");config_arg(p)
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--provenance",required=True,
                   help="Held-out corpus provenance; imports existing windows without generation")
    p=sub.add_parser("import-eval");config_arg(p)
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--provenance",required=True)
    p=sub.add_parser("prepare-eval-contexts");config_arg(p)
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--questions",type=int,help="Optional first-N question limit for preparation pilots")
    p.add_argument("--resume",action="store_true",help="Resume a trusted preparation checkpoint")
    p.add_argument("--max-seconds",type=float,help="Stop cleanly after a question when the time budget expires")
    p=sub.add_parser("generate-eval",help="Calls GPT-6.1 Sol; needs OPENAI_API_KEY and prepared clean contexts")
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--model",choices=["gpt-6.1-sol"],default="gpt-6.1-sol")
    p.add_argument("--effort",choices=["low","medium","high","xhigh","max"],default="low")
    p.add_argument("--max-output-tokens",type=int,default=4096)
    p.add_argument("--max-requests",type=int,help="Stop after this many new API calls; rerun to resume")
    p=sub.add_parser("train");config_arg(p)
    p.add_argument("--condition",choices=["unified","split"])
    p.add_argument("--seed",type=int)
    p.add_argument("--train-cache",required=True);p.add_argument("--eval-cache",required=True)
    p.add_argument("--output",required=True);p.add_argument("--resume")
    p.add_argument("--stop-after",type=int,help="Absolute global step to stop at; schedule remains unchanged")
    p=sub.add_parser("evaluate")
    p.add_argument("--run",required=True);p.add_argument("--train-cache",required=True)
    p.add_argument("--eval-cache",required=True)
    p=sub.add_parser("export-agent");p.add_argument("--run",required=True)
    p.add_argument("--output",required=True)
    p=sub.add_parser("aggregate")
    p.add_argument("--runs",nargs="+",required=True);p.add_argument("--output",required=True)
    p.add_argument("--allow-pilot",action="store_true")
    p=sub.add_parser("smoke")
    p.add_argument("--config",default="configs/smoke.json");p.add_argument("--output",default="runs/smoke")
    args=parser.parse_args()
    from . import data
    if args.command=="download":
        data.download_sources(args.output,math_repo=args.math_repo)
    elif args.command=="prepare-train":
        data.prepare_train(Config.load(args.config),args.source,args.output,args.candidates)
    elif args.command in ("prepare-eval", "import-eval"):
        data.import_eval(Config.load(args.config),args.source,args.output,args.provenance)
    elif args.command=="prepare-eval-contexts":
        from .eval_corpus import prepare_eval_contexts
        print(json.dumps(prepare_eval_contexts(Config.load(args.config),args.source,args.output,
                                              args.questions,args.resume,args.max_seconds),indent=2))
    elif args.command=="generate-eval":
        from .eval_corpus import generate_eval_corpus
        print(json.dumps(generate_eval_corpus(args.source,args.output,args.model,args.effort,
                                              args.max_output_tokens,args.max_requests),indent=2))
    elif args.command=="train":
        from .trainer import Trainer
        cfg=Config.load(args.config,condition=args.condition,seed=args.seed)
        checkpoint=Path(args.output)/"checkpoint.pt"
        if checkpoint.exists() and not args.resume:
            raise ValueError("Run exists; pass --resume or use a new output directory")
        trainer=Trainer(cfg,args.train_cache,args.eval_cache,args.output)
        if args.resume:
            trainer.resume(args.resume)
        trainer.run(stop_after=args.stop_after)
    elif args.command=="evaluate":
        from .trainer import Trainer
        cfg=Config(**json.loads(Path(args.run,"run.json").read_text())["config"])
        trainer=Trainer(cfg,args.train_cache,args.eval_cache,args.run)
        trainer.resume(Path(args.run)/"checkpoint.pt")
        print(json.dumps(trainer.evaluate(),indent=2))
    elif args.command=="export-agent":
        import torch
        from .models import load_model
        from .data import sha256
        root=Path(args.run); path=root/"checkpoint.pt"
        if sha256(path)!=json.loads(Path(str(path)+".sha256.json").read_text())["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        state=torch.load(path,map_location="cpu",weights_only=False)
        cfg=Config(**state["config"])
        model,tok=load_model(cfg)
        model.load_state_dict(state["roles"]["agent"])
        model.save_pretrained(args.output);tok.save_pretrained(args.output)
        data.write_json(Path(args.output)/"experiment.json",state["run_manifest"])
    elif args.command=="aggregate":
        from .evaluation import aggregate
        print(json.dumps(aggregate(args.runs,args.output,allow_pilot=args.allow_pilot)["summary"],indent=2))
    elif args.command=="smoke":
        import torch
        from .trainer import Trainer
        from .evaluation import aggregate
        torch.set_num_threads(1)
        cfg=Config.load(args.config)
        root=Path(args.output)
        data.make_smoke_data(root/"data",cfg)
        for condition in ("unified","split"):
            trainer=Trainer(cfg.with_changes(condition=condition),root/"data/train.jsonl",
                            root/"data/eval.jsonl",root/condition)
            trainer.run()
            del trainer;gc.collect()
        report=aggregate([root/"unified",root/"split"],root/"comparison",allow_pilot=True)
        print(json.dumps(report["summary"],indent=2))


if __name__=="__main__":
    main()
