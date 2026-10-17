"""Optional public benchmark adapters. Never conflate step accuracy with ACCreason.

These adapters reuse published data schemas; the local inference harness is a
modified protocol. Explanation grading requires external, provenance-bearing
human or official-judge decisions; this module never invokes a paid service.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

from .config import Config
from .data import write_cache, read_cache, write_json, sha256
from .models import load_model, Rollouts, rng_state, restore_rng, seed_all
from .prompts import Prompts, DIAGNOSIS, AGENT, REVISION
from .verification import correct


def load_json_rows(path):
    text=Path(path).read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(s) for s in text.splitlines() if s.strip()]


def prepare_mr(source, output, kind, source_url, revision):
    rows=[]
    for i,r in enumerate(load_json_rows(source)):
        if kind=="mr-gsm8k":
            q=r["question"];steps=r["model_output_steps"]
            judgement=r["model_output_solution_correctness"]
            first=r["model_output_solution_first_error_step"]
            reason=r["model_output_solution_first_error_reason"]
            identity=r.get("uuid",str(i))
        else:
            q=r["Question"];steps=r["Model_Solution_Steps"]
            judgement=r["Model_Solution_Correctness"]
            first=r.get("Model_Solution_Rectified_First_Error_Step",r["Model_Solution_First_Error_Step"])
            reason=r["Model_Solution_Error_Reason"]
            identity=str(r["Question_UUID"])+"-"+str(r["Sampled_Model"])
        solution="\n".join(f"{j+1}. {s}" for j,s in enumerate(steps)) if isinstance(steps,list) else str(steps)
        rows.append({"id":str(identity),"question":q,"solution":solution,
            "gold_correct":str(judgement).lower()=="correct","gold_step":str(first),
            "gold_reason":reason,"benchmark":kind})
    write_cache(output,rows,{"kind":"diagnosis","source_url":source_url,"revision":revision,
        "input_sha256":sha256(source),"protocol":"modified-local-appendix-C-full-set"})


def prepare_rup(source, output, benchmark, source_url, revision):
    rows=[]
    with open(source,newline="") as f:
        reader=csv.DictReader(f)
        if "Question" not in reader.fieldnames or "Answer" not in reader.fieldnames:
            raise ValueError("RUPBench CSV must include Question and Answer; inspect upstream schema")
        cols=[c for c in reader.fieldnames if c.startswith("Question_")]
        if not cols:raise ValueError("No perturbation columns")
        for i,r in enumerate(reader):
            options="\n".join(f"{x}: {r['Option '+x]}" for x in "ABCDE" if r.get("Option "+x))
            for col in cols:
                if not r[col].strip():raise ValueError(f"Missing perturbation at row {i}, {col}")
                rows.append({"id":f"{benchmark}-{i}-{col}","question":r[col]+("\n"+options if options else ""),
                    "original_question":r["Question"],"answer":r["Answer"],"perturbation":col,
                    "benchmark":benchmark,"verifier":"choice" if options else "numeric"})
    write_cache(output,rows,{"kind":"reliability","source_url":source_url,"revision":revision,
        "input_sha256":sha256(source),"protocol":"modified-local-agent-prompt-RUPBench-full-set"})


def load_agent(run):
    import torch
    path=Path(run)/"checkpoint.pt"
    if sha256(path)!=json.loads(Path(str(path)+".sha256.json").read_text())["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    state=torch.load(path,map_location="cpu",weights_only=False)
    cfg=Config(**state["config"])
    model,tok=load_model(cfg)
    model.load_state_dict(state["roles"]["agent"])
    return model,tok,cfg


def evaluate_benchmark(run, cache, output):
    model,tok,cfg=load_agent(run)
    rows,manifest=read_cache(cache)
    kind=manifest["provenance"]["kind"]
    p=Prompts(tok,cfg);rollout=Rollouts(tok,cfg)
    seed_all(cfg.eval_seed)
    outputs=[]
    for r in rows:
        if kind=="diagnosis":
            text=p.chat("Act as a grade school math teacher and score the following problem solution.",
                        DIAGNOSIS.format(q=r["question"],solution=r["solution"]))
            ids=tok.encode(text,add_special_tokens=False)
        elif kind=="reliability":
            ids=p.agent(r["question"])
        else:raise ValueError("Unsupported benchmark kind")
        if len(ids)>cfg.prompt_tokens:
            raise ValueError("Benchmark prompt exceeds cap; use a separately labeled paired evaluation config")
        sample=rollout.generate(model,ids,1,cfg.completion_tokens,"benchmark",temperature=0)[0]
        result=r|{"prediction":sample.text,"input_sha256":manifest["sha256"]}
        if kind=="diagnosis":
            result.update(parse_diagnosis(sample.text))
        else:
            result["correct"]=correct(sample.text,r["answer"],r["verifier"])
        outputs.append(result)
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    (directory/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in outputs))
    metrics={"protocol":manifest["provenance"]["protocol"],"n":len(outputs),"counts":rollout.counts,
             "cache_sha256":manifest["sha256"]}
    if kind=="reliability":
        metrics["accuracy"]=statistics.mean(r["correct"] for r in outputs)
        metrics["by_perturbation"]={key:statistics.mean(r["correct"] for r in outputs if r["perturbation"]==key)
                                     for key in sorted({r["perturbation"] for r in outputs})}
    else:
        metrics.update(score_diagnosis(outputs))
        # Blank judge records are deliberately NOT scored as incorrect explanations.
        write_json(directory/"judge_template.json",{"judge_provenance":None,"decisions":[
            {"id":r["id"],"prediction_sha256":__import__('hashlib').sha256(r["prediction"].encode()).hexdigest(),
             "explanation_correct":None} for r in outputs if not r["gold_correct"]]})
    write_json(directory/"metrics.json",metrics)
    return metrics


def parse_diagnosis(text):
    def get(pattern):
        m=re.search(pattern,text,re.S|re.I)
        return m.group(1).strip().strip("*[] ") if m else ""
    judgement=get(r"Final Judgement:\s*(.*?)\s*First Error Step:").lower()
    step=get(r"First Error Step:\s*(.*?)\s*Error Analysis:")
    number=re.search(r"\d+",step)
    return {"predicted_correct":True if judgement=="correct" else False if judgement in ("wrong","incorrect") else None,
            "predicted_step":number.group() if number else "N/A",
            "predicted_reason":get(r"Error Analysis:\s*(.*)")}


def score_diagnosis(rows, judge=None):
    wrong=[r for r in rows if not r["gold_correct"]]
    if not wrong:raise ValueError("ACCreason denominator needs incorrect solutions")
    step_ok=lambda r:r["predicted_correct"] is False and str(r["predicted_step"])==str(r["gold_step"])
    result={"step_accuracy":sum(step_ok(r) for r in wrong)/len(wrong),
            "incorrect_solution_count":len(wrong),"ACCreason":None,
            "reason_status":"Not scored: requires human/official judge correctness decisions"}
    if judge is not None:
        if not judge.get("judge_provenance"):raise ValueError("Missing judge provenance")
        import hashlib
        decisions={d["id"]:d for d in judge["decisions"]}
        for r in wrong:
            d=decisions.get(r["id"],{})
            if type(d.get("explanation_correct")) is not bool:
                raise ValueError("Missing binary explanation judgement")
            if d.get("prediction_sha256")!=hashlib.sha256(r["prediction"].encode()).hexdigest():
                raise ValueError("Judge decision does not match this prediction")
        result.update(ACCreason=sum(step_ok(r) and decisions[r["id"]]["explanation_correct"] for r in wrong)/len(wrong),
                      reason_status="scored",judge_provenance=judge["judge_provenance"])
    return result


def self_revision(run, cache, output):
    model,tok,cfg=load_agent(run)
    rows,manifest=read_cache(cache,"eval")
    p=Prompts(tok,cfg);rollout=Rollouts(tok,cfg)
    seed_all(cfg.eval_seed)
    outputs=[];seen=set()
    for r in rows:
        if r["question_id"] in seen:continue
        seen.add(r["question_id"])
        first=rollout.generate(model,r["clean_prompt_ids"],1,cfg.completion_tokens,"revision_initial",temperature=0)[0]
        first_correct=correct(first.text,r["answer"],cfg.verifier)
        record={"id":r["question_id"],"initial":first.text,"initial_correct":first_correct}
        if not first_correct:
            ids=tok.encode(p.chat(AGENT,REVISION.format(q=r["question"],solution=first.text)),add_special_tokens=False)
            # Do not silently truncate the erroneous solution being evaluated.
            if len(ids)>cfg.prompt_tokens:
                record.update(revision=None,revision_correct=0,error="overlength-revision-prompt")
            else:
                revised=rollout.generate(model,ids,1,cfg.completion_tokens,"self_revision",temperature=0)[0]
                record.update(revision=revised.text,revision_correct=correct(revised.text,r["answer"],cfg.verifier))
        outputs.append(record)
    failures=[r for r in outputs if not r["initial_correct"]]
    recoveries=sum(r["revision_correct"] for r in failures)
    result={"protocol":"modified-Appendix-D-own-failed-subset-greedy-overlength-counted-failure",
            "n":len(outputs),"failed_count":len(failures),
            "revision_success_on_failed":recoveries/len(failures) if failures else None,
            "full_set_accuracy_gain":recoveries/len(outputs),"counts":rollout.counts,
            "eval_sha256":manifest["sha256"]}
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    (directory/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in outputs))
    write_json(directory/"metrics.json",result)
    return result


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare-mr");p.add_argument("--kind",choices=["mr-gsm8k","mhpp"],required=True)
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--source-url",required=True);p.add_argument("--revision",required=True)
    p=sub.add_parser("prepare-rup");p.add_argument("--benchmark",choices=["gsm8k","cqa"],required=True)
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--source-url",required=True);p.add_argument("--revision",required=True)
    for command in ("evaluate","self-revision"):
        p=sub.add_parser(command);p.add_argument("--run",required=True)
        p.add_argument("--cache",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("score-diagnosis");p.add_argument("--predictions",required=True)
    p.add_argument("--judge",required=True);p.add_argument("--output",required=True)
    a=parser.parse_args()
    if a.command=="prepare-mr":prepare_mr(a.source,a.output,a.kind,a.source_url,a.revision)
    elif a.command=="prepare-rup":prepare_rup(a.source,a.output,a.benchmark,a.source_url,a.revision)
    elif a.command=="evaluate":print(evaluate_benchmark(a.run,a.cache,a.output))
    elif a.command=="self-revision":print(self_revision(a.run,a.cache,a.output))
    else:write_json(a.output,score_diagnosis(load_json_rows(a.predictions),json.loads(Path(a.judge).read_text())))


if __name__=="__main__":main()
