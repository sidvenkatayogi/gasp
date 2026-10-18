import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import time
from importlib.metadata import version, PackageNotFoundError, distributions

import torch

from .config import Config
from .data import (read_cache, assert_disjoint, cut_window, write_json, sha256,
                   validate_eval_provenance)
from .losses import advantages, clipped_loss
from .models import (Roles, Rollouts, load_model, token_logps, rng_state, restore_rng,
                     model_digest, seed_all)
from .prompts import Prompts
from .verification import correct, reasoning, pollution, polluter_reward, extract_answer, repair_target


def storage_bytes(tree):
    if torch.is_tensor(tree):
        return tree.numel() * tree.element_size()
    if isinstance(tree, dict):
        return sum(storage_bytes(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(storage_bytes(v) for v in tree)
    return 0


def schedule_multiplier(step, total, warmup):
    # LambdaLR step zero supplies the learning rate for the first update.
    warm = int(total * warmup)
    if step < warm:
        return (step + 1) / max(1, warm)
    progress = (step - warm) / max(1, total - warm)
    return .5 * (1 + math.cos(math.pi * min(1., progress)))


class Trainer:
    def __init__(self, config, train_cache, eval_cache, output, model=None, tokenizer=None):
        self.cfg = config
        self.train_rows, self.train_manifest = read_cache(train_cache, "train")
        self.eval_rows, self.eval_manifest = read_cache(eval_cache, "eval")
        assert_disjoint(self.train_rows, self.eval_rows)
        self._check_cache_config()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output/"run.json").exists():
            previous = json.loads((self.output/"run.json").read_text())
            if previous["config"] != asdict(config):
                raise ValueError("Existing output has a different config; use a new directory")
        if model is None:
            model, tokenizer = load_model(config)
        self.tokenizer = tokenizer
        self.roles = Roles(model, config.condition)
        self.prompts = Prompts(tokenizer, config)
        self.rollouts = Rollouts(tokenizer, config)
        # AdamW owns one state per unique parameter. Unified shares moments along
        # with weights; split has independent moments because its weights differ.
        # Inactive split parameters have grad=None, so AdamW does not touch them.
        params = {id(p): p for role in ("agent", "polluter")
                  for p in self.roles.active(role).parameters() if p.requires_grad}
        self.optimizer = torch.optim.AdamW(list(params.values()), lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
            weight_decay=config.weight_decay, foreach=False)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            lambda step: schedule_multiplier(step, 2*config.role_steps, config.warmup_ratio))
        self.global_step = 0
        self.role = "agent"
        self.block_position = 0
        self.role_counts = {"agent": 0, "polluter": 0}
        self.data_rng = random.Random(config.seed)
        self.order = list(range(len(self.train_rows)))
        self.data_rng.shuffle(self.order)
        self.cursor = 0
        self.loss_tokens = {"agent": 0, "polluter": 0, "repair": 0}
        self.forward_train_tokens = 0
        self.wall_seconds = 0.
        self.eval_wall_seconds = 0.
        self.checkpoint_wall_seconds = 0.
        self.gpu_interval_seconds = 0.
        self.history = []
        self.last_evaluated = -1
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.run_manifest = {"config": asdict(config), "paired_config_sha256": config.paired_hash(),
            "train_sha256": self.train_manifest["sha256"], "eval_sha256": self.eval_manifest["sha256"],
            "eval_protocol": self.eval_manifest["provenance"]["protocol"],
            "initial_model_sha256": model_digest(model), "torch": torch.__version__,
            "python": platform.python_version(), "device": str(next(model.parameters()).device),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "source_sha256": self.source_digest()}
        import hashlib
        token_config = {"vocab": tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else tokenizer.vocab,
                        "chat_template": getattr(tokenizer, "chat_template", None),
                        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
        self.run_manifest["tokenizer_sha256"] = hashlib.sha256(json.dumps(token_config,sort_keys=True).encode()).hexdigest()
        self.run_manifest["packages"] = {}
        for package in ("torch","transformers","tokenizers","numpy","peft","accelerate","datasets","math-verify"):
            try:self.run_manifest["packages"][package]=version(package)
            except PackageNotFoundError:pass
        write_json(self.output/"environment.json", {d.metadata["Name"]:d.version for d in distributions()})
        write_json(self.output/"run.json", self.run_manifest)

    @staticmethod
    def source_digest():
        import hashlib
        h = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob("*.py")):
            h.update(path.name.encode())
            h.update(path.read_bytes())
        return h.hexdigest()

    def _check_cache_config(self):
        if not self.cfg.tiny:
            validate_eval_provenance(self.eval_manifest["provenance"])
        for m in (self.train_manifest, self.eval_manifest):
            prompt_hash = m["provenance"].get("prompt_sha256")
            if prompt_hash and prompt_hash != sha256(Path(__file__).with_name("prompts.py")):
                raise ValueError("Cached prompt source differs; prepare a new cache version")
            c = m["provenance"].get("config")
            if c:
                for key in ("model", "revision", "tiny", "prompt_tokens", "verifier"):
                    if c[key] != getattr(self.cfg, key):
                        raise ValueError(f"Cache/config mismatch for {key}")
        if not self.cfg.tiny and self.cfg.revision == "main":
            raise ValueError("Use resolved_config.json from preparation with immutable model revision")

    def next_row(self):
        if self.cursor == len(self.order):
            self.data_rng.shuffle(self.order)
            self.cursor = 0
        row = self.train_rows[self.order[self.cursor]]
        self.cursor += 1
        return row, self.data_rng.choice((0., .25, .5, .75))

    def place_optimizer_moments(self):
        """Keep the active role's moments on-device and inactive moments on CPU.

        Transfers occur at role changes, not between every update. Unified has
        no inactive independent parameters, so its shared moments stay put.
        Scalar Adam steps stay on CPU, as required by non-capturable AdamW.
        """
        if not self.cfg.optimizer_offload:
            return
        active = set(self.roles.active(self.role).parameters())
        # Evict first: never temporarily place both complete moment sets on GPU.
        for loading in (False, True):
            for parameter, state in self.optimizer.state.items():
                if (parameter in active) != loading:
                    continue
                device = parameter.device if loading else "cpu"
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key] = state[key].to(device)

    def make_context(self, row, alpha):
        # During a polluter block the designated agent is the fixed opponent.
        agent = self.roles.agent if self.role == "agent" else self.roles.opponent
        for attempt in range(1, self.cfg.max_clean_attempts + 1):
            sample = self.rollouts.generate(agent, self.prompts.agent(row["question"]),
                1, self.cfg.completion_tokens, self.role+"_clean_attempts")[0]
            trace = reasoning(sample.text)
            if trace and correct(sample.text, row["answer"], self.cfg.verifier):
                prefix, clean = cut_window(trace, self.tokenizer, alpha, self.cfg.window_cap)
                return prefix, clean, attempt
        raise RuntimeError(f"No correct on-policy clean trace for {row['id']} after "
                           f"{self.cfg.max_clean_attempts} attempts; no optimizer update was made. "
                           "Check generation/verifier settings; resume from the last checkpoint.")

    def backward_group(self, samples, rewards):
        adv = advantages(rewards, self.cfg.advantage_epsilon)
        model = self.roles.active(self.role)
        model.train()
        total = 0.
        for sample, a in zip(samples, adv):
            logps = token_logps(model, sample.prompt, sample.tokens, self.cfg)
            loss = clipped_loss(logps, sample.old_logps.to(logps.device), a, self.cfg.grpo_clip)
            # Equation (1): average over every trajectory in the sampled group.
            (loss / (len(samples) * self.cfg.accumulation)).backward()
            total += float(loss.detach()) / len(samples)
            self.loss_tokens[self.role] += len(sample.tokens)
            self.forward_train_tokens += len(sample.prompt) + len(sample.tokens)
        return total, int((adv != 0).sum())

    def agent_context(self, row, prefix, clean):
        c = self.cfg
        candidate = self.rollouts.generate(self.roles.opponent,
            self.prompts.polluter(row["question"], prefix, clean), 1, c.corruption_tokens,
            "agent_corruption")[0]
        window, tagged = pollution(candidate.text,
            self.tokenizer.decode(candidate.prompt, skip_special_tokens=False))
        deployment = self.prompts.agent(row["question"], prefix, window)
        samples = self.rollouts.generate(self.roles.agent, deployment, c.group_size,
            c.completion_tokens, "agent_completion", old=True)
        rewards = [correct(s.text, row["answer"], c.verifier) for s in samples]
        privileged = self.prompts.repair(row["question"], prefix, clean, window)
        repair = self.rollouts.generate(self.roles.agent, privileged, 1, c.repair_tokens, "repair_generation")[0]
        target = repair_target(repair.text, self.tokenizer,
            self.tokenizer.decode(repair.prompt, skip_special_tokens=False))
        grpo, informative = self.backward_group(samples, rewards)
        weight = c.guidance_weight(self.role_counts["agent"])
        # Target tokens come from current AGENT under privileged context; training
        # conditioning is exactly the same corrupted deployment prompt as GRPO.
        # Appendix A.2/A.3 write log pi(w_fix | s_poll), i.e. a sequence sum.
        # GRPO remains length-normalized; the auxiliary imitation term does not.
        nll = 0.
        if target and weight:
            logps = token_logps(self.roles.agent, deployment, target, c)
            loss = -logps.sum()
            (loss * weight / c.accumulation).backward()
            nll = float(loss.detach())
            self.loss_tokens["repair"] += len(target)
            self.forward_train_tokens += len(deployment) + len(target)
        return {"grpo_loss": grpo, "guidance_nll": nll, "guidance_weight": weight,
            "rewards": rewards, "polluted_tag_present": [tagged],
            "informative_samples": informative,
            "polluted": [window], "raw_corruption": candidate.text, "repair": repair.text,
            "repair_target_ids": target,
            "privileged_prompt_ids": privileged, "guidance_conditioning_ids": deployment}

    def polluter_context(self, row, prefix, clean):
        c = self.cfg
        samples = self.rollouts.generate(self.roles.polluter,
            self.prompts.polluter(row["question"], prefix, clean), c.group_size,
            c.corruption_tokens, "polluter_candidate", old=True)
        rewards, tags, windows, correctness, scoring_outputs = [], [], [], [], []
        scoring_prompts = []
        for sample in samples:
            window, tagged = pollution(sample.text,
                self.tokenizer.decode(sample.prompt, skip_special_tokens=False))
            scoring_prompts.append(self.prompts.agent(row["question"], prefix, window))
            tags.append(tagged)
            windows.append(window)
        completions = []
        for start in range(0, len(scoring_prompts), c.rollout_batch_size):
            batch = scoring_prompts[start:start + c.rollout_batch_size]
            if c.rollout_batch_size == 1:
                completions.append(self.rollouts.generate(self.roles.opponent, batch[0],
                    1, c.completion_tokens, "polluter_agent_scoring")[0])
            else:
                groups = self.rollouts.generate_many(self.roles.opponent, batch, 1,
                    c.completion_tokens, "polluter_agent_scoring")
                completions.extend(group[0] for group in groups)
        for completion in completions:
            score = correct(completion.text, row["answer"], c.verifier)
            rewards.append(polluter_reward(score))
            correctness.append(score)
            scoring_outputs.append(completion.text)
        loss, informative = self.backward_group(samples, rewards)
        return {"grpo_loss": loss, "rewards": rewards, "polluted_tag_present": tags,
                "informative_samples": informative, "polluted": windows,
                "raw_corruptions": [s.text for s in samples],
                "agent_correctness": correctness, "agent_scoring_outputs": scoring_outputs}

    def step(self):
        if self.global_step >= self.cfg.role_steps*2:
            raise StopIteration("Training complete")
        start = time.perf_counter()
        cuda = next(self.roles.agent.parameters()).is_cuda
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            event_start.record()
        if self.block_position == 0:
            self.roles.refresh(self.role)
        role = self.role
        # Remove stale grads from BOTH models, including the inactive split role.
        self.optimizer.zero_grad(set_to_none=True)
        self.place_optimizer_moments()
        entries = []
        for _ in range(self.cfg.accumulation):
            row, alpha = self.next_row()
            prefix, clean, attempts = self.make_context(row, alpha)
            result = (self.agent_context if role == "agent" else self.polluter_context)(row, prefix, clean)
            entries.append(result | {"question_id": row["id"], "alpha": alpha, "clean_attempts": attempts})
        params = [p for p in self.roles.active(role).parameters() if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient; refusing optimizer step")
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.roles.updated(role)
        self.role_counts[role] += 1
        self.global_step += 1
        self.block_position += 1
        if self.block_position == self.cfg.block_steps:
            self.block_position = 0
            self.role = "polluter" if role == "agent" else "agent"
        if cuda:
            event_end.record()
            torch.cuda.synchronize()
            self.gpu_interval_seconds += event_start.elapsed_time(event_end) / 1000
            self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
            self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())
        self.wall_seconds += time.perf_counter()-start
        entry = {"step": self.global_step, "role": role, "role_counts": self.role_counts.copy(),
            "snapshot_role": self.roles.snapshot_role, "snapshot_version": self.roles.snapshot_version,
            "live_versions": self.roles.versions.copy(), "lr": lr, "grad_norm": grad_norm,
            "contexts": entries, "telemetry": self.telemetry()}
        self.history.append(entry)
        # Checkpoint history is authoritative after interrupted steps; save rewrites log.
        with (self.output/"training.jsonl").open("a") as f:
            f.write(json.dumps(entry)+"\n")
        return entry

    def telemetry(self):
        unique = {id(p): p for role in ("agent", "polluter") for p in self.roles.active(role).parameters()}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"rollouts": copy.deepcopy(self.rollouts.counts), "loss_tokens": self.loss_tokens.copy(),
            "forward_training_tokens": self.forward_train_tokens, "training_wall_seconds": self.wall_seconds,
            "evaluation_wall_seconds": self.eval_wall_seconds,
            "checkpoint_wall_seconds": self.checkpoint_wall_seconds,
            "total_accounted_wall_seconds": self.wall_seconds+self.eval_wall_seconds+self.checkpoint_wall_seconds,
            "cuda_elapsed_interval_seconds": self.gpu_interval_seconds if torch.cuda.is_available() else None,
            "peak_gpu_allocated_bytes": self.peak_allocated, "peak_gpu_reserved_bytes": self.peak_reserved,
            "peak_process_rss_bytes": int(rss * (1 if platform.system() == "Darwin" else 1024)),
            "unique_role_parameter_bytes": sum(p.numel()*p.element_size() for p in unique.values()),
            "trainable_parameter_count": sum(p.numel() for p in unique.values() if p.requires_grad),
            "optimizer_bytes": storage_bytes(self.optimizer.state_dict()),
            "optimizer_cpu_bytes": sum(v.numel()*v.element_size() for state in self.optimizer.state.values()
                for v in state.values() if torch.is_tensor(v) and v.device.type == "cpu"),
            "optimizer_cuda_bytes": sum(v.numel()*v.element_size() for state in self.optimizer.state.values()
                for v in state.values() if torch.is_tensor(v) and v.device.type == "cuda"),
            "snapshot_bytes": storage_bytes(self.roles.snapshot.state_dict()) if self.roles.snapshot else 0}

    def save(self, path=None):
        start = time.perf_counter()
        path = Path(path or self.output/"checkpoint.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"schema": 2, "config": asdict(self.cfg), "run_manifest": self.run_manifest,
            "roles": self.roles.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "rng": rng_state(),
            "data_rng": self.data_rng.getstate(), "cursor": self.cursor, "order": self.order,
            "global_step": self.global_step, "role": self.role, "block_position": self.block_position,
            "role_counts": self.role_counts, "rollout_counts": self.rollouts.counts,
            "loss_tokens": self.loss_tokens, "forward_train_tokens": self.forward_train_tokens,
            "wall_seconds": self.wall_seconds, "gpu_interval_seconds": self.gpu_interval_seconds,
            "eval_wall_seconds": self.eval_wall_seconds, "checkpoint_wall_seconds": self.checkpoint_wall_seconds,
            "peak_allocated": self.peak_allocated, "peak_reserved": self.peak_reserved,
            "history": self.history, "last_evaluated": self.last_evaluated}
        temporary = path.with_suffix(path.suffix+".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        write_json(str(path)+".sha256.json", {"sha256": sha256(path), "step": self.global_step})
        self.checkpoint_wall_seconds += time.perf_counter()-start
        write_json(self.output/"telemetry.json", self.telemetry())
        return path

    def resume(self, path):
        # Only load checkpoints created by this project from trusted storage.
        expected = json.loads(Path(str(path)+".sha256.json").read_text())["sha256"]
        if sha256(path) != expected:
            raise ValueError("checkpoint checksum mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("schema") != 2:
            raise ValueError("Old training protocol checkpoint; start a fresh run with rebuilt caches")
        if state["config"] != asdict(self.cfg):
            raise ValueError("Resume config differs (including budgets/schedules)")
        for key in ("train_sha256", "eval_sha256", "source_sha256", "initial_model_sha256", "tokenizer_sha256", "packages"):
            if state["run_manifest"][key] != self.run_manifest[key]:
                raise ValueError("Resume mismatch: " + key)
        self.roles.load_state_dict(state["roles"])
        if self.cfg.optimizer_offload:
            # PyTorch casts loaded moments to the parameter device. Load while
            # parameters are on CPU so both models' states never fill CUDA at
            # once; model.to preserves these existing Parameter identities.
            models = {id(m): m for m in (self.roles.agent, self.roles.polluter,
                                         self.roles.snapshot) if m is not None}
            devices = {key: next(model.parameters()).device for key, model in models.items()}
            for model in models.values():
                model.to("cpu")
            self.optimizer.load_state_dict(state["optimizer"])
            for key, model in models.items():
                model.to(devices[key])
        else:
            self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        for name in ("cursor", "order", "global_step", "role", "block_position", "role_counts",
                     "loss_tokens", "forward_train_tokens", "wall_seconds", "gpu_interval_seconds",
                     "peak_allocated", "peak_reserved", "history", "last_evaluated",
                     "eval_wall_seconds", "checkpoint_wall_seconds"):
            setattr(self, name, state[name])
        self.rollouts.counts = state["rollout_counts"]
        self.data_rng.setstate(state["data_rng"])
        restore_rng(state["rng"])
        (self.output/"training.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.history))

    def evaluate(self):
        from .evaluation import evaluate
        result = evaluate(self.roles.agent, self.tokenizer, self.cfg, self.eval_rows,
                          self.eval_manifest, self.output, self.global_step, self.run_manifest)
        self.last_evaluated = self.global_step
        self.eval_wall_seconds += result["wall_seconds"]
        return result

    def run(self, stop_after=None, evaluate=True):
        if evaluate and self.last_evaluated < 0:
            self.evaluate()
        end = min(self.cfg.role_steps*2, stop_after) if stop_after is not None else self.cfg.role_steps*2
        while self.global_step < end:
            entry = self.step()
            print(f"step={entry['step']} role={entry['role']} grad_norm={entry['grad_norm']:.5g}", flush=True)
            if evaluate and (self.global_step % self.cfg.eval_every == 0 or self.global_step == self.cfg.role_steps*2):
                self.evaluate()
            if self.global_step % self.cfg.checkpoint_every == 0 or self.global_step == end:
                self.save()
        return self.telemetry()
