from contextlib import nullcontext
import copy
from dataclasses import dataclass
import hashlib
import random
import re

import numpy as np
import torch
from torch import nn


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint requires CUDA for exact RNG restoration")
        torch.cuda.set_rng_state_all(state["cuda"])


class TinyTokenizer:
    """Offline binary arithmetic smoke vocabulary; NOT a DeepSeek tokenizer."""
    vocab = ["<pad>", "<eos>", "<unk>", "0", "1", "agent", "polluter", "repair",
             "prefix", "clean", "polluted", "answer", "question", "train", "test",
             "a", "b", "c", "d", "e", "f", "g", "h", "=", "+", "2", "3",
             "4", "5", "6", "7", "8"]
    eos_token_id, pad_token_id = 1, 0

    def encode(self, text, add_special_tokens=False):
        lookup = {x: i for i, x in enumerate(self.vocab)}
        return [lookup.get(x, 2) for x in re.findall(r"\w+|[=+]", text)] or [2]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.vocab[int(x)] for x in ids if not skip_special_tokens or int(x) > 1)

    def save_pretrained(self, path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        Path(path, "tiny-tokenizer.txt").write_text("\n".join(self.vocab))


def device_for(config):
    return "cuda" if config.device == "auto" and torch.cuda.is_available() else (
        "cpu" if config.device == "auto" else config.device)


def load_model(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    seed_all(config.seed)
    device = device_for(config)
    if config.precision == "bf16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a supported CUDA GPU; use precision=fp32 for CPU/T4 pilots")
    if config.tiny:
        tok = TinyTokenizer()
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=256,
            bos_token_id=2, eos_token_id=1, pad_token_id=0, attention_dropout=0.0))
    else:
        tok = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        # Full-tune uses fp32 master weights; autocast controls compute precision.
        dtype = torch.bfloat16 if config.tuning == "lora" and config.precision == "bf16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(config.model, revision=config.revision,
            torch_dtype=dtype, attn_implementation="sdpa")
    if config.tuning == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=config.lora_rank,
            lora_alpha=2*config.lora_rank, lora_dropout=0., bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"))
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if config.tuning == "lora":
            model.enable_input_require_grads()
    model.to(device)
    return model, tok


def autocast(config):
    if config.precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def model_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Roles:
    def __init__(self, model, condition):
        self.agent = model
        self.polluter = model if condition == "unified" else copy.deepcopy(model)
        self.condition = condition
        self.snapshot = None
        self.snapshot_role = None
        self.snapshot_version = -1
        self.versions = {"agent": 0, "polluter": 0}

    def active(self, role):
        return getattr(self, role)

    @property
    def opponent(self):
        if self.snapshot_role is None:
            raise RuntimeError("Enter a training block before requesting its opponent")
        return self.snapshot if self.condition == "unified" else self.active(self.snapshot_role)

    def refresh(self, active_role):
        opposite = "polluter" if active_role == "agent" else "agent"
        # Release before copying to avoid a transient second full snapshot allocation.
        self.snapshot = None
        if self.condition == "unified":
            self.snapshot = copy.deepcopy(self.active(opposite)).eval()
            self.snapshot.requires_grad_(False)
            for p in self.snapshot.parameters():
                p.grad = None
            if hasattr(self.snapshot, "gradient_checkpointing_disable"):
                self.snapshot.gradient_checkpointing_disable()
        # Split's inactive model is already fixed for the block. Rollouts run
        # under no_grad/eval, without changing its trainable parameter flags.
        self.snapshot_role = opposite
        self.snapshot_version = self.versions[opposite]

    def updated(self, role):
        self.versions[role] += 1
        if self.condition == "unified":
            self.versions["polluter" if role == "agent" else "agent"] += 1

    def state_dict(self):
        return {"agent": self.agent.state_dict(),
                "polluter": self.polluter.state_dict() if self.condition == "split" else None,
                "snapshot": self.snapshot.state_dict() if self.snapshot is not None else None,
                "snapshot_role": self.snapshot_role, "snapshot_version": self.snapshot_version,
                "versions": self.versions}

    def load_state_dict(self, state):
        self.agent.load_state_dict(state["agent"])
        if self.condition == "split":
            self.polluter.load_state_dict(state["polluter"])
        if state["snapshot"] is not None:
            self.refresh("agent" if state["snapshot_role"] == "polluter" else "polluter")
            self.snapshot.load_state_dict(state["snapshot"])
        else:
            self.snapshot = None
        self.snapshot_role = state["snapshot_role"]
        self.snapshot_version = state["snapshot_version"]
        self.versions = state["versions"]


@dataclass
class Sample:
    prompt: list
    tokens: list
    text: str
    old_logps: torch.Tensor | None = None


def token_logps(model, prompt, completion, config):
    """Only returned continuation positions participate in loss; no prompt loss."""
    if not prompt or not completion:
        raise ValueError("nonempty prompt and continuation required")
    ids = torch.tensor([prompt + completion], device=next(model.parameters()).device)
    with autocast(config):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    logits = logits[0, len(prompt)-1:-1].float()
    if config.tiny:
        # Tiny smoke policy is explicitly a binary-action Qwen2 model.
        mask = torch.full_like(logits, float("-inf"))
        mask[:, 3:5] = 0
        logits = logits + mask
    labels = ids[0, len(prompt):]
    return logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)


class Rollouts:
    """Synchronous inference: live role object or explicit frozen block snapshot.

    No worker copy, remote server or weight synchronization lag exists. Every call
    records its purpose. GRPO stores raw-policy log probabilities at sampling time
    (standard temperature-sampled GRPO convention), not opponent snapshot logits.
    """
    def __init__(self, tokenizer, config):
        self.tok, self.cfg = tokenizer, config
        self.counts = {}

    @torch.no_grad()
    def generate_many(self, model, prompts, n, limit, purpose, temperature=None, old=False):
        """n independent samples per left-padded prompt on the supplied live model.

        Optional old log probabilities are computed on unpadded trajectories with
        the same model, before any update. Batching changes RNG consumption.
        """
        if not prompts or n < 1 or any(not prompt for prompt in prompts):
            raise ValueError("Nonempty prompts and a positive sample count are required")
        if self.cfg.tiny or (len(prompts) == 1 and self.cfg.rollout_batch_size == 1):
            return [self.generate(model, prompt, n, limit, purpose, temperature=temperature, old=old)
                    for prompt in prompts]
        was_training = model.training
        model.eval()
        temp = self.cfg.temperature if temperature is None else temperature
        width = max(map(len, prompts))
        device = next(model.parameters()).device
        ids = torch.full((len(prompts), width), self.tok.pad_token_id,
                         dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, prompt in enumerate(prompts):
            ids[i, -len(prompt):] = torch.tensor(prompt, device=device)
            mask[i, -len(prompt):] = 1
        kwargs = {"max_new_tokens": limit, "do_sample": temp > 0,
                  "pad_token_id": self.tok.pad_token_id, "eos_token_id": self.tok.eos_token_id,
                  "use_cache": True}
        if temp > 0:
            kwargs.update(temperature=temp, top_k=self.cfg.top_k, top_p=self.cfg.top_p)
        # Greedy decoding needs only one unique result per prompt. Copies preserve
        # the public n-results contract without introducing beam search.
        returns = n if temp > 0 else 1
        try:
            with autocast(self.cfg):
                generated = model.generate(input_ids=ids, attention_mask=mask,
                                           num_return_sequences=returns, **kwargs)
            sequences = generated[:, width:].tolist()
            count = self.counts.setdefault(purpose, {"sequences": 0, "generated_tokens": 0,
                "prompt_tokens": 0, "allocated_tokens": 0})
            groups = []
            for i, prompt in enumerate(prompts):
                group = []
                for j in range(n):
                    tokens = sequences[i * returns + (j if returns > 1 else 0)]
                    if self.tok.eos_token_id in tokens:
                        tokens = tokens[:tokens.index(self.tok.eos_token_id)+1]
                    else:
                        tokens = tokens.copy()
                    old_logps = token_logps(model, prompt, tokens, self.cfg).detach().cpu() if old else None
                    group.append(Sample(prompt.copy(), tokens,
                                        self.tok.decode(tokens, skip_special_tokens=True), old_logps))
                    count["sequences"] += 1
                    count["generated_tokens"] += len(tokens)
                    count["prompt_tokens"] += len(prompt)
                    count["allocated_tokens"] += limit
                groups.append(group)
            return groups
        finally:
            model.train(was_training)

    @torch.no_grad()
    def generate(self, model, prompt, n, limit, purpose, old=False, temperature=None):
        if not self.cfg.tiny and self.cfg.rollout_batch_size > 1 and n > 1:
            outputs = []
            for start in range(0, n, self.cfg.rollout_batch_size):
                size = min(self.cfg.rollout_batch_size, n - start)
                outputs.extend(self.generate_many(model, [prompt], size, limit, purpose,
                                                  temperature=temperature, old=old)[0])
            return outputs
        was_training = model.training
        model.eval()
        count = self.counts.setdefault(purpose, {"sequences": 0, "generated_tokens": 0,
            "prompt_tokens": 0, "allocated_tokens": 0})
        outputs = []
        temp = self.cfg.temperature if temperature is None else temperature
        try:
            for _ in range(n):
                device = next(model.parameters()).device
                ids = torch.tensor([prompt], device=device)
                if self.cfg.tiny:
                    # Actual Qwen2 forward, stochastic action, no scripted output/reward.
                    with autocast(self.cfg):
                        logits = model(input_ids=ids, use_cache=False).logits[0, -1, 3:5].float()
                    token = int(torch.multinomial((logits / temp).softmax(-1), 1)) if temp else int(logits.argmax())
                    tokens = [token + 3]
                else:
                    kw = {"max_new_tokens": limit, "do_sample": temp > 0,
                          "pad_token_id": self.tok.pad_token_id, "eos_token_id": self.tok.eos_token_id,
                          "use_cache": True}
                    if temp > 0:
                        kw.update(temperature=temp, top_k=self.cfg.top_k, top_p=self.cfg.top_p)
                    with autocast(self.cfg):
                        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
                    tokens = result[0, len(prompt):].tolist()
                old_logps = token_logps(model, prompt, tokens, self.cfg).detach().cpu() if old else None
                outputs.append(Sample(prompt.copy(), tokens, self.tok.decode(tokens, skip_special_tokens=True), old_logps))
                count["sequences"] += 1
                count["generated_tokens"] += len(tokens)
                count["prompt_tokens"] += len(prompt)
                count["allocated_tokens"] += limit
        finally:
            model.train(was_training)
        return outputs
