from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    condition: str = "split"
    label: str = "gasp-full"
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    revision: str = "main"  # Preparation resolves this to an immutable commit.
    seed: int = 42
    device: str = "auto"
    precision: str = "bf16"
    tuning: str = "full"
    lora_rank: int = 16
    gradient_checkpointing: bool = True
    optimizer_offload: bool = False
    preparation_batch_questions: int = 1
    role_steps: int = 1000
    block_steps: int = 5
    accumulation: int = 4
    group_size: int = 64
    max_clean_attempts: int = 32  # Stop the run on exhaustion; never reuse old traces.
    prompt_tokens: int = 1024
    completion_tokens: int = 1024
    corruption_tokens: int = 1024
    repair_tokens: int = 1024
    window_cap: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    learning_rate: float = 3e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0
    grpo_clip: float = 0.2
    advantage_epsilon: float = 1e-8
    guidance: float = 0.07
    guidance_decay_fraction: float = 0.2
    eval_every: int = 100  # global optimizer updates, plus initial and final
    checkpoint_every: int = 10
    eval_seed: int = 10001
    eval_temperature: float = 0.0
    verifier: str = "math_verify"
    # Tiny smoke has real Qwen2 losses/generation, abbreviated prompts and vocabulary.
    tiny: bool = False

    def __post_init__(self):
        if self.condition not in ("unified", "split"):
            raise ValueError("condition must be unified or split")
        for key in ("role_steps", "block_steps", "accumulation", "group_size",
                    "max_clean_attempts", "prompt_tokens", "completion_tokens",
                    "corruption_tokens", "repair_tokens", "window_cap",
                    "eval_every", "checkpoint_every", "preparation_batch_questions"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.group_size < 2 or self.role_steps % self.block_steps:
            raise ValueError("group_size >= 2; role_steps must be a multiple of block_steps")
        if self.tuning not in ("full", "lora") or self.precision not in ("fp32", "bf16"):
            raise ValueError("unsupported tuning or precision")
        if not 0 < self.guidance_decay_fraction <= 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("invalid schedule fractions")
        if self.temperature <= 0 or self.top_p != 1.0:
            raise ValueError("training requires temperature > 0 and top_p=1 for audited sampler")

    @classmethod
    def load(cls, path, **overrides):
        values = json.loads(Path(path).read_text())
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def paired_hash(self):
        d = asdict(self)
        d.pop("condition")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    def with_changes(self, **kwargs):
        return replace(self, **kwargs)

    def guidance_weight(self, completed_agent_steps):
        # Last allocated agent update has weight 0; first has full lambda.
        start = int(self.role_steps * (1 - self.guidance_decay_fraction))
        if completed_agent_steps < start:
            return self.guidance
        return self.guidance * max(0., (self.role_steps - 1 - completed_agent_steps)
                                  / max(1, self.role_steps - 1 - start))
