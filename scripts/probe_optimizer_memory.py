"""Three full-size synthetic updates: memory check, not experiment training."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from gasp.config import Config
from gasp.models import load_model, Roles, token_logps, model_digest
from gasp.trainer import Trainer, storage_bytes


cfg = Config.load("data/full-v4/resolved_config.json")
model, tok = load_model(cfg)
roles = Roles(model, "split")
optimizer = torch.optim.AdamW([*roles.agent.parameters(), *roles.polluter.parameters()],
    lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_epsilon,
    weight_decay=cfg.weight_decay, foreach=False)
probe = SimpleNamespace(cfg=cfg, roles=roles, optimizer=optimizer, role="agent")
report = {"success": False, "synthetic_memory_probe": True,
          "gpu": torch.cuda.get_device_name(), "updates": []}
try:
    for role in ("agent", "polluter", "agent"):
        probe.role = role
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        Trainer.place_optimizer_moments(probe)
        torch.cuda.synchronize()
        transfer_seconds = time.perf_counter()-started
        inactive = roles.active("polluter" if role == "agent" else "agent")
        before = model_digest(inactive)
        # Match the maximum sequence/logit dimensions used by token_logps.
        roles.active(role).train()
        # The second pass retains the first pass's gradients, exercising the
        # peak that also occurs inside GRPO's sequential gradient accumulation.
        for _ in range(2):
            logps = token_logps(roles.active(role), [100]*cfg.prompt_tokens,
                               [100]*cfg.completion_tokens, cfg)
            (-logps.mean()/2).backward()
            del logps
        norm = torch.nn.utils.clip_grad_norm_(roles.active(role).parameters(), cfg.grad_clip)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
        torch.cuda.synchronize()
        assert before == model_digest(inactive), "Inactive weights changed"
        report["updates"].append({"role": role, "seconds": time.perf_counter()-started,
            "accumulated_backward_passes": 2,
            "state_transfer_seconds": transfer_seconds, "grad_norm": float(norm),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_bytes": storage_bytes(optimizer.state_dict())})
        print(json.dumps(report["updates"][-1]), flush=True)
    report["success"] = True
finally:
    Path("artifacts/traces/memory-probe.json").write_text(json.dumps(report, indent=2)+"\n")
