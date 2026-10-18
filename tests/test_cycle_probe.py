import importlib.util
from pathlib import Path

import pytest

from gasp.models import model_digest


def probe():
    path = Path(__file__).parents[1] / "scripts/probe_cycle.py"
    spec = importlib.util.spec_from_file_location("cycle_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cycle_profile_keeps_full_schedule_and_paired_settings():
    module = probe()
    a, b = (module.cycle_config(c) for c in ("unified", "split"))
    assert a.paired_hash() == b.paired_hash()
    assert (a.role_steps, a.block_steps, a.group_size, a.accumulation) == (100, 5, 16, 1)
    assert a.prompt_tokens == 2048 and a.rollout_batch_size == 16
    assert a.completion_tokens == a.corruption_tokens == 1024
    assert a.optimizer_offload and a.eval_every == 200
    assert a.revision != "main"


def test_instrumentation_preserves_real_update_and_rng(factory, tmp_path):
    reference = factory("split")
    expected = reference.step()
    digest = model_digest(reference.roles.agent)
    measured = factory("split")
    timing = probe().Timings(tmp_path / "phases.jsonl", lambda: None)
    timing.wrap(measured.roles, "refresh", "snapshot")
    timing.wrap(measured, "place_optimizer_moments", "transfer")
    timing.wrap(measured, "backward_group", "backward")
    actual = timing.call("update", measured.step)
    assert model_digest(measured.roles.agent) == digest
    assert actual["contexts"] == expected["contexts"]
    assert actual["lr"] == expected["lr"]
    assert {e["phase"] for e in timing.events} == {"snapshot", "transfer", "backward", "update"}
    assert all(e["success"] and e["seconds"] >= 0 for e in timing.events)


def test_failed_phase_is_recorded_and_propagated(tmp_path):
    timing = probe().Timings(tmp_path / "phases.jsonl", lambda: None)
    def fail():
        raise ValueError("test failure")
    with pytest.raises(ValueError, match="test failure"):
        timing.call("failed", fail)
    assert timing.events[-1]["success"] is False
