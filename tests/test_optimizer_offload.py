import copy

import pytest
import torch

from gasp.data import make_smoke_data
from gasp.models import model_digest, rng_state
from gasp.trainer import Trainer
from test_training import assert_tree_equal


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


@pytest.mark.parametrize("condition", ["unified", "split"])
@pytest.mark.parametrize("device,precision", [("cpu", "fp32"), pytest.param("cuda", "bf16", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA transfer equivalence requires a GPU"))])
def test_offload_matches_resident_adam_and_resumes(config, tmp_path, condition, device, precision):
    cfg = config.with_changes(condition=condition, device=device, precision=precision)
    make_smoke_data(tmp_path / "data", cfg)

    def create(name, offload):
        return Trainer(cfg.with_changes(optimizer_offload=offload),
            tmp_path / "data/train.jsonl", tmp_path / "data/eval.jsonl", tmp_path / name)

    resident = create("resident", False)
    for _ in range(6):
        resident.step()
    expected_roles = cpu_tree(resident.roles.state_dict())
    expected_optimizer = cpu_tree(resident.optimizer.state_dict())
    expected_rng = rng_state()
    offloaded = create("offloaded", True)
    # Cross into the polluter block so both independent moment sets exist.
    for _ in range(3):
        offloaded.step()
    active = set(offloaded.roles.active("polluter").parameters())
    for parameter, state in offloaded.optimizer.state.items():
        expected_device = device if parameter in active else "cpu"
        assert state["exp_avg"].device.type == expected_device
        assert state["exp_avg_sq"].device.type == expected_device
        assert state["step"].device.type == "cpu"
    # The next update is in the same block: it must reuse the resident moments.
    moment_storage = {p: s["exp_avg"].data_ptr() for p, s in offloaded.optimizer.state.items()}
    offloaded.place_optimizer_moments()
    assert moment_storage == {p: s["exp_avg"].data_ptr() for p, s in offloaded.optimizer.state.items()}
    path = offloaded.save()
    resumed = create("resumed", True)
    identities = {id(p) for group in resumed.optimizer.param_groups for p in group["params"]}
    resumed.resume(path)
    assert identities == {id(p) for role in ("agent", "polluter")
                          for p in resumed.roles.active(role).parameters() if p.requires_grad}
    assert model_digest(resumed.roles.agent) == model_digest(offloaded.roles.agent)
    assert all(state["exp_avg"].device.type == "cpu" for state in resumed.optimizer.state.values())
    for _ in range(3):
        resumed.step()
    assert_tree_equal(expected_roles, cpu_tree(resumed.roles.state_dict()))
    assert_tree_equal(expected_optimizer, cpu_tree(resumed.optimizer.state_dict()))
    assert_tree_equal(expected_rng, rng_state())
