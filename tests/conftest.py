from pathlib import Path
import pytest
import torch

from gasp.config import Config
from gasp.data import make_smoke_data
from gasp.trainer import Trainer


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


@pytest.fixture
def config():
    return Config.load(Path(__file__).parents[1]/"configs/smoke.json")


@pytest.fixture
def factory(tmp_path, config):
    make_smoke_data(tmp_path/"data",config)
    counter = [0]
    def create(condition="unified"):
        counter[0] += 1
        return Trainer(config.with_changes(condition=condition),tmp_path/"data/train.jsonl",
                       tmp_path/"data/eval.jsonl",tmp_path/f"run-{counter[0]}")
    return create
