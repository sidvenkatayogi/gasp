"""Use Modal's supported detached lifecycle with the unchanged frozen runner.

Only the local App.run invocation changes. GPU worker code, data, deadlines and
budget accounting remain exactly those in source.zip.
"""
from pathlib import Path
import inspect
import sys

import modal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_modal_experiment import main

original_run = inspect.getattr_static(modal.App, "run")


def detached_run(self, *args, **kwargs):
    bound_run = original_run.__get__(self, type(self))
    return bound_run(*args, **(kwargs | {"detach": True}))


if __name__ == "__main__":
    modal.App.run = detached_run
    try:
        main()
    finally:
        modal.App.run = original_run
