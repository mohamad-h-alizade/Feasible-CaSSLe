import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_repo_module(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def require_cassle_methods():
    try:
        from cassle.methods import METHODS
    except ModuleNotFoundError as exc:
        if exc.name == "pytorch_lightning":
            raise RuntimeError(
                "The original cassle model package requires pytorch_lightning. "
                "Install the repository dependencies from README.md before running training."
            ) from exc
        raise
    return METHODS


_simclr = load_repo_module("cassle/losses/simclr.py", "_feasible_cassle_simclr_loss")
_byol = load_repo_module("cassle/losses/byol.py", "_feasible_cassle_byol_loss")
_barlow = load_repo_module("cassle/losses/barlow.py", "_feasible_cassle_barlow_loss")

simclr_loss_func = _simclr.simclr_loss_func
simclr_distill_loss_func = _simclr.simclr_distill_loss_func
byol_loss_func = _byol.byol_loss_func
barlow_loss_func = _barlow.barlow_loss_func

