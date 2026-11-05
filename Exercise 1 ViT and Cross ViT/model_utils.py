from torch import save

from pathlib import Path


def save_model_ckpt(model_ckpt: dict):
    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"ckpt-{model_ckpt['model']}_CIFAR10_ep={model_ckpt['epoch']}_acc={model_ckpt['accuracy']}.pt"
    save(model_ckpt, out_dir / fname)
    print(f"Saved model checkpoint to: {out_dir/fname}")

# TODO: load_model_ckpt()

def load_model_ckpt(file: Path) -> dict :
    pass

# TODO: evaluation utils (inference and/or visualization)