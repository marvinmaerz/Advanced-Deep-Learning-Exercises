from torch import save
from pathlib import Path


def save_model_ckpt(model_ckpt: dict, name_suffix: str):
    model_name = model_ckpt["model"]
    epoch = model_ckpt["epoch"]
    accuracy = model_ckpt["accuracy"]

    out_dir = Path(f"checkpoints/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model_name}_ep={epoch}_acc={accuracy:.2f}_{name_suffix}.ckpt.pt"
    save(model_ckpt, out_dir / fname)
    print(f"Saved model checkpoint to: {out_dir/fname}")

# TODO: load_model_ckpt()

def load_model_ckpt(file: Path) -> dict :
    pass

# TODO: evaluation utils (inference and/or visualization)