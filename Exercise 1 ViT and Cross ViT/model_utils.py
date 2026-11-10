import matplotlib.pyplot as plt
import numpy as np

from torch import save
from pathlib import Path


def save_model_ckpt(model_ckpt: dict, name_suffix: str = ""):
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

def plot_accuracy(accuracy_vals: list[float], accuracy_max: float, model_type: str, epochs: int):
    out_dir = Path(f"plots/{model_type}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fname = out_dir / f"{model_type}_accuracy_ep={epochs}_max={accuracy_max:.2f}.png"

    # value processing
    x = range(1, len(accuracy_vals) + 1)
    plt.plot(x, accuracy_vals)
    plt.ylim(0.0, 1.0)

    # labels, etc.
    plt.title(f"{model_type} accuracy for {epochs} epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(out_fname)
    # plt.show()
    print(f"Saved accuracy plot to {out_fname}")
