from __future__ import annotations

from pathlib import Path

import torch


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best_flow_v5.pt"
LATEST_CHECKPOINT_PATH = OUTPUT_DIR / "latest_flow_v5.pt"


def choose_checkpoint_path() -> Path:
    if BEST_CHECKPOINT_PATH.exists():
        return BEST_CHECKPOINT_PATH
    if LATEST_CHECKPOINT_PATH.exists():
        return LATEST_CHECKPOINT_PATH
    raise FileNotFoundError(
        f"No checkpoint found in {OUTPUT_DIR}. "
        f"Expected {BEST_CHECKPOINT_PATH.name} or {LATEST_CHECKPOINT_PATH.name}."
    )


def main() -> None:
    checkpoint_path = choose_checkpoint_path()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    best_val_loss = checkpoint.get("best_val_loss")
    epoch = checkpoint.get("epoch")

    print(f"checkpoint={checkpoint_path}")
    print(f"epoch={epoch}")
    print(f"best_val_nll={best_val_loss}")


if __name__ == "__main__":
    main()
