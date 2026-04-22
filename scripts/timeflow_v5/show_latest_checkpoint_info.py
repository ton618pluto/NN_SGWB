from __future__ import annotations

from pathlib import Path

import torch


CURRENT_DIR = Path(__file__).resolve().parent
LATEST_CHECKPOINT_PATH = CURRENT_DIR / "outputs" / "latest_flow_v5.pt"


def shape_of(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return None


def main() -> None:
    if not LATEST_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {LATEST_CHECKPOINT_PATH}")

    checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    print(f"checkpoint={LATEST_CHECKPOINT_PATH}")
    print(f"keys={list(checkpoint.keys())}")
    print(f"epoch={checkpoint.get('epoch')}")
    print(f"best_val_loss={checkpoint.get('best_val_loss')}")
    print(f"seed={checkpoint.get('seed')}")
    print(f"val_fraction={checkpoint.get('val_fraction')}")
    print(f"test_fraction={checkpoint.get('test_fraction')}")
    print(f"num_channels={checkpoint.get('num_channels')}")
    print(f"param_dim={checkpoint.get('param_dim')}")

    label_mean = checkpoint.get("label_mean")
    label_std = checkpoint.get("label_std")
    print(f"label_mean_shape={shape_of(label_mean)}")
    print(f"label_std_shape={shape_of(label_std)}")

    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    scaler_state = checkpoint.get("scaler_state_dict")
    model_state = checkpoint.get("model_state_dict")

    print(f"optimizer_param_groups={len(optimizer_state.get('param_groups', [])) if optimizer_state else 0}")
    print(f"optimizer_state_entries={len(optimizer_state.get('state', {})) if optimizer_state else 0}")
    print(f"scheduler_keys={list(scheduler_state.keys()) if scheduler_state else []}")
    print(f"scaler_keys={list(scaler_state.keys()) if scaler_state else []}")
    print(f"model_state_param_count={len(model_state) if model_state else 0}")


if __name__ == "__main__":
    main()
