from pathlib import Path

import numpy as np


VERSION = "v21"


def candidate_paths(version: str) -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    file_name = f"joint_hyperparams_train_{version}.npz"
    return [
        script_dir / "joint_hyperparams_train" / version / file_name,
        script_dir.parent / "joint_hyperparams_train" / version / file_name,
    ]


def resolve_npz_path(version: str) -> Path:
    for path in candidate_paths(version):
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidate_paths(version))
    raise FileNotFoundError(f"Could not find NPZ file for {version}. Searched:\n{searched}")


def summarize_array(name: str, values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return (
        f"{name:<15}"
        f"count={arr.size:<8d}"
        f"min={arr.min():>12.6f}"
        f"max={arr.max():>12.6f}"
        f"mean={arr.mean():>12.6f}"
        f"std={arr.std():>12.6f}"
    )


def main() -> None:
    npz_path = resolve_npz_path(VERSION)

    with np.load(npz_path, allow_pickle=True) as data:
        print(f"File: {npz_path}")
        print(f"Keys: {data.files}\n")
        print("Per-parameter range summary:")
        print("-" * 80)
        for key in data.files:
            print(summarize_array(key, data[key]))


if __name__ == "__main__":
    main()
