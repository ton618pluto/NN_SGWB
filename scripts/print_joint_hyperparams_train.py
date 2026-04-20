from pathlib import Path

import numpy as np


def main() -> None:
    npz_path = Path(__file__).parent / "joint_hyperparams_train" / "v3" / "joint_hyperparams_train_v3.npz"

    with np.load(npz_path, allow_pickle=True) as data:
        print(f"File: {npz_path}")
        print(f"Keys: {data.files}\n")

        with np.printoptions(precision=6, suppress=True, threshold=np.inf, linewidth=160):
            for key in data.files:
                values = data[key]
                print(f"{key} | shape={values.shape} | dtype={values.dtype}")
                print(values)
                print()


if __name__ == "__main__":
    main()
