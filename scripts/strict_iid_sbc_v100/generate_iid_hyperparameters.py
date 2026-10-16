from __future__ import annotations

import json

import numpy as np

from config import (
    FIXED_PARAMETERS,
    HYPERPARAMETER_PATH,
    HYPERPARAMETER_SEED,
    INFERRED_PARAMETER_RANGES,
    NUM_SBC_SAMPLES,
    OUTPUT_ROOT,
    validate_config,
)


def main() -> None:
    validate_config()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(HYPERPARAMETER_SEED)

    samples: dict[str, np.ndarray] = {}
    for name, (lower, upper) in INFERRED_PARAMETER_RANGES.items():
        samples[name] = rng.uniform(lower, upper, size=NUM_SBC_SAMPLES).astype(
            np.float64
        )
    for name, value in FIXED_PARAMETERS.items():
        samples[name] = np.full(NUM_SBC_SAMPLES, value, dtype=np.float64)

    np.savez_compressed(HYPERPARAMETER_PATH, **samples)
    metadata = {
        "sampling": "independent uniform draws; not Latin hypercube sampling",
        "num_samples": NUM_SBC_SAMPLES,
        "seed": HYPERPARAMETER_SEED,
        "inferred_parameter_ranges": INFERRED_PARAMETER_RANGES,
        "fixed_parameters": FIXED_PARAMETERS,
        "output": str(HYPERPARAMETER_PATH),
    }
    metadata_path = HYPERPARAMETER_PATH.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Generated strict-IID hyperparameters")
    print(f"Samples: {NUM_SBC_SAMPLES}")
    print(f"Seed: {HYPERPARAMETER_SEED}")
    print(f"Saved parameters: {HYPERPARAMETER_PATH}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
