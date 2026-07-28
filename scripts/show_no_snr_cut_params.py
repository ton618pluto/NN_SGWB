from pathlib import Path

import numpy as np


VERSION = "v100"
FILE_INDEX = 0
NUM_EVENTS_TO_SHOW = 10


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    npz_path = scripts_dir / "parameter_sampling_train" / VERSION / f"CBC_params_example{FILE_INDEX:05d}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {npz_path}")

    data = np.load(npz_path)

    print(f"File: {npz_path}")
    print()

    print("True hyperparameters:")
    for key in [
        "true_R0",
        "true_alpha_z",
        "true_beta_z",
        "true_zp",
        "true_alpha_m",
        "true_m_max",
        "true_delta_m",
        "true_m_min",
        "true_lambda_peak",
        "true_mu_m",
        "true_sigma_m",
        "true_beta_q",
    ]:
        if key in data:
            print(f"  {key} = {float(data[key]):.10f}")
    print()

    num_events = int(data["num_events"]) if "num_events" in data else len(data["tc"])
    print(f"num_events = {num_events}")
    if "attempt_index" in data:
        print(f"attempt_index = {int(data['attempt_index'])}")
    if "attempts_used" in data:
        print(f"attempts_used = {int(data['attempts_used'])}")
    if "total_duration_seconds" in data:
        print(f"total_duration_seconds = {float(data['total_duration_seconds'])}")
    print()

    show_count = min(NUM_EVENTS_TO_SHOW, num_events)
    print(f"First {show_count} events:")
    for idx in range(show_count):
        print(
            f"  event[{idx}] "
            f"tc={float(data['tc'][idx]):.10f}, "
            f"m1={float(data['m1'][idx]):.10f}, "
            f"m2={float(data['m2'][idx]):.10f}, "
            f"dL={float(data['dL'][idx]):.10f}, "
            f"z={float(data['z'][idx]):.10f}, "
            f"ra={float(data['ra'][idx]):.10f}, "
            f"dec={float(data['dec'][idx]):.10f}, "
            f"psi={float(data['psi'][idx]):.10f}, "
            f"phi_c={float(data['phi_c'][idx]):.10f}, "
            f"iota={float(data['iota'][idx]):.10f}"
        )


if __name__ == "__main__":
    main()
