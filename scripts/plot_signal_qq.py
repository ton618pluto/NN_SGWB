from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy import stats


#VERSION = "v71"
VERSION = "v100"
first_folder = "training_set"
#first_folder = "training_set_superimposed"
FOLDER_NAME = "training_set0"
DETECTOR = "L1"
POP_INDEX = 0
SAMPLE_INDEX = 0




def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    base_dir = scripts_dir / first_folder / VERSION / FOLDER_NAME / DETECTOR
#    base_dir = scripts_dir / first_folder / VERSION  / DETECTOR
    
    file_name = f"{DETECTOR}-pop{POP_INDEX:05d}_sample{SAMPLE_INDEX:05d}.gwf"
    input_path = base_dir / file_name

    if not input_path.exists():
        raise FileNotFoundError(f"GWF file not found: {input_path}")

    ts = TimeSeries.read(str(input_path), channel=f"{DETECTOR}:TEST_INJ")
    data = np.asarray(ts.value, dtype=np.float64)

    mean = data.mean()
    std = data.std()
    if std <= 0:
        raise ValueError(f"Signal std is non-positive for file: {input_path}")

    data_norm = (data - mean) / std

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(data_norm, dist="norm", plot=ax)
    ax.set_title(
        f"Q-Q Plot vs Gaussian\n{DETECTOR} pop{POP_INDEX:05d} sample{SAMPLE_INDEX:05d}"
    )
    ax.set_xlabel("Theoretical Quantiles (Gaussian)")
    ax.set_ylabel("Sample Quantiles (Normalized Signal)")
    ax.grid(alpha=0.3)

    output_dir = scripts_dir / "qq_fig"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"qq_{VERSION}_{DETECTOR}_pop{POP_INDEX:05d}_sample{SAMPLE_INDEX:05d}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Input GWF: {input_path}")
    print(f"Detector channel: {DETECTOR}:TEST_INJ")
    print(f"Signal length: {len(data)}")
    print(f"Signal mean: {mean:.6e}")
    print(f"Signal std: {std:.6e}")
    print(f"Saved Q-Q figure: {output_path}")


if __name__ == "__main__":
    main()
