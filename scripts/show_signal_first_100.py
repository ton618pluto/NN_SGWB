from pathlib import Path

from gwpy.timeseries import TimeSeries


VERSION = "v61"
FOLDER_NAME = "training_set0"
DETECTOR = "H1"
POP_INDEX = 0
SAMPLE_INDEX = 0
NUM_VALUES = 100


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    signal_dir = scripts_dir / "training_set" / VERSION / FOLDER_NAME / DETECTOR
    sample_id = f"pop{POP_INDEX:05d}_sample{SAMPLE_INDEX:05d}"
    gwf_path = signal_dir / f"{DETECTOR}-{sample_id}.gwf"
    channel = f"{DETECTOR}:TEST_INJ"

    if not gwf_path.exists():
        raise FileNotFoundError(f"Signal file not found: {gwf_path}")

    data = TimeSeries.read(gwf_path, channel=channel).value
    head = data[:NUM_VALUES]

    print(f"Signal file: {gwf_path}")
    print(f"Channel: {channel}")
    print(f"Total length: {len(data)}")
    print(f"First {len(head)} values:")
    print(head)


if __name__ == "__main__":
    main()
