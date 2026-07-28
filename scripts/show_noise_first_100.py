from pathlib import Path

from gwpy.timeseries import TimeSeries


VERSION = "v61"
DETECTOR = "H1"
JOB_NUMBER = 1
NUM_VALUES = 100


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    noise_dir = scripts_dir / "noise_waveform" / VERSION / f"noise_waveform_{DETECTOR}"
    gwf_path = noise_dir / f"{DETECTOR}-STRAIN-{JOB_NUMBER:05d}-duration.gwf"
    channel = f"{DETECTOR}:TEST_INJ"

    if not gwf_path.exists():
        raise FileNotFoundError(f"Noise file not found: {gwf_path}")

    data = TimeSeries.read(gwf_path, channel=channel).value
    head = data[:NUM_VALUES]

    print(f"Noise file: {gwf_path}")
    print(f"Channel: {channel}")
    print(f"Total length: {len(data)}")
    print(f"First {len(head)} values:")
    print(head)


if __name__ == "__main__":
    main()
