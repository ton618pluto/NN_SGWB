from pathlib import Path

from gwpy.timeseries import TimeSeries


POP_NUMS = 1
FRAME_NUMS = 10
#BASE_VERSION = "v71" # this will be changed
BASE_VERSION = "v61"
SIGNAL_VERSION = f"fixed_one_test_{BASE_VERSION}"
NOISE_VERSION = BASE_VERSION
DETECTORS = ["H1", "L1"]
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"

# Use one fixed noise sample number for all test frames.
NOISE_FILE_NUMBER = 1 # this will be changed,is id+1

SCRIPTS_DIR = Path(__file__).resolve().parent
SIGNAL_ROOT = SCRIPTS_DIR / "training_set_test" / SIGNAL_VERSION / "training_set0"
NOISE_ROOT_TEMPLATE = "./noise_waveform/{version}/noise_waveform_{detector}"
OUTPUT_ROOT = SCRIPTS_DIR / "training_set_superimposed_test" / SIGNAL_VERSION


def superimpose(
    detector_name: str,
    pop_nums: int = POP_NUMS,
    frame_nums: int = FRAME_NUMS,
    noise_file_number: int = NOISE_FILE_NUMBER,
) -> None:
    signal_dir = SIGNAL_ROOT / detector_name
    noise_dir = SCRIPTS_DIR / NOISE_ROOT_TEMPLATE.format(version=NOISE_VERSION, detector=detector_name)
    output_dir = OUTPUT_ROOT / detector_name
    strain_channel = f"{detector_name}:{STRAIN_CHANNEL_SUFFIX}"

    output_dir.mkdir(parents=True, exist_ok=True)

    total_tasks = pop_nums * frame_nums
    success_count = 0
    skipped_existing = 0
    skipped_missing = 0
    skipped_mismatch = 0
    failed_count = 0

    print(
        f"[{detector_name}] start superposition, total_tasks={total_tasks}, "
        f"noise_file_number={noise_file_number}"
    )

    for pop_num in range(pop_nums):
        for frame_num in range(frame_nums):
            signal_path = signal_dir / f"{detector_name}-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            noise_path = noise_dir / f"{detector_name}-STRAIN-{noise_file_number:05d}-duration.gwf"
            output_path = output_dir / f"{detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            label = f"pop{pop_num:05d}_sample{frame_num:05d}"

            if output_path.exists():
                skipped_existing += 1
                print(f"  - skip {label}: output already exists")
                continue

            if not signal_path.exists():
                skipped_missing += 1
                print(f"  - skip {label}: missing signal file")
                continue

            if not noise_path.exists():
                skipped_missing += 1
                print(f"  - skip {label}: missing noise file {noise_path.name}")
                continue

            try:
                signal_ts = TimeSeries.read(signal_path, strain_channel)
                noise_ts = TimeSeries.read(noise_path, strain_channel)

                if len(signal_ts) != len(noise_ts) or signal_ts.dt != noise_ts.dt:
                    skipped_mismatch += 1
                    print(f"  - skip {label}: signal/noise length or dt mismatch")
                    continue

                superimposed_ts = signal_ts + noise_ts
                superimposed_ts.write(output_path)
                success_count += 1

            except Exception as exc:
                failed_count += 1
                print(f"  - failed {label}: {exc}")

        completed_pops = pop_num + 1
        if completed_pops % 10 == 0 or completed_pops == pop_nums:
            print(f"[{detector_name}] completed {completed_pops}/{pop_nums} pops")

    print(
        f"[{detector_name}] done. success={success_count}, "
        f"skipped_existing={skipped_existing}, skipped_missing={skipped_missing}, "
        f"skipped_mismatch={skipped_mismatch}, failed={failed_count}"
    )


def main() -> None:
    for detector_name in DETECTORS:
        superimpose(detector_name)


if __name__ == "__main__":
    main()
