from pathlib import Path
import random

from gwpy.timeseries import TimeSeries
import csv

# 如果要改就改这里
POP_NUMS = 100000
FRAME_NUMS = 1
version = "v61"
OUTPUT_VERSION = "v61_random_two_noise_500"
DETECTORS = ["L1"]
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"
OUTPUT_PREFIX = "SUPERIMPOSED"
#NOISE_SCALE = 1.0  # this will be changed
NOISE_SCALE = 2.0  # this will be changed
NOISES_PER_SIGNAL = 2  # this will be changed
RANDOM_SEED = 42

SCRIPTS_DIR = Path(__file__).resolve().parent
SIGNAL_ROOT = SCRIPTS_DIR / "training_set" / version / "training_set0"
NOISE_ROOT_TEMPLATE = "./noise_waveform/{version}/noise_waveform_{detector}"
OUTPUT_ROOT = SCRIPTS_DIR / "training_set_superimposed" / OUTPUT_VERSION


def superimpose(
    detector_name: str,
    pop_nums: int = POP_NUMS,
    frame_nums: int = FRAME_NUMS,
    noise_scale: float = NOISE_SCALE,
    noises_per_signal: int = NOISES_PER_SIGNAL,
    random_seed: int = RANDOM_SEED,
) -> None:
    if pop_nums < noises_per_signal:
        raise ValueError("POP_NUMS must be >= NOISES_PER_SIGNAL.")

    signal_dir = SIGNAL_ROOT / detector_name
    noise_dir = SCRIPTS_DIR / NOISE_ROOT_TEMPLATE.format(version=version, detector=detector_name)
    output_dir = OUTPUT_ROOT / detector_name
    mapping_path = OUTPUT_ROOT / f"{detector_name}_noise_mapping.csv"
    strain_channel = f"{detector_name}:{STRAIN_CHANNEL_SUFFIX}"
    rng = random.Random(random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    mapping_exists = mapping_path.exists()
    mapping_handle = mapping_path.open("a", encoding="utf-8", newline="")
    mapping_writer = csv.writer(mapping_handle)
    if not mapping_exists:
        mapping_writer.writerow(
            [
                "detector",
                "pop_num",
                "source_frame_num",
                "expanded_sample_num",
                "output_file",
                "noise_slot",
                "noise_number",
                "noise_file",
            ]
        )

    total_tasks = pop_nums * frame_nums * noises_per_signal
    success_count = 0
    skipped_existing = 0
    skipped_missing = 0
    skipped_mismatch = 0
    failed_count = 0

    print(f"[{detector_name}] Start superposition, total {total_tasks} files.")

    for pop_num in range(pop_nums):
        for frame_num in range(frame_nums):
            signal_path = signal_dir / f"{detector_name}-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            noise_numbers = rng.sample(range(1, pop_nums + 1), noises_per_signal)
            noise_paths = [
                noise_dir / f"{detector_name}-STRAIN-{noise_number:05d}-duration.gwf"
                for noise_number in noise_numbers
            ]

            if not signal_path.exists():
                skipped_missing += 1
                print(f"  - Skip pop{pop_num:05d}_sample{frame_num:05d}: signal file not found.")
                continue

            missing_noise = [path.name for path in noise_paths if not path.exists()]
            if missing_noise:
                skipped_missing += 1
                print(
                    f"  - Skip pop{pop_num:05d}_sample{frame_num:05d}: "
                    f"missing noise file(s): {', '.join(missing_noise)}."
                )
                continue

            try:
                signal_ts = TimeSeries.read(signal_path, strain_channel)
                noise_series = [TimeSeries.read(noise_path, strain_channel) for noise_path in noise_paths]

                if any(len(signal_ts) != len(noise_ts) or signal_ts.dt != noise_ts.dt for noise_ts in noise_series):
                    skipped_mismatch += 1
                    print(
                        f"  - Skip pop{pop_num:05d}_sample{frame_num:05d}: "
                        f"signal/noise length or sample spacing mismatch."
                    )
                    continue

                for noise_slot, noise_ts in enumerate(noise_series):
                    expanded_sample_num = frame_num * noises_per_signal + noise_slot
                    output_path = output_dir / (
                        f"{detector_name}-{OUTPUT_PREFIX}-pop{pop_num:05d}_sample{expanded_sample_num:05d}.gwf"
                    )
                    label = (
                        f"pop{pop_num:05d}_sample{frame_num:05d}"
                        f" -> sample{expanded_sample_num:05d} with noise {noise_numbers[noise_slot]:05d}"
                    )

                    if output_path.exists():
                        skipped_existing += 1
                        print(f"  - Skip {label}: output already exists.")
                        continue

                    superimposed_ts = signal_ts + noise_scale * noise_ts
                    superimposed_ts.write(output_path)
                    mapping_writer.writerow(
                        [
                            detector_name,
                            pop_num,
                            frame_num,
                            expanded_sample_num,
                            output_path.name,
                            noise_slot,
                            noise_numbers[noise_slot],
                            noise_paths[noise_slot].name,
                        ]
                    )
                    mapping_handle.flush()
                    success_count += 1

            except Exception as exc:
                failed_count += 1
                print(f"  - Failed pop{pop_num:05d}_sample{frame_num:05d}: {exc}")

        completed_pops = pop_num + 1
        if completed_pops % 10 == 0 or completed_pops == pop_nums:
            print(f"[{detector_name}] Completed {completed_pops}/{pop_nums} pops.")

    print(
        f"[{detector_name}] Done. Success {success_count}, "
        f"skip existing {skipped_existing}, "
        f"skip missing {skipped_missing}, "
        f"skip mismatch {skipped_mismatch}, "
        f"failed {failed_count}."
    )
    mapping_handle.close()


def main() -> None:
    for detector_name in DETECTORS:
        superimpose(detector_name)


if __name__ == "__main__":
    main()
