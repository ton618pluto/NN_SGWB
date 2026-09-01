from pathlib import Path
import csv
import random
import re

import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries


#POP_NUMS = 100000 # this will be changed
POP_NUMS = 300000 # this will be changed
FRAME_NUMS = 1
#version = "v61"  # this will be changed
version = "v100"  # this will be changed
LABEL_VERSION = version
#OUTPUT_VERSION = "v61_random_two_noise_200" # this will be changed
#OUTPUT_VERSION = "v61_random_two_noise_100" # this will be changed
OUTPUT_VERSION = "v100_random_two_noise_100" # this will be changed

DETECTORS = ["L1"]

STRAIN_CHANNEL_SUFFIX = "TEST_INJ"
#NOISE_SCALE = 5.0  # this will be changed
#NOISE_SCALE = 10.0  # this will be changed
NOISE_SCALE = 1/100.0  # this will be changed

NOISES_PER_SIGNAL = 2 # this will be changed
RANDOM_SEED = 42
NUM_SPLITS = 1
PROGRESS_EVERY_POPS = 1000

POP_PATTERN = re.compile(r"(pop\d+)")

SCRIPTS_DIR = Path(__file__).resolve().parent
SIGNAL_ROOT = SCRIPTS_DIR / "training_set" / version / "training_set0"
LABEL_ROOT = SCRIPTS_DIR / "training_set" / LABEL_VERSION / "training_set0"
NOISE_ROOT_TEMPLATE = "./noise_waveform/{version}/noise_waveform_{detector}"
OUTPUT_ROOT = SCRIPTS_DIR / "processed_data_superimposed" / OUTPUT_VERSION


def get_labels_by_pop(file_path: Path) -> dict[str, list]:
    df = pd.read_csv(file_path)
    labels_by_pop: dict[str, list] = {}

    for row in df.itertuples(index=False):
        sample_id = str(row[0])
        match = POP_PATTERN.search(sample_id)
        if match is None:
            continue

        pop_id = match.group(1)
        if pop_id not in labels_by_pop:
            labels_by_pop[pop_id] = list(row[2:])

    return labels_by_pop


def save_superimposed_as_npy(
    detector_name: str,
    labels_by_pop: dict[str, list],
    pop_nums: int = POP_NUMS,
    frame_nums: int = FRAME_NUMS,
    noise_scale: float = NOISE_SCALE,
    noises_per_signal: int = NOISES_PER_SIGNAL,
    random_seed: int = RANDOM_SEED,
    num_splits: int = NUM_SPLITS,
) -> None:
    if pop_nums < noises_per_signal:
        raise ValueError("POP_NUMS must be >= NOISES_PER_SIGNAL.")

    signal_dir = SIGNAL_ROOT / detector_name
    noise_dir = SCRIPTS_DIR / NOISE_ROOT_TEMPLATE.format(version=version, detector=detector_name)
    output_dir = OUTPUT_ROOT / f"{detector_name}_splits"
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
                "split_index",
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
    skipped_label = 0
    failed_count = 0

    print(f"[{detector_name}] Start direct superposition to npy, total {total_tasks} files.")

    for pop_num in range(pop_nums):
        pop_id = f"pop{pop_num:05d}"
        if pop_id not in labels_by_pop:
            skipped_label += frame_nums
            print(f"  - Skip {pop_id}: label not found.")
            continue

        label = labels_by_pop[pop_id]

        for frame_num in range(frame_nums):
            signal_path = signal_dir / f"{detector_name}-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            noise_numbers = rng.sample(range(1, pop_nums + 1), noises_per_signal)
            noise_paths = [
                noise_dir / f"{detector_name}-STRAIN-{noise_number:05d}-duration.gwf"
                for noise_number in noise_numbers
            ]
            output_groups = []
            for noise_slot in range(noises_per_signal):
                expanded_sample_num = frame_num * noises_per_signal + noise_slot
                base_name = (
                    f"{detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{expanded_sample_num:05d}"
                )
                split_paths = [
                    output_dir / f"{base_name}_p{split_index}.npy"
                    for split_index in range(num_splits)
                ]
                output_groups.append((noise_slot, expanded_sample_num, split_paths))

            # Avoid reading large GWF files when every expected output already exists.
            if all(all(path.exists() for path in split_paths) for _, _, split_paths in output_groups):
                skipped_existing += noises_per_signal
                continue

            if not signal_path.exists():
                skipped_missing += noises_per_signal
                print(f"  - Skip {pop_id}_sample{frame_num:05d}: signal file not found.")
                continue

            missing_noise = [path.name for path in noise_paths if not path.exists()]
            if missing_noise:
                skipped_missing += noises_per_signal
                print(
                    f"  - Skip {pop_id}_sample{frame_num:05d}: "
                    f"missing noise file(s): {', '.join(missing_noise)}."
                )
                continue

            try:
                signal_ts = TimeSeries.read(signal_path, strain_channel)
                noise_series = [TimeSeries.read(noise_path, strain_channel) for noise_path in noise_paths]

                if any(len(signal_ts) != len(noise_ts) or signal_ts.dt != noise_ts.dt for noise_ts in noise_series):
                    skipped_mismatch += noises_per_signal
                    print(
                        f"  - Skip {pop_id}_sample{frame_num:05d}: "
                        f"signal/noise length or sample spacing mismatch."
                    )
                    continue

                signal_data = np.asarray(signal_ts.value, dtype=np.float32)
                total_length = len(signal_data)
                split_length = total_length // num_splits

                for noise_slot, expanded_sample_num, split_paths in output_groups:
                    noise_ts = noise_series[noise_slot]

                    if all(path.exists() for path in split_paths):
                        skipped_existing += 1
                        continue

                    noise_data = np.asarray(noise_ts.value, dtype=np.float32)
                    superimposed_data = signal_data + noise_scale * noise_data

                    for split_index, split_path in enumerate(split_paths):
                        start = split_index * split_length
                        end = (split_index + 1) * split_length
                        chunk = superimposed_data[start:end]
                        sample = {
                            "data": chunk,
                            "label": label,
                        }
                        np.save(split_path, sample)

                        mapping_writer.writerow(
                            [
                                detector_name,
                                pop_num,
                                frame_num,
                                expanded_sample_num,
                                split_index,
                                split_path.name,
                                noise_slot,
                                noise_numbers[noise_slot],
                                noise_paths[noise_slot].name,
                            ]
                        )

                    mapping_handle.flush()
                    success_count += 1

            except Exception as exc:
                failed_count += 1
                print(f"  - Failed {pop_id}_sample{frame_num:05d}: {exc}")

        completed_pops = pop_num + 1
        if completed_pops % PROGRESS_EVERY_POPS == 0 or completed_pops == pop_nums:
            print(f"[{detector_name}] Completed {completed_pops}/{pop_nums} pops.")

    print(
        f"[{detector_name}] Done. Success {success_count}, "
        f"skip existing {skipped_existing}, "
        f"skip missing {skipped_missing}, "
        f"skip mismatch {skipped_mismatch}, "
        f"skip label {skipped_label}, "
        f"failed {failed_count}."
    )
    mapping_handle.close()


def main() -> None:
    label_path = LABEL_ROOT / "train_idx.csv"
    labels_by_pop = get_labels_by_pop(label_path)

    for detector_name in DETECTORS:
        save_superimposed_as_npy(detector_name, labels_by_pop)


if __name__ == "__main__":
    main()
