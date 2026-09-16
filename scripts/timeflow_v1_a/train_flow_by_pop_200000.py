from __future__ import annotations

import re

import torch

import train_flow_by_pop as base


# One signal population has two independently noised records. Select complete
# populations so the two records always stay together through the by-pop split.
MAX_TOTAL_SAMPLES = 200_000
EXPECTED_SAMPLES_PER_POP = 2
POP_PATTERN = re.compile(r"(pop\d+)")

_ORIGINAL_BUILD_RECORDS = base.GWDatasetV1A.build_records
_ORIGINAL_BUILD_PATHS = base.build_paths


def extract_pop_id(record: dict) -> str:
    file_name = record["h1"].name
    match = POP_PATTERN.search(file_name)
    if match is None:
        raise ValueError(f"Unable to parse pop id from file name: {file_name}")
    return match.group(1)


def select_complete_populations(records: list[dict], seed: int) -> list[dict]:
    if MAX_TOTAL_SAMPLES <= 0:
        raise ValueError("MAX_TOTAL_SAMPLES must be positive.")
    if EXPECTED_SAMPLES_PER_POP <= 0:
        raise ValueError("EXPECTED_SAMPLES_PER_POP must be positive.")
    if MAX_TOTAL_SAMPLES % EXPECTED_SAMPLES_PER_POP != 0:
        raise ValueError(
            f"MAX_TOTAL_SAMPLES={MAX_TOTAL_SAMPLES} must be divisible by "
            f"EXPECTED_SAMPLES_PER_POP={EXPECTED_SAMPLES_PER_POP}."
        )

    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(extract_pop_id(record), []).append(record)

    invalid_groups = {
        pop_id: len(pop_records)
        for pop_id, pop_records in grouped.items()
        if len(pop_records) != EXPECTED_SAMPLES_PER_POP
    }
    if invalid_groups:
        preview = ", ".join(
            f"{pop_id}:{count}" for pop_id, count in list(invalid_groups.items())[:10]
        )
        raise ValueError(
            f"Found {len(invalid_groups)} populations that do not contain exactly "
            f"{EXPECTED_SAMPLES_PER_POP} samples. First entries: {preview}"
        )

    requested_populations = MAX_TOTAL_SAMPLES // EXPECTED_SAMPLES_PER_POP
    pop_ids = sorted(grouped)
    if requested_populations > len(pop_ids):
        raise ValueError(
            f"Requested {requested_populations} complete populations "
            f"({MAX_TOTAL_SAMPLES} samples), but only found {len(pop_ids)} populations."
        )

    generator = torch.Generator().manual_seed(seed)
    selected_indices = torch.randperm(len(pop_ids), generator=generator)[
        :requested_populations
    ].tolist()
    selected_pop_ids = {pop_ids[index] for index in selected_indices}
    selected_records = [
        record for record in records if extract_pop_id(record) in selected_pop_ids
    ]
    if len(selected_records) != MAX_TOTAL_SAMPLES:
        raise RuntimeError(
            f"Population-preserving selection produced {len(selected_records)} samples; "
            f"expected {MAX_TOTAL_SAMPLES}."
        )

    print(
        f"Selected {requested_populations} complete populations, "
        f"{EXPECTED_SAMPLES_PER_POP} samples each, "
        f"{len(selected_records)} samples total."
    )
    return selected_records


def build_limited_records(*args, **kwargs) -> list[dict]:
    records = _ORIGINAL_BUILD_RECORDS(*args, **kwargs)
    return select_complete_populations(records, seed=base.SEED)


def build_limited_paths():
    h1_dir, l1_dir, original_output_dir = _ORIGINAL_BUILD_PATHS()
    limited_output_dir = original_output_dir.with_name(
        f"{original_output_dir.name}_samples{MAX_TOTAL_SAMPLES}_bypop"
    )
    return h1_dir, l1_dir, limited_output_dir


def main() -> None:
    if base.TRAIN_SAMPLE_STEP != 1:
        raise ValueError(
            "train_flow_by_pop_200000.py requires TRAIN_SAMPLE_STEP=1 so that "
            "both noise realizations are available before population selection."
        )

    # The original script used a five-digit regex, but this dataset contains
    # population IDs above pop99999. Override it only in this independent run.
    base.POP_PATTERN = POP_PATTERN
    base.GWDatasetV1A.build_records = staticmethod(build_limited_records)
    base.build_paths = build_limited_paths
    base.main()


if __name__ == "__main__":
    main()
