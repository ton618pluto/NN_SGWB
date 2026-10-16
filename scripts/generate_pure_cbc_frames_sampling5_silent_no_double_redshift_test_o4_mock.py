"""Generate the test CBC frames with an explicit O4_mock ASD.

This entry point reuses the tested frame-generation implementation while
forcing Bilby's H1 and L1 detector PSDs to be constructed from O4_mock.txt.
The original test generator remains unchanged when run directly because its
ASD_FILE_PATH setting defaults to None.
"""

from pathlib import Path

import generate_pure_cbc_frames_sampling5_silent_no_double_redshift_test as generator


SCRIPT_DIR = Path(__file__).resolve().parent

VERSION = "fixed_one_test_v100"
NUM_POPULATIONS = 1
ALL_JOBS = 2  # frame count + 1
MAX_WORKERS = 8

O4_ASD_PATH = SCRIPT_DIR / "O4_mock.txt"
INPUT_DIR = SCRIPT_DIR / "parameter_sampling_test" / VERSION
OUTPUT_DIR = (
    SCRIPT_DIR
    / "training_set_test"
    / f"{VERSION}_o4_mock"
    / "training_set0"
)


def main() -> None:
    if not O4_ASD_PATH.exists():
        raise FileNotFoundError(f"O4 ASD file not found: {O4_ASD_PATH}")

    generator.version = VERSION
    generator.num_populations = NUM_POPULATIONS
    generator.all_jobs = ALL_JOBS
    generator.input_dir = str(INPUT_DIR)
    generator.output_base_dir = str(OUTPUT_DIR)
    generator.ASD_FILE_PATH = str(O4_ASD_PATH)

    print(f"Explicit O4 ASD: {O4_ASD_PATH}")
    print(f"Input parameter directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    generator.generate_samples(max_workers=MAX_WORKERS)


if __name__ == "__main__":
    main()
