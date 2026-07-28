from pathlib import Path


VERSION = "v61"
DETECTORS = ["H1", "L1"]


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    base_dir = scripts_dir / "training_set_superimposed" / VERSION

    print(f"Superimposed base dir: {base_dir}")
    print()

    total_count = 0
    for detector in DETECTORS:
        detector_dir = base_dir / detector
        if not detector_dir.exists():
            print(f"{detector} directory missing: {detector_dir}")
            print(f"{detector} superimposed file count: 0")
            print()
            continue

        files = sorted(detector_dir.glob(f"{detector}-SUPERIMPOSED-*.gwf"))
        count = len(files)
        total_count += count

        print(f"{detector} directory: {detector_dir}")
        print(f"{detector} superimposed file count: {count}")
        print()

    print(f"Total superimposed file count: {total_count}")


if __name__ == "__main__":
    main()
