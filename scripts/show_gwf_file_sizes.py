from pathlib import Path


VERSION = "v100"
FOLDER_NAME = "training_set0"
POP_INDEX = 0
SAMPLE_INDEX = 0


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    base_dir = scripts_dir / "training_set" / VERSION / FOLDER_NAME

    for detector in ["H1", "L1"]:
        file_name = f"{detector}-pop{POP_INDEX:05d}_sample{SAMPLE_INDEX:05d}.gwf"
        file_path = base_dir / detector / file_name

        if not file_path.exists():
            print(f"{detector}: file not found -> {file_path}")
            continue

        size_bytes = file_path.stat().st_size
        size_mb = bytes_to_mb(size_bytes)
        print(f"{detector}: {file_path}")
        print(f"  size_bytes = {size_bytes}")
        print(f"  size_mb    = {size_mb:.3f} MB")


if __name__ == "__main__":
    main()
