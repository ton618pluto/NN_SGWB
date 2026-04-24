from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
TARGET_DIRS = [
    SCRIPTS_DIR / "processed_data_superimposed" / "v0" / "H1_splits",
    SCRIPTS_DIR / "processed_data_superimposed" / "v0" / "L1_splits",
]


def find_empty_npy_files(target_dir: Path) -> list[Path]:
    if not target_dir.exists():
        print(f"目录不存在: {target_dir}")
        return []

    return sorted(
        path for path in target_dir.iterdir()
        if path.suffix == ".npy" and path.is_file() and path.stat().st_size == 0
    )


def main() -> None:
    total_empty = 0

    for target_dir in TARGET_DIRS:
        empty_files = find_empty_npy_files(target_dir)
        print(f"\n检查目录: {target_dir}")

        if not empty_files:
            print("  未发现大小为 0 的 .npy 文件。")
            continue

        print(f"  发现 {len(empty_files)} 个大小为 0 的 .npy 文件:")
        for path in empty_files:
            print(f"  - {path}")
        total_empty += len(empty_files)

    print(f"\n检查完成。大小为 0 的 .npy 文件总数: {total_empty}")


if __name__ == "__main__":
    main()
