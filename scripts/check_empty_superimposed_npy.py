from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
# folder = "processed_data_superimposed"
folder = "processed_data_train"
# version = "v10_2"
# version = "v6"
# version = "v11"
#version = "v21"
#version = "v22"
#version = "v23"
#version = "v30_ns"
#version = "v44"
#version = "v45"
#version = "v61"
#version = "v62"
version = "v63"
TARGET_DIRS = [
    SCRIPTS_DIR / folder / version / "H1_splits",
    SCRIPTS_DIR / folder / version / "L1_splits",
]


def find_empty_npy_files(target_dir: Path) -> list[Path]:
    if not target_dir.exists():
        print(f"目录不存在: {target_dir}")
        return []

    return sorted(
        path
        for path in target_dir.iterdir()
        if path.suffix == ".npy" and path.is_file() and path.stat().st_size == 0
    )


def count_npy_files(target_dir: Path) -> int:
    if not target_dir.exists():
        return 0
    return sum(1 for path in target_dir.iterdir() if path.suffix == ".npy" and path.is_file())


def sample_key(file_name: str) -> str:
    stem = Path(file_name).stem
    if "-" in stem:
        return stem.split("-", 1)[-1]
    if stem.startswith("H1_") or stem.startswith("L1_"):
        return stem[3:]
    if stem.startswith("H1") or stem.startswith("L1"):
        return stem[2:]
    return stem


def list_npy_name_map(target_dir: Path) -> dict[str, str]:
    if not target_dir.exists():
        return {}
    return {
        sample_key(path.name): path.name
        for path in target_dir.iterdir()
        if path.suffix == ".npy" and path.is_file()
    }


def main() -> None:
    total_empty = 0
    total_checked = 0

    h1_dir, l1_dir = TARGET_DIRS
    h1_count = count_npy_files(h1_dir)
    l1_count = count_npy_files(l1_dir)
    h1_name_map = list_npy_name_map(h1_dir)
    l1_name_map = list_npy_name_map(l1_dir)

    print(f"H1 .npy 文件总数: {h1_count}")
    print(f"L1 .npy 文件总数: {l1_count}")

    missing_in_l1_keys = sorted(set(h1_name_map) - set(l1_name_map))
    missing_in_h1_keys = sorted(set(l1_name_map) - set(h1_name_map))

    if missing_in_l1_keys:
        print(f"\nL1 缺失 {len(missing_in_l1_keys)} 个文件:")
        for key in missing_in_l1_keys:
            print(f"  - {h1_name_map[key]}")
    if missing_in_h1_keys:
        print(f"\nH1 缺失 {len(missing_in_h1_keys)} 个文件:")
        for key in missing_in_h1_keys:
            print(f"  - {l1_name_map[key]}")
    if not missing_in_l1_keys and not missing_in_h1_keys:
        print("\nH1 和 L1 的 .npy 文件一一对应。")

    for target_dir in TARGET_DIRS:
        checked_count = count_npy_files(target_dir)
        empty_files = find_empty_npy_files(target_dir)
        print(f"\n检查目录: {target_dir}")
        print(f"  检查到的 .npy 文件数: {checked_count}")
        total_checked += checked_count

        if not empty_files:
            print("  未发现大小为 0 的 .npy 文件。")
            continue

        print(f"  发现 {len(empty_files)} 个大小为 0 的 .npy 文件。")
        total_empty += len(empty_files)

    print(f"\n总共检查到的 .npy 文件数: {total_checked}")
    print(f"\n检查完成。大小为 0 的 .npy 文件总数: {total_empty}")


if __name__ == "__main__":
    main()
