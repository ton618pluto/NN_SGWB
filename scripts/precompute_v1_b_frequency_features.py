from pathlib import Path

import numpy as np
from scipy.signal import csd, welch


SCRIPTS_DIR = Path(__file__).resolve().parent

INPUT_FOLDER = "processed_data_train"
VERSION = "v21"
OUTPUT_FOLDER = "processed_data_freq_features"

SAMPLING_FREQUENCY = 64
NPERSEG = 64
INPUT_SCALE = 1e23
EPS = 1e-12


def sample_key(file_name: str) -> str:
    return Path(file_name).stem.split("-", 1)[-1]


def load_channel(path: Path) -> tuple[np.ndarray, dict]:
    sample = np.load(path, allow_pickle=True).item()
    data = np.asarray(sample["data"], dtype=np.float32) * INPUT_SCALE
    data = data - data.mean(dtype=np.float32)
    return data, sample


def compress_signed(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def compute_input_features(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    _, psd1 = welch(h1, fs=SAMPLING_FREQUENCY, nperseg=NPERSEG)
    _, psd2 = welch(h2, fs=SAMPLING_FREQUENCY, nperseg=NPERSEG)
    _, cpsd = csd(h1, h2, fs=SAMPLING_FREQUENCY, nperseg=NPERSEG)

    coherence = (np.abs(cpsd) ** 2) / (psd1 * psd2 + EPS)
    coherence = np.clip(coherence, 0.0, 1.0)

    return np.stack(
        [
            np.log(psd1 + EPS),
            np.log(psd2 + EPS),
            compress_signed(cpsd.real),
            compress_signed(cpsd.imag),
            coherence,
        ],
        axis=0,
    ).astype(np.float32)


def build_paths() -> tuple[Path, Path, Path]:
    input_base = SCRIPTS_DIR / INPUT_FOLDER / VERSION
    h1_dir = input_base / "H1_splits"
    l1_dir = input_base / "L1_splits"
    output_dir = SCRIPTS_DIR / OUTPUT_FOLDER / VERSION / f"fs{SAMPLING_FREQUENCY}_nperseg{NPERSEG}"
    return h1_dir, l1_dir, output_dir


def is_valid_existing_feature_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        sample = np.load(path, allow_pickle=True).item()
        data = np.asarray(sample["data"])
        return data.ndim == 2 and data.shape[0] == 5
    except Exception:
        return False


def main() -> None:
    h1_dir, l1_dir, output_dir = build_paths()
    if not h1_dir.exists():
        raise FileNotFoundError(f"H1 directory not found: {h1_dir}")
    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 directory not found: {l1_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    h1_files = sorted((path for path in h1_dir.iterdir() if path.is_file() and path.suffix == ".npy"), key=lambda p: p.name)
    l1_map = {
        sample_key(path.name): path
        for path in l1_dir.iterdir()
        if path.is_file() and path.suffix == ".npy"
    }

    saved = 0
    skipped_existing = 0
    skipped_missing_pair = 0
    failed = 0

    print(f"Input H1 dir: {h1_dir}")
    print(f"Input L1 dir: {l1_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Sampling frequency: {SAMPLING_FREQUENCY}")
    print(f"nperseg: {NPERSEG}")

    total_files = len(h1_files)
    for index, h1_path in enumerate(h1_files, start=1):
        key = sample_key(h1_path.name)
        l1_path = l1_map.get(key)
        if l1_path is None:
            skipped_missing_pair += 1
            if index % 200 == 0:
                print(
                    f"Processed {index}/{total_files} files | "
                    f"saved={saved} skipped_existing={skipped_existing} "
                    f"skipped_missing_pair={skipped_missing_pair} failed={failed}"
                )
            continue

        output_path = output_dir / f"feat-{key}.npy"
        if is_valid_existing_feature_file(output_path):
            skipped_existing += 1
            continue
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

        try:
            h1, h1_sample = load_channel(h1_path)
            h2, _ = load_channel(l1_path)
            features = compute_input_features(h1, h2)

            payload = {
                "data": features,
                "label": np.asarray(h1_sample["label"], dtype=np.float32),
                "source_h1": str(h1_path),
                "source_l1": str(l1_path),
                "sampling_frequency": SAMPLING_FREQUENCY,
                "nperseg": NPERSEG,
            }
            if "st" in h1_sample:
                payload["st"] = h1_sample["st"]

            np.save(output_path, payload, allow_pickle=True)
            saved += 1
        except Exception as exc:
            failed += 1
            print(f"Failed on {h1_path.name}: {exc}")

        if index % 200 == 0:
            print(
                f"Processed {index}/{total_files} files | "
                f"saved={saved} skipped_existing={skipped_existing} "
                f"skipped_missing_pair={skipped_missing_pair} failed={failed}"
            )

    print("\nDone.")
    print(f"Saved feature files: {saved}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped missing pairs: {skipped_missing_pair}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
