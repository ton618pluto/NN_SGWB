from pathlib import Path


VERSION = "v61"


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    base_dir = scripts_dir / "noise_waveform" / VERSION
    h1_dir = base_dir / "noise_waveform_H1"
    l1_dir = base_dir / "noise_waveform_L1"

    if not h1_dir.exists():
        raise FileNotFoundError(f"H1 noise directory not found: {h1_dir}")
    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 noise directory not found: {l1_dir}")

    h1_files = sorted(h1_dir.glob("H1-STRAIN-*-duration.gwf"))
    l1_files = sorted(l1_dir.glob("L1-STRAIN-*-duration.gwf"))

    print(f"Noise base dir: {base_dir}")
    print(f"H1 directory: {h1_dir}")
    print(f"L1 directory: {l1_dir}")
    print()
    print(f"H1 noise file count: {len(h1_files)}")
    print(f"L1 noise file count: {len(l1_files)}")


if __name__ == "__main__":
    main()
