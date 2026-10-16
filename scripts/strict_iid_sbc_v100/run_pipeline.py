from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERATION_MAX_RESTARTS = 20
GENERATION_RESTART_DELAY_SECONDS = 2.0


def run(script_name: str) -> None:
    script_path = THIS_DIR / script_name
    print(f"\n=== Running {script_path.name} ===", flush=True)
    subprocess.run([sys.executable, str(script_path)], cwd=THIS_DIR, check=True)


def run_generation_with_restarts() -> None:
    script_path = THIS_DIR / "generate_iid_dataset.py"
    for restart_index in range(GENERATION_MAX_RESTARTS + 1):
        print(f"\n=== Running {script_path.name} ===", flush=True)
        completed = subprocess.run([sys.executable, str(script_path)], cwd=THIS_DIR)
        if completed.returncode == 0:
            return

        if restart_index >= GENERATION_MAX_RESTARTS:
            raise RuntimeError(
                f"IID data generation failed {GENERATION_MAX_RESTARTS + 1} "
                "consecutive times. The same population may be causing a "
                "persistent native-library failure; inspect the last displayed "
                "population index before continuing."
            )

        if completed.returncode < 0:
            failure = f"signal {-completed.returncode}"
        else:
            failure = f"exit code {completed.returncode}"
        print(
            f"Generation process stopped with {failure}. "
            f"Restarting from saved files "
            f"({restart_index + 1}/{GENERATION_MAX_RESTARTS})...",
            flush=True,
        )
        time.sleep(GENERATION_RESTART_DELAY_SECONDS)


def main() -> None:
    run("generate_iid_hyperparameters.py")
    run_generation_with_restarts()
    run("evaluate_iid_sbc.py")


if __name__ == "__main__":
    main()
