from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR.parent / "timeflow_v7") not in sys.path:
    sys.path.append(str(CURRENT_DIR.parent / "timeflow_v7"))
import evaluate_test_set as _v7_eval_test

_v7_eval_test.PARAMETER_RANGES = {
    "zp": (0.85, 2.85),
    "alpha_m": (3.2, 3.6),
    "delta_m": (2.0, 10.0),
    "lambda_peak": (0.02, 0.06),
    "sigma_m": (3.52, 3.60),
    "mu_m": (33.2, 35.4),
    "beta_q": (-5.0, 5.0),
}
_v7_eval_test.OUTPUT_DIR = CURRENT_DIR / "test_fig"
_v7_eval_test.OUTPUT_FIG_NAME = "test_set_param_mae_v9.png"
_v7_eval_test.OUTPUT_TXT_NAME = "test_set_metrics_v9.txt"


if __name__ == "__main__":
    _v7_eval_test.main()

