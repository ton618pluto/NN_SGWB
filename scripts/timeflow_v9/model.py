from __future__ import annotations

import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from timeflow_v7.model import GWFlowModelV7
else:
    from ..timeflow_v7.model import GWFlowModelV7


class GWFlowModelV9(GWFlowModelV7):
    pass

