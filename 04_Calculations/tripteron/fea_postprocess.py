"""Post-processing of the SolidWorks Simulation exports of the chassis.

The static and buckling studies of delivery 3 were exported as CSV lists of
sampled nodes.  This module reads them, converts the Spanish number format,
and produces the summary table used in the corrected calculation report.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "fea_exports"

FILES = {
    "Bending stress (MPa)": "stress_bending_TauXY.csv",
    "Axial stress (MPa)": "stress_axial_SX.csv",
    "Shear stress dir. 1 (MPa)": "shear_dir1_TauYZ.csv",
    "Shear stress dir. 2 (MPa)": "shear_dir2_P1.csv",
    "Strain (-)": "strain_EPSX.csv",
    "Displacement (mm)": "displacement_URES.csv",
    "Buckling mode 1, X (mm)": "buckling_mode1_AMPX.csv",
    "Buckling mode 1, Y (mm)": "buckling_mode1_AMPY.csv",
    "Buckling mode 1, Z (mm)": "buckling_mode1_AMPZ.csv",
}

BUCKLING_LOAD_FACTOR = 2237.5      # reported by SolidWorks for mode 1


def _to_float(text: str) -> float | None:
    text = text.strip().replace(".", "").replace(",", ".") if re.match(r"^-?\d{1,3}(\.\d{3})+,", text) \
        else text.strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def read_export(path: Path) -> np.ndarray:
    """Values of the sampled nodes of one export."""
    values = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter=";"):
            if len(row) < 2:
                continue
            v = _to_float(row[1])
            if v is not None and _to_float(row[0]) is not None:
                values.append(v)
    return np.array(values)


def summary(folder: Path | None = None) -> list[dict]:
    folder = DATA if folder is None else folder
    out = []
    for label, name in FILES.items():
        path = folder / name
        if not path.exists():
            continue
        v = read_export(path)
        out.append(dict(quantity=label, samples=len(v), minimum=float(v.min()),
                        maximum=float(v.max()), mean=float(v.mean()),
                        abs_max=float(np.abs(v).max())))
    return out


def safety_factors(rows: list[dict], yield_strength: float = 620.0) -> dict:
    """Static and buckling safety factors of the chassis."""
    stresses = [r for r in rows if "stress" in r["quantity"].lower()]
    worst = max(stresses, key=lambda r: r["abs_max"]) if stresses else None
    return dict(
        worst_stress_quantity=worst["quantity"] if worst else None,
        worst_stress_MPa=worst["abs_max"] if worst else None,
        yield_strength_MPa=yield_strength,
        static_safety_factor=yield_strength / worst["abs_max"] if worst else None,
        buckling_load_factor=BUCKLING_LOAD_FACTOR,
    )
