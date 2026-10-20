"""Feature engineering based on how the machine physically works."""

import numpy as np
import pandas as pd

BASE_FEATURES = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
ENGINEERED_FEATURES = ["temp_diff_k", "power_w", "strain"]
CATEGORICAL_FEATURES = ["type"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically meaningful features.

    temp_diff_k  Process minus air temperature. A small difference means the
                 machine can't get rid of heat well (heat dissipation failures).
    power_w      Mechanical power = torque * angular velocity (rad/s).
                 Too little or too much power stresses the machine.
    strain       Tool wear * torque. Worn tools under high load tend to break
                 (overstrain failures).
    """
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rpm"] * 2 * np.pi / 60
    out["strain"] = out["tool_wear_min"] * out["torque_nm"]
    return out
