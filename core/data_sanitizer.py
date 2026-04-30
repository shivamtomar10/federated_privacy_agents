# core/data_sanitizer.py

import pandas as pd
import numpy as np


def sanitize(df):
    """
    Apply data sanitization:
    - Fill missing values instead of dropping rows
    - Detects column type safely (handles pandas StringDtype, object, numeric)
    - Keeps age as numeric so hospital_agent can bucket it consistently
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue  # nothing to fill

        # Try to coerce to numeric first
        as_numeric = pd.to_numeric(df[col], errors="coerce")
        mostly_numeric = as_numeric.notna().sum() / max(len(df), 1) > 0.5

        if mostly_numeric:
            # Fill with median of numeric values
            median_val = as_numeric.median()
            df[col] = as_numeric.fillna(median_val)
        else:
            # Fill with mode (most frequent value)
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])

    return df
