import pandas as pd

def mask_age(age):
    """
    Convert numerical age into categorical buckets.
    """

    if age < 30:
        return "young"
    elif age < 50:
        return "middle"
    else:
        return "senior"


def sanitize(df):
    """
    Apply full data sanitization:
    - Drop missing values
    - Mask age into categories
    """

    df = df.dropna()

    if "age" in df.columns:
        df["age"] = df["age"].apply(mask_age)

    return df

