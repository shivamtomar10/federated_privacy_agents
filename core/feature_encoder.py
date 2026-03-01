import pandas as pd

# Fixed categorical schema for federated stability
AGE_CATEGORIES = ["young", "middle", "senior"]

def encode(df):
    """
    Encodes categorical features into numeric form for federated learning.
    
    - One-hot encodes categorical columns
    - Ensures consistent age category columns across all clients
    - Sorts columns to maintain consistent model dimension
    """

    df = df.copy()

    # --- Ensure age column follows fixed category order ---
    if "age" in df.columns:
        df["age"] = pd.Categorical(df["age"], categories=AGE_CATEGORIES)

    # --- One-hot encoding ---
    df = pd.get_dummies(df, drop_first=True)

    # --- Ensure required age dummy columns always exist ---
    required_age_columns = ["age_middle", "age_senior"]

    for col in required_age_columns:
        if col not in df.columns:
            df[col] = 0

    # --- Sort columns for deterministic order (critical in FL) ---
    df = df.reindex(sorted(df.columns), axis=1)

    return df

