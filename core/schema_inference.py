def infer_schema(df):
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

