PII_KEYWORDS = [
    "name", "email", "address", "phone", "mobile", "id"
]


def detect_sensitive_columns(schema):
    sensitivity = {}

    for col in schema:
        if any(k in col.lower() for k in PII_KEYWORDS):
            sensitivity[col] = "HIGH"
        else:
            sensitivity[col] = "LOW"

    return sensitivity

