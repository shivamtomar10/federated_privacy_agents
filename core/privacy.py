import numpy as np
from agents.policy_agent import PolicyAgent

policy = PolicyAgent()


def apply_privacy(df, schema, country):
    df = df.copy()
    for col in df.columns:
        if policy.decide(col, country) == "HIGH":
            df[col] = "***"
    return df


def add_differential_privacy(weights, epsilon=0.5, clip_value=1.0):
    """
    Proper Differential Privacy implementation:
    1. Clip gradients (sensitivity control)
    2. Add Laplace noise scaled as (clip_value / epsilon)
    """
    weights = np.array(weights)

    if epsilon <= 0:
        raise ValueError("Epsilon must be > 0")

    # Gradient clipping
    clipped_weights = np.clip(weights, -clip_value, clip_value)

    # DP noise
    scale = clip_value / epsilon
    noise = np.random.laplace(0, scale, size=weights.shape)

    return clipped_weights + noise

