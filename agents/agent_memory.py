import json
import os
from statistics import mean

# File to store all agent experiences
MEMORY_FILE = "agent_memory.json"

def load_memory():
    """
    Load all past agent experiences from JSON.
    Returns an empty list if file does not exist or is corrupted.
    """
    if not os.path.exists(MEMORY_FILE):
        return []

    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        print("⚠️ Warning: memory file is empty or corrupted. Starting fresh.")
        return []


def save_experience(experience, max_memory=None):
    """
    Append a new experience to memory and save it.

    :param experience: dict containing metrics for this run
    :param max_memory: optional, maximum number of experiences to retain
    """
    memory = load_memory()
    memory.append(experience)

    if max_memory is not None:
        memory = memory[-max_memory:]  # keep only the last N experiences

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def summarize_memory():
    """
    Summarizes cumulative experience for strategy learning.

    Returns:
    {
        "avg_mse": float,
        "avg_reward": float,
        "num_experiences": int,
        "bad_epsilons": set,
        "good_epsilons": set
    }
    """
    memory = load_memory()

    if not memory:
        return {}

    return {
        "avg_mse": mean(m.get("score", 0) for m in memory),
        "avg_reward": mean(m.get("reward", 0) for m in memory),
        "num_experiences": len(memory),
        "bad_epsilons": {m.get("epsilon") for m in memory if m.get("reward", 0) == 0},
        "good_epsilons": {m.get("epsilon") for m in memory if m.get("reward", 0) >= 2},
    }

