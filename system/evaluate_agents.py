import os
import json
from agents.hospital_agent import HospitalAgent
from agents.agent_memory import load_memory

# Paths
DATA_DIR = "data/"
METRICS_DIR = "metrics"
HISTORY_FILE = os.path.join(METRICS_DIR, "agent_history.json")

os.makedirs(METRICS_DIR, exist_ok=True)

datasets = [
    "india_hospital.csv",
    "usa_hospital.csv"
]

# Load existing history
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE) as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = []
else:
    history = []

# Evaluate each dataset
for round_idx, csv_file in enumerate(datasets, start=1):
    country = csv_file.split("_")[0].capitalize()
    csv_path = os.path.join(DATA_DIR, csv_file)

    print(f"\n🏥 Processing {country} dataset (Round {round_idx})")

    agent = HospitalAgent(country, csv_path)
    result = agent.process()  # returns weights, analysis, strategy

    # Load last memory for this country to get reward & score
    mem = load_memory()
    mem_entry = next((m for m in mem if m["country"].lower() == country.lower()), None)

    entry = {
        "round": round_idx,
        "country": country,
        "rows": result["analysis"]["rows"],
        "mse": mem_entry["score"] if mem_entry else None,
        "reward": mem_entry["reward"] if mem_entry else None,
        "epsilon": result["strategy"]["epsilon"]
    }

    history.append(entry)

# Save updated history
with open(HISTORY_FILE, "w") as f:
    json.dump(history, f, indent=2)

print("\n✅ Evaluation complete. Metrics stored in", HISTORY_FILE)

