import json
import os
import matplotlib.pyplot as plt

# Paths
METRICS_DIR = "metrics"
METRICS_FILE = os.path.join(METRICS_DIR, "agent_history.json")

# Ensure metrics directory exists
os.makedirs(METRICS_DIR, exist_ok=True)

# Check if metrics file exists
if not os.path.exists(METRICS_FILE):
    print(f"❌ Metrics file not found: {METRICS_FILE}")
    exit(1)

# Load metrics
with open(METRICS_FILE, "r") as f:
    history = json.load(f)

if not history:
    print("❌ Metrics file is empty.")
    exit(1)

# Prepare data
rounds = [entry["round"] for entry in history]
countries = [entry["country"] for entry in history]
mses = [entry["mse"] for entry in history]
rewards = [entry["reward"] for entry in history]
epsilons = [entry["epsilon"] for entry in history]


# ================================
# Plot function to save figures
# ================================
def save_plot(x, y, xlabel, ylabel, title, filename, marker="o"):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker=marker, label=ylabel)
    plt.xticks(x, countries, rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, filename))
    plt.close()  # Free memory


# Save all plots
save_plot(rounds, mses, "Country / Round", "MSE", "Agent MSE over rounds", "mse_over_rounds.png")
save_plot(rounds, rewards, "Country / Round", "Reward", "Agent Reward over rounds", "reward_over_rounds.png")
save_plot(rounds, epsilons, "Country / Round", "Epsilon", "Agent Epsilon over rounds", "epsilon_over_rounds.png")

print(f"✅ Plots saved to {METRICS_DIR}/")

