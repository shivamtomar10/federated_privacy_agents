import numpy as np

<<<<<<< HEAD
class GlobalAgent:
    def __init__(self):
        # NEW: Store the state so we don't lose progress between rounds
        self.current_weights = None
=======

class GlobalAgent:
    """
    Federated Aggregator (Matrix-Aware)
    -----------------------------------
    - averages weight matrices (features x classes)
    - handles different shapes via padding
    - no secure aggregation mismatch
    """
>>>>>>> 19b0456 (Initial federated privacy agents code)

    def aggregate(self, country_updates):
        updates = []

        for country, update in country_updates.items():
            if update["strategy"]["risk_level"] == "HIGH":
                print(f"⚠️ Skipping {country} (high risk)")
                continue

            weights = update["weights"]
            updates.append(weights)

        if not updates:
            raise ValueError("❌ No valid updates to aggregate")

<<<<<<< HEAD
        # Find max shape
=======
        # find max shape
>>>>>>> 19b0456 (Initial federated privacy agents code)
        max_rows = max(w.shape[0] for w in updates)
        max_cols = max(w.shape[1] for w in updates)

        padded_updates = []
<<<<<<< HEAD
=======

>>>>>>> 19b0456 (Initial federated privacy agents code)
        for w in updates:
            padded = np.zeros((max_rows, max_cols))
            padded[:w.shape[0], :w.shape[1]] = w
            padded_updates.append(padded)

<<<<<<< HEAD
        # --- THE FIX STARTS HERE ---
        # 1. Calculate the average of the NEW updates
        avg_update = np.mean(padded_updates, axis=0)

        # 2. Use a Global Learning Rate (eta) to dampen noise
        # 0.2 means we keep 80% of old knowledge and add 20% of new findings.
        global_lr = 1.0

        if self.current_weights is None:
            # First round: take the average directly
            self.current_weights = avg_update
        else:
            # Rounds 2+: ADD the new update to the old weights slowly
            # This is the "FedAvg" way to maintain high accuracy
            self.current_weights = self.current_weights + (global_lr * avg_update)

        return self.current_weights, {}
=======
        # average matrices
        global_weights = np.mean(padded_updates, axis=0)

        return global_weights, {}

>>>>>>> 19b0456 (Initial federated privacy agents code)
