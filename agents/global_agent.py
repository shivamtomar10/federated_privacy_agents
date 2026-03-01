import numpy as np


class GlobalAgent:
    """
    Federated Aggregator (Matrix-Aware)
    -----------------------------------
    - averages weight matrices (features x classes)
    - handles different shapes via padding
    - no secure aggregation mismatch
    """

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

        # find max shape
        max_rows = max(w.shape[0] for w in updates)
        max_cols = max(w.shape[1] for w in updates)

        padded_updates = []

        for w in updates:
            padded = np.zeros((max_rows, max_cols))
            padded[:w.shape[0], :w.shape[1]] = w
            padded_updates.append(padded)

        # average matrices
        global_weights = np.mean(padded_updates, axis=0)

        return global_weights, {}

