# agents/global_agent.py

import numpy as np


class GlobalAgent:
    """
    Federated Aggregator (Stable & Tuned)
    -------------------------------------
    - True FedAvg (delta-based updates)
    - Safer gradient clipping
    - Soft normalization instead of hard clipping
    - Prevents norm saturation (fixes constant 1.0 issue)
    """

    def __init__(self):
        self.current_weights = None

    def aggregate(self, filtered_updates):
        if not filtered_updates:
            return self.current_weights, {"error": "No valid updates"}

        # ------------------------------
        # 1️⃣ Total data size
        # ------------------------------
        total_rows = sum(u["analysis"]["rows"] for u in filtered_updates.values())

        # ------------------------------
        # 2️⃣ Find max shape (padding)
        # ------------------------------
        max_rows = max(u["weights"].shape[0] for u in filtered_updates.values())
        max_cols = max(u["weights"].shape[1] for u in filtered_updates.values())

        aggregated_update = np.zeros((max_rows, max_cols))

        # ------------------------------
        # 3️⃣ Aggregate Updates (FedAvg)
        # ------------------------------
        for country, data in filtered_updates.items():
            update = data["weights"]
            rows = data["analysis"]["rows"]

            padded = np.zeros((max_rows, max_cols))
            padded[:update.shape[0], :update.shape[1]] = update

            # ✅ Improved clipping (less aggressive)
            norm = np.linalg.norm(padded)
            if norm > 10.0:
                padded = padded * (10.0 / (norm + 1e-8))

            contribution_weight = rows / total_rows
            aggregated_update += padded * contribution_weight

        # ------------------------------
        # 4️⃣ Apply UPDATE correctly
        # ------------------------------
        if self.current_weights is None:
            # First round → initialize directly
            self.current_weights = aggregated_update
        else:
            # ✅ Tuned global learning rate
            lr_global = 0.15

            # Match shapes safely
            new_weights = np.zeros_like(aggregated_update)

            r = min(self.current_weights.shape[0], aggregated_update.shape[0])
            c = min(self.current_weights.shape[1], aggregated_update.shape[1])

            new_weights[:r, :c] = self.current_weights[:r, :c]

            # ✅ TRUE UPDATE STEP
            self.current_weights = new_weights + lr_global * aggregated_update

        # ------------------------------
        # 5️⃣ Soft Normalization (instead of hard clip)
        # ------------------------------
        norm = np.linalg.norm(self.current_weights)
        if norm > 10.0:
            self.current_weights = self.current_weights * (10.0 / (norm + 1e-8))

        return self.current_weights, {"total_rows": total_rows}

