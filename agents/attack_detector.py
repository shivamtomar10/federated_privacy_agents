# agents/attack_detector.py

import numpy as np

class AttackDetector:
    def __init__(self, threshold=2.5):
        # Higher threshold = more lenient (less agents kicked out)
        # Lower threshold = stricter (more sensitive to outliers)
        self.threshold = threshold

    def inspect(self, updates_list):
        """
        Uses Z-Score to identify Byzantine (malicious or broken) weight updates.
        """
        if len(updates_list) < 2:
            return list(range(len(updates_list)))

        # Flatten weights to calculate a magnitude (norm) for each country
        norms = np.array([np.linalg.norm(u) for u in updates_list])
        
        mean_norm = np.mean(norms)
        std_norm = np.std(norms) + 1e-9 # Avoid division by zero
        
        # Calculate how many standard deviations each update is from the mean
        z_scores = np.abs((norms - mean_norm) / std_norm)
        
        # Keep agents whose Z-score is within our safety threshold
        clean_indices = [i for i, z in enumerate(z_scores) if z < self.threshold]

        if len(clean_indices) < len(updates_list):
            diff = len(updates_list) - len(clean_indices)
            print(f"🛡️ [AttackDetector] Detected {diff} anomalous updates. Filtering them out...")

        return clean_indices