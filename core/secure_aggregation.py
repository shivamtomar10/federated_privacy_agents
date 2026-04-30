# core/secure_aggregation.py

import numpy as np

def unmask_aggregate(masked_updates, masks, robust=True):
    """
    Recovers updates and aggregates. 
    If robust=True, uses Median for Byzantine resistance.
    """
    masked_updates = np.array(masked_updates)
    masks = np.array(masks)
    
    # Recover original updates
    unmasked_updates = masked_updates - masks

    if robust:
        # BYZANTINE RESISTANT: Coordinate-wise median
        # Even if one update is 1,000,000, it won't break the model.
        return np.median(unmasked_updates, axis=0)
    
    # Standard FedAvg
    return np.mean(unmasked_updates, axis=0)