import numpy as np


def mask_update(update):
    """
    Add random mask to client update.
    """
    update = np.array(update)
    mask = np.random.randn(*update.shape)
    return update + mask, mask


def unmask_aggregate(masked_updates, masks, robust=False):
    """
    Remove masks and aggregate updates.

    If robust=True -> use coordinate-wise median
    Else -> use average aggregation
    """

    masked_updates = np.array(masked_updates)
    masks = np.array(masks)

    # Step 1: Recover each client's original update
    unmasked_updates = masked_updates - masks

    if robust:
        # Median across clients (robust)
        return np.atleast_1d(np.median(unmasked_updates, axis=0))

    # Standard federated averaging
    return np.atleast_1d(np.mean(unmasked_updates, axis=0))

