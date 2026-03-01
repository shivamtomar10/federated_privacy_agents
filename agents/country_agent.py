import numpy as np
from core.secure_aggregation import mask_update, unmask_aggregate


class CountryAgent:
    def aggregate(self, hospital_updates):
        masked = []
        masks = []

        for update in hospital_updates:
            m, mask = mask_update(update)
            masked.append(m)
            masks.append(mask)

        return unmask_aggregate(masked, masks) / len(hospital_updates)

