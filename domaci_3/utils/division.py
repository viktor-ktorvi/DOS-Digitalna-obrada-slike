import numpy as np

def get_mass_division(hist, division):
    for i in range(len(hist)):
        if np.sum(hist[0:i]) / np.sum(hist) > division:
            return i
