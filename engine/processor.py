import numpy as np

def non_linear_contrast_stretching_symmetric(image, power):
    xp = np.power(image, power)
    xip = np.power(1.0 - image, power)
    return xp / (xp + xip)