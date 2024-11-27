import numpy as np

def non_linear_contrast_stretching_symmetric(image, power):
    xp = np.power(image, power)
    xip = np.power(1.0 - image, power)
    return xp / (xp + xip)

def non_linear_contrast_stretching_asymmetric(image, power, pivot):
    y = image.copy()
    y[image <= pivot] = pivot * np.power(image[image <= pivot] / pivot, power)
    rpivot = 1.0 - pivot
    y[image > pivot] = 1.0 - rpivot * np.power((1.0 - image[image > pivot]) / rpivot, power)
    return y

def non_linear_contrast_stretching_log(image, alpha):
    y = np.log1p(alpha * image)
    y = y / np.max(y)
    return y

def non_linear_contrast_stretching_exp(image, power):
    y = np.power(image, power)
    return y