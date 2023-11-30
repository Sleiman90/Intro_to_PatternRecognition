import numpy as np
import math


def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    # TODO
    c=(freqto-freqfrom)/duration
    k=(freqto/freqfrom)**(1/duration)
    t = np.linspace(0, duration, samplingrate * duration) #sampling rate refers to the number of samples taken per second  so we apply third rule

    if linear:
        x_t = np.sin(2 * np.pi * (freqfrom + ((c*t)/ 2))*t)
    else:
        x_t = np.sin(2 * np.pi * freqfrom / (np.log(k)) * (k ** t - 1))
    return x_t



