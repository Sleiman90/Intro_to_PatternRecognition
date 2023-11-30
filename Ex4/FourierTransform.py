'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!



def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    center_x = shape[1] // 2  # x-coordinate of the image center
    center_y = shape[0] // 2  # y-coordinate of the image center

    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)

    return y,x


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img:
    :return:
    '''
    # Apply FFT to the image
    fft = np.fft.fftshift(np.fft.fft2(img))

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(fft)

    # Convert magnitude spectrum to decibels
    magnitude_spectrum_db = 20 * np.log10(magnitude_spectrum)

    return magnitude_spectrum_db
    

def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    num_rows, num_cols = magnitude_spectrum.shape
    feature_vector = []

    max_radius = min(num_rows, num_cols) // 2

    for i in range(k):
        radius = max_radius * (i + 1) // k

        theta_values = np.linspace(0, np.pi, sampling_steps, endpoint=True)
        intensity_sum = 0

        for theta in theta_values:
            y, x = polarToKart(magnitude_spectrum.shape, radius, theta)
            y = int(round(y))
            x = int(round(x))

            # Check if the calculated indices are within the valid range
            if 0 <= y < num_rows and 0 <= x < num_cols:
                intensity_sum += magnitude_spectrum[y, x]

        feature_vector.append(intensity_sum)

    return np.array(feature_vector)

   
    

    

def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have the same length regardless of the angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum: magnitude spectrum of the image
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area
    :return: feature vector of length k
    """
   


    
    num_rows, num_cols = magnitude_spectrum.shape
    feature_vector = []

   
    ray_length = 64

    for i in range(k):
        radius_values = np.linspace(0, ray_length, sampling_steps)
        intensity_sum=0

        for radius in radius_values:
            theta = i * 2 * np.pi / k
            y, x = polarToKart(magnitude_spectrum.shape, radius, theta)
            y = int(round(y))
            x = int(round(x))
            intensity_sum += magnitude_spectrum[y, x]

        feature_vector.append(intensity_sum)

    return np.array(feature_vector)








def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    # Step 1: Calculate the magnitude spectrum
    magnitude_spectrum = calculateMagnitudeSpectrum(img)

    # Step 2: Extract features using the fan-like method
    fan_features = extractFanFeatures(magnitude_spectrum, k, sampling_steps)

    # Step 3: Extract features using another method (e.g., ring features)
    ring_features = extractRingFeatures(magnitude_spectrum, k, sampling_steps)

    # Return the extracted feature vectors
    return fan_features, ring_features
    
