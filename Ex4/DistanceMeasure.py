'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    # Calculate the absolute differences between corresponding elements
    absolute_diff = np.abs(Rx - Ry)

    # Sum the absolute differences
    sum_diff = np.sum(absolute_diff)

    # Normalize the sum by the number of features
    num_features = len(Rx)
    norm_diff = sum_diff / num_features

    return norm_diff
    

def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    # Calculate lxx, lyy, and lxy
    lxx = np.sum((Thetax - np.mean(Thetax)) ** 2)
    lyy = np.sum((Thetay - np.mean(Thetay)) ** 2)
    lxy = np.sum((Thetax - np.mean(Thetax)) * (Thetay - np.mean(Thetay)))

    # Calculate the similarity index
    numerator = 1 - (lxy ** 2) / (lxx * lyy)
    similarity_index = numerator * 100

    return similarity_index
    
