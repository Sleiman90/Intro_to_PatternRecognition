import numpy as np
import math
import matplotlib.pyplot as plt
def createTriangleSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    t = np.linspace(0, 1, samples)
    f_triangle=np.zeros(samples)
    for i in range(1,k_max+1,2): 
    # triangle wave is composed of odd harmonics only, and skipping even values of i helps generate the correct shape of the waveform
     
       
       f_triangle += (-1)**((i-1)//2)*np.sin(2*np.pi*i*frequency*t)/i**2

    f_triangle *=(8/np.pi**2)

    return f_triangle


def createSquareSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    t = np.linspace(0, 1, samples,endpoint='False')
    f_square=np.zeros(samples)
    for i in range(1,k_max+1):
        
        f_square+=((4/(np.pi))*(np.sin(2*np.pi*((2*i)-1)*frequency*t))/((2*i)-1))
           
    return f_square


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    t = np.linspace(0, 1, samples)
    term = amplitude / 2
    f_saw=np.zeros(samples)
    for i in range(1,k_max+1):
        
        
        f_saw +=(amplitude/np.pi)*(np.sin(2*np.pi*i*frequency*t)/i)
        
    f_saw=term -f_saw
     

    return f_saw


