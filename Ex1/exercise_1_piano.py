from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

sample_freq=44100
def load_sample(filename, duration=4*44100, offset=44100//10):
  
    sound=np.load(filename)
    start=np.argmax(np.abs(sound))+offset 
    #The absolute value operation ensures that we consider the magnitude of the signal rather than its direction
    #argmax return the index
    end=start+duration #index of end
    return sound[start:end]



def compute_frequency(signal, min_freq=20):
    # Complete this function
    
    
    # Compute Fourier transform of signal
    
    
    fft_signal = np.fft.fft(signal) #return the frequency components where each component is complex nb contaiing the amplitude(real) and (imaginary) phase
    
    # Compute magnitudes of Fourier coefficients
    
    magnitudes = np.abs(fft_signal) #it is the sqaure root of the complex number
    
    # Compute frequencies of Fourier coefficients
    freqs = np.fft.fftfreq(len(signal), d=1/44100) #return array rray of length len(signal) containing sample frequencies that are normalized.
    freqs_hz=(freqs*44100)/len(signal)
    #fn=(f*n)/fs
    #f=fn*fs/n
    # Set magnitudes corresponding to frequencies below min_freq to zero
    magnitudes[freqs_hz < min_freq] = 0
    
    # Find index of highest magnitude above min_freq
    index = np.argmax(magnitudes)
    
    # Return frequency corresponding to highest magnitude
    max_freq = freqs_hz[index]
    max_freqq=max_freq*len(signal)/44100 #return it back to normalized form
    return max_freqq

    
a_frequencies = [110.00, 220.00, 440.00, 880.00, 1760.00, 3520.00,1174.659] 
list=[]
if __name__ == '__main__':
    # Implement the code to answer the questions here
    for filename in sorted(os.listdir('sounds')):
        if not filename.endswith('npy'):
            continue
        sample = load_sample(os.path.join('sounds', filename))
        freq = compute_frequency(sample, min_freq=20)
        tone=filename.split('.')[2]
        print('{}: {}'.format(tone, freq))
        list.append(freq)
    
    print(list)
#print(np.allclose(np.array(list),np.array(a_frequencies) ,rtol=10e-10))
      



 
 
