from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
import chirp
import decomposition
import matplotlib.pyplot as plt
import numpy as np
t=np.linspace(0, 1, int(200 * 1))
# TODO: Test the functions imported in lines 1 and 2 of this file.
# Generate a linear chirp signal
linear_chirp = chirp.createChirpSignal(samplingrate=200, duration=1, freqfrom=1, freqto=10, linear=True)

# Generate an exponential chirp signal
exp_chirp = chirp.createChirpSignal(samplingrate=200, duration=1, freqfrom=1, freqto=10, linear=False)

t=np.linspace(0, 1, int(200 * 1))
fig,ax= plt.subplots(1,2)
ax[0].plot(t,linear_chirp )
ax[1].plot(t,exp_chirp )



TriangleSignal=decomposition.createTriangleSignal(200,2,1000)
SquareSignal=decomposition.createSquareSignal(200,2,1000)
SawtoothSignal= decomposition.createSawtoothSignal(200,2,1000,1)


t1 = np.linspace(0, 1, 200)
fig1,ax1= plt.subplots(1,3)
fig1=plt.figure(figsize=(5,1))
ax1[0].plot(t1,TriangleSignal)
ax1[1].plot(t1,SquareSignal)
ax1[2].plot(t1,SawtoothSignal)
