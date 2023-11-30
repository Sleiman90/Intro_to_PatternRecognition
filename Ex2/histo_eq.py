# Implement the histogram equalization in this file
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
#img = Image.open('hello.png').convert('L')

img= cv2.imread('hello.png')
numpydata= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




plt.imshow(numpydata,cmap='gray')
plt.show()




'''
counts_array = np.zeros((1,255))
for i in range(numpydata.shape[0]):
    for j in range(numpydata.shape[1]):
        intensity=numpydata[i,j]
        counts_array[0,intensity]+=1
print(np.sum(counts_array[:91,]))
'''

counts_array = np.zeros(256)
for val in numpydata.flatten():
    counts_array[val] += 1 

'''
counts_array,_=np.histogram(numpydata,bins=256, range=[0, 256])

'''
print(np.sum(counts_array[:90]))
print(counts_array.shape)
fig,ax0=plt.subplots()
ax0.bar(range(256), counts_array)  # Plot the histogram
ax0.set_xlabel('Pixel Intensity')
ax0.set_ylabel('Frequency')
plt.show()

'''
counts=[]
for i in range(256):
     x=np.where(numpydata==i,1,0)  
     cou=np.count_nonzero(x)
     counts.append(cou)
counts_array=np.array(counts) 
'''





#pdf=counts_array/np.sum(counts_array)
pdf=counts_array/len( numpydata.flatten())
#print(pdf)

fig,ax1=plt.subplots()
ax1.semilogy(np.arange(0,256),pdf,color='r')
plt.show()

cdf=[]
for i in range(256):
    cdf_element=np.sum(pdf[:i+1])
    cdf.append(cdf_element)


print(sum(cdf[:90]))
fig,ax2=plt.subplots()
ax2.plot(np.arange(0,256),cdf)

cdf_min=0
for i in range(256):
    x=cdf[i]
    if x!=0:
        cdf_min=x
        break
print(cdf_min)  



new_count_list=[]
for i in range(256):
    new_count=((cdf[i]-cdf_min)/(1-cdf_min))*255
    new_count_list.append(new_count)
fig,ax3=plt.subplots()   
ax3.bar(range(256), new_count_list)  # Plot the histogram
ax3.set_xlabel('Pixel Intensity')
ax3.set_ylabel('Frequency')
plt.show()    

print(new_count_list)



transformed_image = np.zeros_like(numpydata)
for i in range(numpydata.shape[0]):
    for j in range(numpydata.shape[1]):
        pixel_value =numpydata[i, j]
        transformed_image[i, j] = new_count_list[pixel_value]

fig,ax3=plt.subplots()
ax3.imshow(transformed_image,cmap='gray')
