
import numpy as np
from scipy.signal import convolve2d
from PIL import Image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize))
    
    for i in range(ksize):
        for j in range(ksize):
           kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((i - ksize//2)**2 + (j - ksize//2)**2) / (2 * sigma**2))
    
    kernel /= np.sum(kernel)
    
    return kernel


def slow_convolve(arr, k):
    k = np.flipud(np.fliplr(k))
    U, V = np.shape(k)
    I,J= np.shape(arr)
    
    padd_array = np.pad(arr, [(U// 2, U // 2), (V // 2, V // 2)], mode='constant')
    
    new_image = np.zeros_like(arr)
    
    for i in range(I):
        for j in range(J):
            for u in range(U):
                for v in range(V):
                    new_image[i, j] += k[u , v ] * padd_array[i + u, j + v]
 
    return new_image





if __name__ == '__main__':
    k = make_kernel(5,5 )   # Updated kernel size and sigma

    # Choose the image you prefer
    # im = np.array(Image.open('input2.jpg'))
    #im = np.array(Image.open('input3.jpg').convert('L'))
    im = np.array(Image.open('input1.jpg').convert('L'))
    plt.imshow(im,cmap='gray')
    plt.show()
    
    # Perform convolution
    result= slow_convolve(im,k)
    
    # Compute the unsharp mask
    #unsharp_mask = im - smoothed_image

    # Enhance contrast by adding the unsharp mask to the original image
    #result = im + unsharp_mask
    
    #result = np.clip(result, 0, 255).astype(np.uint8)
    plt.imshow(result,cmap='gray')
    plt.show()

    #Image.fromarray(result).save('res.jpg')
    #Image.fromarray(result, mode='RGB').save('res.jpg')


    Image.fromarray(result.astype(np.uint8), mode='L').save('res.jpg')

