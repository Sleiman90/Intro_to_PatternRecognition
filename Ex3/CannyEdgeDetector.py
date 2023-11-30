import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import convo
from convo import make_kernel
from PIL import Image
import math
import cv2
from scipy.ndimage import gaussian_filter
#
# NO MORE MODULES ALLOWED
#

img_in = np.array(Image.open("contrast.jpg").convert("L"))


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # kernel = gaussian_filter(np.zeros((ksize, ksize)), sigma)
    kernel = convo.make_kernel(ksize, sigma)
    filtered_image = convolve(img_in, kernel)
    filtered_image = filtered_image.astype(int)
    return kernel, filtered_image



def sobel(img_in):
    # Define Sobel filters
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Apply Sobel filters using convolution
    gx = convolve(img_in, sobel_x).astype(int)
    gy = convolve(img_in, sobel_y).astype(int)

    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO
    g = np.sqrt(np.square(gx) + np.square(gy)).astype(int)
    theta = np.arctan2(gy, gx).astype(float)
    return g, theta


def convertAngle(theta):
    degrees = math.degrees(theta) % 180
    if 0 <= degrees < 22.5 or 157.5 <= degrees < 180:
        degrees = 0
    elif 22.5 <= degrees <= 67.5:
        degrees = 45
    elif 67.5 <= degrees < 112.5:
        degrees = 90
    elif 112.5 <= degrees < 157.5:
        degrees = 135

    return degrees


def maxSuppress(g, theta):
    """
    Calculate maximum suppression.
    :param g: 2D image (np.ndarray)
    :param theta: 2D image (np.ndarray)
    :return: max_sup: Maximum suppressed image (np.ndarray)
    """

    g = np.pad(g, [(1, 1), (1, 1)], 'constant')
    theta = np.pad(theta, [(1, 1), (1, 1)], 'constant')
    max_sup = np.zeros_like(g)

    for x in range(1, g.shape[0] - 1):
        for y in range(1, g.shape[1] - 1):
            # Compute angle for the current pixel
            angle = convertAngle(theta[x, y])
            if angle == 0:
                if g[x, y] >= g[x, y+1] and g[x, y] >= g[x, y-1]:
                    max_sup[x, y] = g[x, y]
            elif angle == 45:
                if g[x, y] >= g[x - 1, y + 1] and g[x, y] >= g[x + 1, y - 1]:
                    max_sup[x, y] = g[x, y]
            elif angle == 90:
                if g[x, y] >= g[x + 1, y] and g[x, y] >= g[x - 1, y]:
                    max_sup[x, y] = g[x, y]
            elif angle == 135:
                if g[x, y] >= g[x - 1, y - 1] and g[x, y] >= g[x + 1, y + 1]:
                    max_sup[x, y] = g[x, y]

    max_sup = max_sup[1:- 1, 1: - 1]
    return max_sup


def hysteris(imgin, tlow, thigh):
    """
    Perform hysteris thresholding.

    :param imgin: Input image (np.ndarray)
    :param tlow: Lower threshold value (int)
    :param thigh: Upper threshold value (int)
    :return: Hysteris thresholded image (np.ndarray)
    """
    threshimg = np.where(imgin <= tlow, 0, imgin)
    threshimg1 = np.where(np.logical_and(
        threshimg > tlow, threshimg <= thigh), 1, threshimg)
    threshimg2 = np.where(threshimg1 > thigh, 2, threshimg1)
    print(threshimg2)
    threshimg2 = np.pad(threshimg2, [(1, 1), (1, 1)], 'constant')
    new_array = np.zeros_like(threshimg2)
    width, height = new_array.shape
    for x in range(1, width-1):
        for y in range(1, height-1):
            if threshimg2[x, y] == 2:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if threshimg2[x + dx, y+dy] == 1 or threshimg2[x + dx, y+dy] == 2:

                            new_array[x+dx, y+dy] = 2

    new_array = np.where(new_array == 2, 255, new_array)
    new_array = new_array[1:-1, 1:-1]
    return new_array


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)
    plt.imshow(result, 'gray')
    plt.show()


print(canny(img_in))

