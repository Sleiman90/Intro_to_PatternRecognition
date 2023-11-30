import numpy as np
#
# NO OTHER IMPORTS ALLOWED
#
from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('contrast.jpg').convert("L")
plt.imshow(img, cmap='gray')
plt.axis('off')  # Optional: Remove axes
plt.show()
def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    histogram,_=np.histogram(img,bins=256, range=[0, 256])
    #np.histogram return the counts and the gray level, we only need the counts
    
    return histogram
y=create_greyscale_histogram(img)
plt.plot(np.linspace(0,256,256),y)
plt.show()

def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    
  

    img_array = np.array(img)
  
    binary_image = np.where(img_array>t, 255, 0)
   

    return binary_image

x=binarize_threshold(img,150)
y=create_greyscale_histogram(x)
plt.plot(np.linspace(0,256,256),y)
plt.imshow(x,cmap='gray')
plt.show()
        

def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    total_pixels=np.sum(hist)
    nb_pixels_left=np.sum(hist[0:theta+1])
    nb_pixels_right=np.sum(hist[theta+1:])
    p0= nb_pixels_left/total_pixels
    p1= nb_pixels_right/total_pixels
    return p0,p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    '''
    mu_0=(1/p0)*np.sum(np.arange(theta+1)*(np.sum(hist[:theta+1])/np.sum(hist)))
    mu_1=(1/p1)*np.sum(np.arange(theta+1,len(hist),1)*np.sum(hist[theta+1:])/np.sum(hist))
    '''
    mu_0=(1/p0)*np.sum(np.arange(theta+1)*hist[:theta+1])
    mu_1=(1/p1)*np.sum(np.arange(theta+1,len(hist),1)*hist[theta+1:])

    return mu_0,mu_1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables
    max_inter = 0
    threshold=0
    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 
    normalized_hist = hist / np.sum(hist)

    # TODO loop through all possible thetas
    for theta in range(len(normalized_hist)):
        # TODO compute p0 and p1 using the helper function
        p0,p1=p_helper(hist, theta)
        # TODO compute mu and m1 using the helper function
        mu_0,mu_1=mu_helper(hist, theta, p0, p1)
        # TODO compute variance
        mu = p0 * mu_0 + p1 * mu_1
        variance = p0*(mu_0-mu)**2+p1*(mu_1-mu)**2
        # TODO update the threshold
        if variance > max_inter:
           max_inter = variance
           threshold = theta
    return threshold

print(calculate_otsu_threshold(y))
def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    
    histo=create_greyscale_histogram(img)
    theta=calculate_otsu_threshold(histo)
    image_binarized=binarize_threshold(img, theta)
    # TODO
    return  image_binarized
plt.imshow(otsu(img),cmap='gray')
plt.show()