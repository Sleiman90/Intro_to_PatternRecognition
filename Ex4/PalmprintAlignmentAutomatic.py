'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!
img=cv2.imread('Hand1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img,'gray')
imgc=img[:]
def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv2.circle(img, (x, y), 4, 255, 5)
    return img

#circle= drawCircle(imgc,10,40)
circle= drawCircle(imgc,44,72)
circle= drawCircle(imgc,10,55)
circle= drawCircle(imgc,30,65)

plt.imshow(circle,'gray')
plt.show()

def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    # Binarize using threshold of 115
    _, binary_img =cv2.threshold(img,115,255,cv2.THRESH_BINARY) # intensities greater than or equal to 115 are set to 255 (white), while pixels below the threshold are set to 0 (black)

    # Smooth with Gaussian kernel (5, 5)
    preprocessed_image= cv2.GaussianBlur(binary_img, (5, 5), 0)
    return preprocessed_image
processed=binarizeAndSmooth(img)
plt.imshow(processed,'gray')
plt.show()

def drawLargestContour(img) -> np.ndarray:
        '''
        find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
        :param img: preprocessed image (mostly b&w)
        :return: contour image
        '''
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a blank image with the same dimensions as the input image
        contour_image = np.zeros_like(img)

        # Draw the largest contour on the blank image
        cv2.drawContours(contour_image, [largest_contour],0,255, 2)

        

        return contour_image
contour=drawLargestContour(processed)
plt.imshow(contour,'gray')
plt.show()
print(contour)



def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    
    contour_img=contour_img[4:235,:]
    plt.imshow(contour_img,'gray')
    plt.show()
    
    img=contour_img[:,x]
    intersections = []
    height, _ = np.shape(contour_img)
    should_append = True

    for y in range(2,height-1):
        y = int(y)
        if img[y] == 255:
            
            intersections.append(y)
    filtered_numbers = []       
    for num in intersections:
      if should_append:
         # Append the number to the filtered list
         filtered_numbers.append(num)
   
      # Check the condition to determine if the next number should be skipped
      if num + 1 in intersections and num + 1 - num <= 10:
        should_append = False
      else:
        should_append = True
    # Trim the intersections list to contain at most 6 values
    intersection = filtered_numbers[:6]

    return np.array(intersection)
intersections=getFingerContourIntersections(contour, 10)
print(intersections)

def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    
    slope = (y2 - y1) / (x2 - x1)  # Calculate the slope
    b = y1 -( slope * x1)
    
    x = x2
    y=y2
    while True:
        

        # Check if the current coordinates belong to the contour
        if img[y, x] !=0:
            return y, x  # Return the intersection point
            break
        x += 1
        y = int((slope * x) + b)
        

    return None  # Return None if no intersection is found
   
    
print(findKPoints(contour,55,10,65,30))

def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    
    # Convert points to numpy arrays
    k1 = np.array(k1)
    k2 = np.array(k2)
    k3 = np.array(k3)

    # Compute the direction vector of the line passing through k1 and k3
    line_direction = k3 - k1
    line_direction = line_direction / np.linalg.norm(line_direction)

    # Compute the direction vector of the line passing through k2 and perpendicular to the line k1k3
    line_perpendicular = np.array([-line_direction[1], line_direction[0]])

    # Calculate the intersection point of the two lines
    intersection_point = np.linalg.solve(np.vstack((line_direction, -line_perpendicular)).T, k2 - k1)

    # Calculate the angle between the line passing through the intersection and the old x-axis
    angle = np.arctan2(intersection_point[1], intersection_point[0]) * 180 / np.pi
    print(angle)
    # Create the transformation matrix using cv2.getRotationMatrix2D with the specified center
    scale = 1  # You can adjust the scale if needed
    center = tuple(intersection_point)
    transform_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    return transform_matrix

def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # TODO threshold and blur

    # TODO find and draw largest contour in image

    # TODO choose two suitable columns and find 6 intersections with the finger's contour

    # TODO compute middle points from these contour intersections

    # TODO extrapolate line to find k1-3

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3

    # TODO rotate the image around new origin
    # Step 1: Threshold and blur the image
    processed = binarizeAndSmooth(img)

    # Step 2: Find and draw the largest contour in the image
    contour = drawLargestContour(processed)

    # Step 3: Choose two suitable columns and find 6 intersections with the finger's contour
    x1 = 10  # Choose the x-coordinate of the first column
    x2 = 20  # Choose the x-coordinate of the second column
    intersections1 = getFingerContourIntersections(contour, x1)
    intersections2 = getFingerContourIntersections(contour, x2)

    # Step 4: Compute middle points from these contour intersections
    middle_points_x1 = []
    for i in range(0,len(intersections1),2):
        y = (intersections1[i+1] - intersections2[i]) // 2
        middle_points_x1.append((y, x1))
    middle_points_x2 = []
    for i in range(0,len(intersections2),2):
        y = (intersections2[i+1] - intersections2[i]) // 2  
        middle_points_x2.append((y, x2))
        
        
    
    # Step 5: Extrapolate line to find k1-3
    k1 = findKPoints(contour, middle_points_x1[0][0], middle_points_x1[0][1], middle_points_x2[0][0], middle_points_x2[0][1])
    k2 = findKPoints(contour, middle_points_x1[1][0], middle_points_x1[1][1], middle_points_x2[1][0], middle_points_x2[1][1])
    k3 = findKPoints(contour, middle_points_x1[2][0], middle_points_x1[2][1], middle_points_x2[2][0], middle_points_x2[2][1])

    # Step 6: Calculate Rotation matrix from coordinate system spanned by k1-3
    rotation_matrix = getCoordinateTransform(k1, k2, k3)

    # Step 7: Rotate the image around the new origin
    aligned_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

    return aligned_image
    
rotated=palmPrintAlignment(img)
plt.imshow(rotated,'gray')
plt.show()