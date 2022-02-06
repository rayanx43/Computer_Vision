import math
import skimage.io  #included the package
import numpy as np
from PIL import Image
from skimage import color, io
from skimage.color import rgb2gray


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    

    ### YOUR CODE HERE
    
    #using function skimage.io.imread to get the images
    out = skimage.io.imread(fname= image_path)
    
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    

    ### YOUR CODE HERE
    
    #copying the array of pixels into x_p
    x_p = image 
    
    #getting a new image by changing the value of every single previous pixel
    out = 0.5*x_p*x_p
    
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    

    ### YOUR CODE HERE
    
    #using the function rgb2gray to convert to gray scale
    #was done using library skimage.color as defined above
    out = rgb2gray(image)
    
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    

    ### YOUR CODE HERE
    
    #option if R is to be excluded
    if channel == 'R':
        #copying image in arguement
        out = image.copy()
        #excluding R
        out[:,:,0] = 0
    
    #option if G is to be excluded
    elif channel == 'G':
        #copying image in arguement
        out = image.copy()
        #excluding G
        out[:,:,1] = 0  
    
    #option if B is to be excluded
    elif channel == 'B':
        #copying image in arguement
        out = image.copy()
        #excluding B
        out[:,:,2] = 0    
    
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
 

    ### YOUR CODE HERE
    
    #option if only L is to be returned 
    if channel == 'L':
        #including L
        out = lab[:,:,0] 
    
    #option if only A is to be returned
    elif channel == 'A':
        #including A
        out = lab[:,:,1]   
    
    #option if only B is to be returned
    elif channel == 'B':
        #including B
        out = lab[:,:,2]  
    
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    

    ### YOUR CODE HERE
    
    #option if only H is to be returned 
    if channel == 'H':
        #including H
        out = hsv[:,:,0] 
    
    #option if only S is to be returned
    elif channel == 'S':
        #including S
        out = hsv[:,:,1]   
    
    #option if only V is to be returned
    elif channel == 'V':
        #including V
        out = hsv[:,:,2]
    
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    
    ### YOUR CODE HERE
    
    #splitting image1 into two parts
    split_image1 = np.hsplit(image1, 2)
    #getting the left half of image1
    left_half = split_image1[0]
    #splitting image2 into two parts
    split_image2 = np.hsplit(image2,2)
    #getting the right half of image1
    right_half = split_image2[1]
    
    #concatenating the two halves; including the RGB exclusion
    out = np.concatenate((rgb_exclusion(left_half, channel1),rgb_exclusion(right_half, channel2)), axis=1)
    
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    

    ### YOUR CODE HERE
    
    #getting half the width and height of the image
    h_half = int(image.shape[0]/2)
    w_half = int(image.shape[1]/2)
    
    #obtaining 4 equal quadrants
    top_left = image[: h_half, : w_half]
    top_right = image[: h_half, w_half :] 
    bottom_left = image[h_half :,: w_half]
    bottom_right = image[h_half :, w_half :]
    
    #removing R from this part
    top_left = rgb_exclusion(top_left, 'R')
    
    #brightening this part
    bottom_left = np.sqrt(bottom_left)
    
    #putting back the left side together
    left_part = np.concatenate((top_left,bottom_left), axis=0)
    
    #dimming this part
    top_right = dim_image(top_right)
    
    #removing R from this part
    bottom_right = rgb_exclusion(bottom_right, 'R')
    
    #putting back the right side together
    right_part = np.concatenate((top_right,bottom_right), axis=0)
    
    #putting back the left and right sides
    out = np.concatenate((left_part, right_part), axis=1)
    
    ### END YOUR CODE

    return out
