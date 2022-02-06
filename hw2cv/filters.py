import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    
    #using 4 for-loops to iterate through both image and kernel
    for x in range (Hi):
        for y in range (Wi):
            #we want the sum for each iteration of the kernal
            sum = 0
            for i in range (Hk):
                for j in range (Wk):
                    #using an if condition to keep loop within bounds
                    if x+1-i < 0 or y+1-j < 0 or x+1-i >= Hi or y+1-j >= Wi:
                        #if out of bounds, no increment is made
                        sum = sum + 0
                    else:    
                        #calculating sum using formula for convolution when within bounds
                        sum = sum + kernel[i][j]*image[x+1-i][y+1-j]
            #Assigning each new value to the output matrix after calculation            
            out[x][y] = sum
                    
    
     ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    hp = pad_height
    wp = pad_width
    

    ### YOUR CODE HERE
    
    #using the numpy function to zero pad the image
    out = np.pad(image, ((hp, hp), (wp, wp)), 'constant')
    
    
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    
    #zero padding the image using parameters of the kernel
    #such that no bounds are exceeded during convolution
    image = zero_pad(image, Hk//2, Wk//2)
    #flipping kernel such that it agrees with the equation
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)
    for x in range(Hi):
        for y in range(Wi):
            #getting sum of the product of the image and kernel parts
            out[x, y] =  np.sum(image[x: x+Hk, y: y+Wk] * kernel)
    
    
    
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    
    #flip the template such that a convolution can be performed (see equation for cross-correlation)
    g = np.flip(np.flip(g, 0), 1)
    #perform the convolution
    out = conv_fast(f, g)
    
    
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    
    #subtracting mean of g from g
    g = g - np.mean(g)
    #performing the cross-correlation
    out = cross_correlation(f, g)
    
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    
    #normalising f and g using np functions
    f = (f - np.mean(f))/np.var(f)
    g = (g - np.mean(g))/np.var(g)
    #performing the cross-correlation
    out = cross_correlation(f, g)
    
    ### END YOUR CODE

    return out
