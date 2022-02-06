import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    
    #flipping kernal such that a fast convolution can be implemented
    kernel = np.flip(np.flip(kernel, 0), 1)
    
    #loop for the convolution of image and kernel
    #Was implemented in the previous assignment
    for i in range(Hi):
        for j in range(Wi):
            #using np.sum to perform the covolution
            out[i, j] = np.sum(padded[i: i+Hk, j: j+Wk] * kernel)
    
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    
    """ The following is a for loop implementing
    the equation for a Gaussian kernel of (size)*(size).
    
    Since the equation given was for a Gaussian kernel of (2k+1)*(2K+1),
    the parameters had to be modified slightly to fit our arguements.
    
    """
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-((i - size//2)**2 + (j - size//2)**2) / float(2*sigma**2))
    
    ### END YOUR CODE

    return kernel

def partial_x(image):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    
    """
    First, we define a kernel that approximates the partial derivatives
    by taking differences at one pixel intervals.
    Next, we convolve the image with our newly defined kernel to
    return the x-derivative image.
    """
    
    kernel = np.array([[0.5, 0, -0.5]])
    out = conv(image, kernel)
    
    ### END YOUR CODE

    return out

def partial_y(image):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    
    """
    Similar to what we did before, we define a kernel that approximates the
    partial derivatives by taking differences at one pixel intervals
    (this time in y-direction).
    Next, we convolve the image with our newly defined kernel to
    return the y-derivative image.
    """
    
    kernel = np.array([[0.5], [0], [-0.5]])
    out = conv(image, kernel)
    
    ### END YOUR CODE

    return out

def gradient(image):
    """ Returns gradient magnitude and direction of input img.

    Args:
        image: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(image.shape)
    theta = np.zeros(image.shape)

    ### YOUR CODE HERE
    
    """
    First, we get the x and y- derivative images of the
    original image using the functions implemented above.
    Next, we calculate G and theta (using np.sqrt and
    np.arctan2 functions) using the formulas
    presented to us.
    Note that theta was computed using the np.rad2deg
    function which made sure that that the condition
    'Direction of gradients should be in range 0 <= theta < 360'
    was satisfied and since 'np.arctan2' returns angles
    in radians
    """
    
    Gx = partial_x(image)
    Gy = partial_y(image)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    #theta = np.arctan2(Gy, Gx)
    theta = (np.rad2deg(np.arctan2(Gy, Gx)) + 180) % 360
    
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    
    """
    Since the gradient direction (theta) was already rounded to the nearest
    45 degrees, we can now loop through through G and theta comparing the
    edge strengths (preserving only the largest values and discarding
    the rest).
    Note that at this stage, the angle is measured clockwise using
    the positive x-axis as a reference.
    """
    
    for i in range(1, H-1):
        for j in range(1, W-1):
            #converting rounded theta to radians to work with sin and cos
            alpha = np.deg2rad(theta[i, j])
            #Condition to get the largest value when comparing edge strengths
            l_val = G[i-int(np.round(np.sin(alpha))), j-int(np.round(np.cos(alpha)))]
            g_val = G[i+int(np.round(np.sin(alpha))), j+int(np.round(np.cos(alpha)))]
            #condition to discard other values
            if not (G[i, j] >= l_val and G[i, j] >= g_val):
                out[i, j] = 0
            else:
                out[i, j] = G[i, j]
    
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array which represents strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    
    """
    Simply comparing the given image matrix with the given
    matrices for high and low thresholds to get the
    resulting matrices.
    """
    
    #For strong edges
    strong_edges = img > high
    #For weak edges
    weak_edges = (img <= high) & (img > low)
    
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak__edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    
    """
    First, we iterate through the whole strong edges matrix
    and get the neighbours of each and every element.
    Next, in the same for loop, we compare each element in
    weak edges with all the neighbours computed at that
    particular point from the strong edge matrix. If at
    least one value from the neighbors matrix at that
    point is equal to the value of the weak edge matrix
    at that particular point, the value is assigned to
    the new matrix 'edges'.
    """
    
    for i in range(1, H):
        for j in range(1, W):
            #Getting a matrix with neighbors of each element
            neighbors = get_neighbors(j, i, H, W)
            #condition for comparison as explained above
            if weak__edges[i, j] and np.any(edges[x, y] for x, y in neighbors):
                edges[i, j] = True
    
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    
    #First, the kernal was computed using 'gaussian kernel'
    kernel = gaussian_kernel(kernel_size, sigma)
    #Next, the image was smoothed
    smoothed = conv(img, kernel)
    #The gradient of the smoothed image was computed
    G, theta = gradient(smoothed)
    #nms converts the "blurred" edges into "sharp" edges
    nms = non_maximum_suppression(G, theta)
    #double thresholding was implemented to remove noise and color variations
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    #relevant weak edges were found and included in the final edge image
    edge = link_edges(strong_edges, weak_edges)
    
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, int(diag_len * 2.0 + 1))
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    
    ### YOUR CODE HERE
    
    """
    The function 'zip' is used to iterate thorough the non-zero
    values of the whole array.
    Next, for all the values of theta, the values of rho
    are calculated and the accumulator is incremented.
    
    """
    
    for i, j in zip(ys, xs):
        for idx in range(thetas.shape[0]):
            r = j * cos_t[idx] + i * sin_t[idx]
            accumulator[int(r + diag_len), idx] += 1
    
    ### END YOUR CODE

    return accumulator, rhos, thetas
