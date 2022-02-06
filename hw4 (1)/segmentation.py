import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        
        """
        In the first for loop, we iterate through all the points and
        create a new array (assignments) which contains the indices
        (0 - 3) which is an assignment of the closest cluster center
        (it is 0-3 since each center has an index in the center array)
        to each point. After this for loop, the centers are copied
        to a temporary array since these centers will change in the
        next for loop.
        """
        for i in range(N):
            #Equation to compute assignment of each point to a cluster center
            assignments[i] = np.argmin(np.sum((features[i] - centers)**2, axis=1))
        tmp = centers.copy()
        
        """
        For this next for loop, we compute the mean of the points
        that are in one cluster and assign that new mean to be
        the new cluster center of that specific cluster. This is
        done for all the clusters.
        """
        for j in range(k):
            #Equation to compute the mean of a specific cluster
            #and assigning it as the new cluster center
            centers[j] = np.mean(features[assignments == j], axis=0)
        
        """
        After each iteration of 'num_iters', this condition checks
        if the old cluster centers are almost equal to the newly
        calculated cluster centers. If they are almost equal, the
        main for loop (using n as the index) breaks and k-means
        clustering is complete.
        """
        if np.allclose(tmp, centers):
            break
        
        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        
        #repeating the features array k times consecutively
        f_tmp = np.tile(features, (k, 1))
        """
        The following line repeats each element in the centers
        array N times followed by the next element N times and so
        forth.
        """
        c_tmp = np.repeat(centers, N, axis=0)
        """
        Using our new arrays, we once again compute the closest
        cluster centers making use of array reshaping to compute
        the result faster (this is possible due to the
        repetitions which allows us to quickly compute these centers
        in one go for all the elements).
        This is much faster than iterating through each
        feature individually and assigning a cluster center to it.
        """
        assignments = np.argmin(np.sum((f_tmp - c_tmp)**2, axis=1).reshape(k, N), axis=0)
        #copying centers as in the function above
        tmp = centers.copy()
        #~refer to explanantion for same loop in previous function~#
        for j in range(k):
            centers[j] = np.mean(features[assignments == j], axis=0)
        #~refer to explanantion for same step in previous function~#
        if np.allclose(tmp, centers):
            break
        
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        ### YOUR CODE HERE
        
        """
        Using the pdist function to compute the euclidean distances
        between all the pairs of clusters
        """
        distances = pdist(centers)
        """
        Using the squareform function to convert a vector-form distance vector
        to a square-form distance matrix
        """
        matrixDistances = squareform(distances)
        """
        Using np.where to multiply all values except 0.0 by 1e10. 
        """
        matrixDistances = np.where(matrixDistances != 0.0, matrixDistances, 1e10)
        """
        Minimizing the distances between clusters to find which ones are
        the closest
        """
        minValue = np.argmin(matrixDistances)
        min_i = minValue // n_clusters
        min_j = minValue - min_i * n_clusters
        if min_j < min_i:
            min_i, min_j = min_j, min_i
        for i in range(N):
            if assignments[i] == min_j:
                assignments[i] = min_i
        for i in range(N):
            if assignments[i] > min_j:
                assignments[i] -= 1
        centers = np.delete(centers, min_j, axis = 0)
        centers[min_i] = np.mean(features[assignments == min_i], axis = 0)
        n_clusters -= 1
        
        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    
    """
    Simply using img.reshape to reshape the 3D array
    into a 2D array (our new feature vector)
    Note that the pixel values at each point remain
    the SAME
    """
    features = img.reshape(H*W, C)
    
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    
    """
    First, we use np.dstack to concatenate the reshaped grid
    containing the heights and the widths (i.e. locations)
    """
    locations = np.dstack(np.mgrid[0 : H, 0 : W]).reshape((H * W, 2))
    """
    Next, we assign the colors to the features array and
    assign the locations next
    """
    features[:, 0 : C] = color.reshape((H * W, C))
    features[:, C : C + 2] = locations
    """
    Finally, we normalize the features array forcing each
    feature to have zero mean and unit variance
    """
    features = (features - np.mean(features, axis = 0)) / np.std(features, axis = 0)
    
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))
    features = img.reshape(H*W, C)
    # normalize the features array forcing each feature to have zero mean and unit variance
    features = (features - np.mean(features, axis = 0)) / np.std(features, axis = 0)
    
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    
    """
    Here, we simply use the np.mean function to check the
    similarity between the individual pixels provided. If
    the pixels in mask_gt (ground truth  segmentation)  and
    mask are exactly the same, this function would yield 1.0.
    """
    accuracy = np.mean(mask_gt == mask)
    
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
