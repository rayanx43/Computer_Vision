import numpy as np


def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """

    
    ### YOUR CODE HERE
    
    #dot product of 2 vectors
    out = np.dot(a, b)
    
    ### END YOUR CODE
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    
    ### YOUR CODE HERE
    
    #calculating dot product between 2 vectors(same as a multiplication in this case)
    x = dot_product(a, b)
    #calculating the transpose
    A = np.transpose(a)
    #multiplying 2 matrices
    y = np.matmul(M, A)
    #getting the final output
    out = x * y
    
    ### END YOUR CODE

    return out


def svd(M):
    """Implement Singular Value Decomposition.

    (optional): Look up `np.linalg` library online for a list of
    helper functions that you might find useful.

    Args:
        M: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    
    ### YOUR CODE HERE
    
    #computing svd using the function np.linalg.svd
    u, s, v = np.linalg.svd(M, full_matrices = True)
    
    ### END YOUR CODE

    return u, s, v


def get_singular_values(M, k):
    """Return top n singular values of matrix.

    (optional): Use the `svd(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, n).
        k: number of singular values to output.

    Returns:
        singular_values: array of shape (k)
    """
    
    ### YOUR CODE HERE
    
    #declaring an array to hold the singular values 
    singular_values = []
    
    #getting the specific array (the second one) from the svd function
    s = svd(M)[1]
    
    #getting all the singular values in the array s upto k-1
    for i in range(0,k):
        singular_values.append(s[i])
    
    ### END YOUR CODE
    return singular_values


def eigen_decomp(M):
    """Implement eigenvalue decomposition.
    
    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    
    ### YOUR CODE HERE
    
    #returns an eigenvalue and an eigenvector
    w, v = np.linalg.eig(M)
    
    ### END YOUR CODE
    return w, v


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    eigenvalues = []
    eigenvectors = []
    #array to place in sorted indexes
    order = []
    
    ### YOUR CODE HERE
    
    #returns an array of indexes such that the indexes give the descending order of the eigenvalues
    order = np.argsort(-abs(eigen_decomp(M)[0]))
    
    #storing the eigenvalues and eigenvectors into the arrays utilizing the sorted indexes
    for i in range(0, k):
        eigenvalues.append(eigen_decomp(M)[0][order[i]])
        eigenvectors.append(eigen_decomp(M)[0][order[i]])
    
    ### END YOUR CODE
    return eigenvalues, eigenvectors
