import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import numpy as np

# this is regular graph since it is an image
# for citation, I have to import cora datset in the form of graph
def grid(m, dtype=np.float32):
    """Return the embedding of a grid graph."""
    M = m**2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z # a matrix of M*2. 
# z is used to find the distance

def distance_scipy_spatial(z, k=8, metric='euclidean'): #k number of neighbours
    """Compute exact pairwise distances."""
    d = scipy.spatial.distance.pdist(z, metric) # sklearn is faster than scipy in this regard
    d = scipy.spatial.distance.squareform(d)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx
# why two types of distance?????????????????

def distance_sklearn_metrics(z, k=8, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = sklearn.metrics.pairwise.pairwise_distances(z, metric=metric, n_jobs=-2)#distance matrix
    #n_jobs: maximum number of concurrently running workers. 
    # If 1, no joblib parallelism is used at all, which is useful for debugging. If set to -1, all CPUs are used. 
    # For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. For example with n_jobs=-2, all CPUs but one are used.
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    # print("inside grph", idx.shape)
    d.sort()
    d = d[:, 1:k+1]
    # print("inside grph", d.shape)
    return d, idx

# def distance_sklearn_metrics(z, k=8, metric='cosine'):
#     """Compute exact pairwise distances."""
#     d = sklearn.metrics.pairwise.pairwise_distances(z, metric=metric, n_jobs=-2)#distance matrix
#     # k-NN graph.
#     idx = np.argsort(d)[:, 1:k+1]
#     d.sort()
#     d = d[:, 1:k+1]
#     return d, idx

# since .LSHForest() has been deprecated, I found another method implementing same task
# from sklearn.metrics.pairwise import pairwise_distances
# def distance_sk_cosine(z, k=4, metric='cosine'):
#     """Return an approximation of the k-nearest cosine distances."""
#     assert metric is 'cosine'
#     pairwise_distances.fit(z)
#     return dist, idx

# def distance_lshforest(z, k=4, metric='cosine'):
#     """Return an approximation of the k-nearest cosine distances."""
#     assert metric=='cosine'#is
#     lshf = sklearn.neighbors.LSHForest()
#     lshf.fit(z)
#     dist, idx = lshf.kneighbors(z, n_neighbors=k+1)
#     assert dist.min() < 1e-10
#     dist[dist < 0] = 0 # only positive cosine is accepted
#     return dist, idx
# # TODO: other ANNs s.a. NMSLIB, EFANNA, FLANN, Annoy, sklearn neighbors, PANN


#*****************************************    adjacency matrix **************************************
def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph. M*M symmetric, no self-loop,
    converts distance to similarity"""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2 # the last column of dist matrix M*k is the largest distance 
    dist = np.exp(-dist**2 / sigma2) # what formula it is?????

    # Weight matrix.
    I = np.arange(0, M).repeat(k) # repeat?
    J = idx.reshape(M*k) # flatten???
    V = dist.reshape(M*k)# row-major
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
    W.setdiag(0)# distance from itself is zero.
    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0                   # double check that W has even number of elements
    assert np.abs(W - W.T).mean() < 1e-10   # 
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def replace_random_edges(A, noise_level):   # what's the application or benefit???????????????
    """Replace randomly chosen edges by random edges.
     noise-level impact the number of edges that omitted or added"""
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)       # evenly distributed among non-zero elements

    indices = np.random.permutation(A.nnz//2)[:n]
    rows = np.random.randint(0, M, n) #np.random.randint(low, high=None, size=None, dtype=int)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n) #random.uniform(low=0.0, high=1.0, size=None)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = scipy.sparse.triu(A, format='coo')
    # scipy.sparse.triu(A, k=0, format=None) # Return the upper triangular portion of a matrix in sparse format
    assert A_coo.nnz == A.nnz // 2
    assert A_coo.nnz >= n
    A = A.tolil()

    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]

        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()# during the above procedure some explicit zero may occur. they will be removed to be efficient
    ############
    # if A is orginally symmetric then, the resulted noisy matrix would be symmetric too. 
    # Since the removed and added edges are performed symmetrically. 
    # Noise level impact the number of elements that are changed.
    return A


def laplacian(W, normalized=True): # A has been used for finding Laplacian
    """Return the Laplacian of the weigth matrix."""
    # Degree matrix.
    d = W.sum(axis=0) #gives a row, addes rows together
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)           # A is a method for making it squeezable
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype)) # adds a very small number close to zero
                                              # finds the nearest number to 0
        d = 1 / np.sqrt(d)  # ignoring np.space the result is as expected in math. spacing makes it different
                            # D^(-1/2) has eeen found here
        D = scipy.sparse.diags(d.A.squeeze(), 0)                #  A?????????????????????????????????????
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D     # D here is D^(-1/2)

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def lmax(L, normalized=True): #leading eigenvalue 
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:       #eigsh: s:sparse h:Hermitian (symmetric)
        return scipy.sparse.linalg.eigsh(L, k=1, which='LM', 
        return_eigenvectors=False)[0] #leading eigenvalue, ‘LM’ : Largest (in magnitude) eigenvalues.


def fourier(L, algo='eigh', k=1): #k=1, gives the leading lambda
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U): 
        idx = lamb.argsort()
        return lamb[idx], U[:, idx] # ascending eigenvalues and corresponding eigenvectors.

    if algo == 'eig': # finds exact ALL lambda 
        lamb, U = np.linalg.eig(L.toarray()) # L is sparse. no k argument since comuptes ALL lambda
        lamb, U = sort(lamb, U)
    elif algo == 'eigh': # assumes matrix is symmetric and uses faster algorithm Hermittian==symmetric
        lamb, U = np.linalg.eigh(L.toarray()) # it is defaul so no need to call sort function anymore.
        # eig and eigh for dense, h sands for Hermittian or real semi-definite.
        # eigs and eigsh for sparse, h sands for Hermittian or real semi-definite.
    elif algo == 'eigs': # finds approximate some lambda
        #scipy.sparse.linalg.eig: related to the Lanczos algorithm, which works very well with sparse matrices.
        #Lanczos algorithm has the property that it works better for "large eigenvalues" (leading one)
        # this simple algorithm does not work very well for computing very many of the eigenvectors because
        #  any round-off error will tend to introduce slight components of the more significant eigenvectors
        #  back into the computation, degrading the accuracy of the computation
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM') #small magnitude????????????????????
        lamb, U = sort(lamb, U)
    elif algo == 'eigsh': # finds approximate some lambda
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM') # cannot find ALL lambda and takes k argument.

    return lamb, U              # eigenvalues and corresponding eigenvector
    #eigenvalues in ascending order. All positive


def plot_spectrum(L, algo='eig', ymin = 0):
    """Plot the spectrum of a list of multi-scale Laplacians L."""
    # Algo is eig to be sure to get ALL eigenvalues.
    plt.figure(figsize=(17, 5))
    for i, lap in enumerate(L):
        lamb, U = fourier(lap, algo)
        step = 2**i
        x = range(step//2, L[0].shape[0], step) # // is floor division
        lb = 'L_{} spectrum in [{:1.2e}, {:1.2e}]'.format(i, lamb[0], lamb[-1])
        plt.plot(x, lamb, '.', label=lb)
    plt.legend(loc='best')
    plt.xlim(0, L[0].shape[0])
    plt.ylim(ymin=ymin)
    plt.ylabel('Value')
    plt.xlabel('Eigenvalue ID')


def lanczos(L, X, K):
    """
    Given the graph Laplacian and a data matrix, return a data matrix which can
    be multiplied by the filter coefficients to filter X using the Lanczos
    polynomial approximation.
    """
    M, N = X.shape
    assert L.dtype == X.dtype

    def basis(L, X, K):
        """
        Lanczos algorithm which computes the orthogonal matrix V and the
        tri-diagonal matrix H.
        """
        a = np.empty((K, N), L.dtype)
        b = np.zeros((K, N), L.dtype) # difference between zero and empty matrix
        V = np.empty((K, M, N), L.dtype)
        V[0, ...] = X / np.linalg.norm(X, axis=0) #????????
        for k in range(K-1):
            W = L.dot(V[k, ...])
            a[k, :] = np.sum(W * V[k, ...], axis=0)
            W = W - a[k, :] * V[k, ...] - (
                    b[k, :] * V[k-1, ...] if k > 0 else 0)
            b[k+1, :] = np.linalg.norm(W, axis=0)
            V[k+1, ...] = W / b[k+1, :]
        a[K-1, :] = np.sum(L.dot(V[K-1, ...]) * V[K-1, ...], axis=0)
        return V, a, b

    def diag_H(a, b, K):
        """Diagonalize the tri-diagonal H matrix."""
        H = np.zeros((K*K, N), a.dtype)
        H[:K**2:K+1, :] = a
        H[1:(K-1)*K:K+1, :] = b[1:, :]
        H.shape = (K, K, N)
        Q = np.linalg.eigh(H.T, UPLO='L')[1]
        Q = np.swapaxes(Q, 1, 2).T
        return Q

    V, a, b = basis(L, X, K)
    Q = diag_H(a, b, K)
    Xt = np.empty((K, M, N), L.dtype)
    for n in range(N):
        Xt[..., n] = Q[..., n].T.dot(V[..., n])
    Xt *= Q[0, :, np.newaxis, :]
    Xt *= np.linalg.norm(X, axis=0)
    return Xt  # Q[0, ...]


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def chebyshev(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN)."""
    M, N = X.shape
    assert L.dtype == X.dtype

    # L = rescale_L(L, lmax)
    # Xt = T @ X: MxM @ MxN.
    Xt = np.empty((K, M, N), L.dtype)
    # Xt_0 = T_0 X = I X = X.
    Xt[0, ...] = X
    # Xt_1 = T_1 X = L X.
    if K > 1:
        Xt[1, ...] = L.dot(X)
    # Xt_k = 2 L Xt_k-1 - Xt_k-2.
    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt

# def cora():
