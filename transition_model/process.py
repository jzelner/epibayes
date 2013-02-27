
def binaryOfIndexed(X, ndim=0):
    ndim = max(np.max(np.unique(X)), ndim)
    Z = np.zeros((ndim,) + X.shape, dtype="int")
    print X
    for i in range(ndim):
        Z[i][(X==i)] = 1
    return Z
    
