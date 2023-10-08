import numpy as np

def read_syntetic(eta=0.06, gamma_0=0.1, gamma_1=0.2, train_size=4000, test_size=4000):
    """
    eta: P(A=1)
    gamma_0: P(Y=1|A=0)
    gamma_1: P(Y=1|A)
    """
    size = train_size + test_size
    A = np.random.choice([0,1], size=size, replace=True, p=[1-eta, eta]) # generates the A values
    Y_0 = np.random.choice([0,1], size=size, replace=True, p=[1-gamma_0, gamma_0]) # generates Y values given A=0
    Y_1 = np.random.choice([0,1], size=size, replace=True, p=[1-gamma_1, gamma_1]) # generates Y values given A=1
    Y = np.where(A, Y_1, Y_0) # choose Y_a for every sample 

    X_1 = np.where(A, -1, Y_0)
    X = np.stack([A, A] + [X_1 + 12*np.random.rand(*X_1.shape) for i in range(16)], axis=-1)
    return X[:train_size], Y[:train_size], A[:train_size], X[train_size:], Y[train_size:], A[train_size:]  