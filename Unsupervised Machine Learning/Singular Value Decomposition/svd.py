import numpy as np

class SVD:
    """
    Reduced Singular Value Decomposition (SVD) via eigen-decomposition of X^T X.

    Attributes
    ----------
    U : ndarray of shape (m, r)
        Left singular vectors.
    S : ndarray of shape (r,)
        Singular values in descending order.
    Vt : ndarray of shape (r, n)
        Transpose of matrix of right singular vectors.

    Methods
    -------
    fit(X)
        Compute the reduced SVD factors for input data X.
    reconstruct()
        Reconstruct the original matrix from U, S, and Vt.
    """

    def __init__(self):
        self.U = None
        self.S = None
        self.Vt = None

    def fit(self, X):
        """
        Compute the reduced SVD of X.

        Parameters
        ----------
        X : array-like of shape (m, n)
            Input data matrix.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        m, n = X.shape
        r = min(m, n)

        # Compute covariance matrix for V
        C = X.T @ X  # shape (n, n)
        eigvals, eigvecs = np.linalg.eigh(C)

        # Sort eigenvalues/vectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take top-r singular values and vectors
        singular_vals = np.sqrt(np.clip(eigvals[:r], 0, None))
        V = eigvecs[:, :r]

        # Compute U
        U = (X @ V) / singular_vals[np.newaxis, :]

        # Store decomposition
        self.U = U
        self.S = singular_vals
        self.Vt = V.T

        return self

    def reconstruct(self):
        """
        Reconstruct the original matrix from the decomposition.

        Returns
        -------
        X_hat : ndarray of shape (m, n)
            Reconstructed matrix U @ diag(S) @ Vt.
        """
        return self.U @ np.diag(self.S) @ self.Vt