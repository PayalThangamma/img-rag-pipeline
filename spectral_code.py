import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.sparse import diags # Needed for constructing D

def compute_spectral_embedding(image_embeddings, text_embeddings, n_components):
    """
    Compute spectral embedding based on the paper's bipartite graph approach.

    Args:
        image_embeddings (np.ndarray): The (normalized) image embedding matrix X.
        text_embeddings (np.ndarray): The (normalized) text embedding matrix Y.
        n_components (int): Number of eigenvectors to return.

    Returns:
        np.ndarray: The combined spectral embeddings F, where rows correspond to
                    the input image and text embeddings.
    """
    n_images = image_embeddings.shape[0]
    n_texts = text_embeddings.shape[0]

    # Step 1: Compute W = X Y^T (cross-modal similarity)
    W = cosine_similarity(image_embeddings, text_embeddings)

    # cosine_similarity can be negative
    W[W < 0] = 0

    # Step 2: Form the matrix A = [[0, W], [W^T, 0]]

    zero_img_img = np.zeros((n_images, n_images))
    zero_text_text = np.zeros((n_texts, n_texts))

    A_top = np.hstack((zero_img_img, W))
    A_bottom = np.hstack((W.T, zero_text_text))
    A = np.vstack((A_top, A_bottom))

    # Add a small value to the diagonal to prevent isolated nodes
    # np.fill_diagonal(A, 1.0 + 1e-9)
    np.fill_diagonal(A, 1e-12)

    # Compute degree matrix D
    # D is a diagonal matrix where D_ii = sum of row i of A
    degrees = A.sum(axis=1)
    D = diags(degrees, 0).toarray()

    # Compute Laplacian L = D - A
    L = D - A

    # Step 3: Compute Lrw = D^-1 L (Random Walk Normalized Laplacian)
    # Handle cases where degrees might be zero (though fill_diagonal should largely prevent this)
    # Inverse of D: 1/degrees, set inf to 0 for numerical stability
    D_inv_diag = np.where(degrees > 0, 1.0 / degrees, 0.0)
    D_inv = diags(D_inv_diag, 0).toarray()

    Lrw = D_inv @ L # Matrix multiplication D_inv * L

    # Compute eigenvalues and eigenvectors of Lrw
    eigvals, eigvecs = eigh(Lrw)

    # Sort eigenvectors by eigenvalue (ascending)
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Return eigenvectors corresponding to the k smallest nonzero eigenvalues.
    num_available_eigvecs = eigvecs.shape[1] - 1 # Exclude the first trivial one
    k_to_return = min(n_components, num_available_eigvecs)

    return eigvecs[:, 1 : 1 + k_to_return]