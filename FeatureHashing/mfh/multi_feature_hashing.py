import numpy as np
import mmh3
from scipy.sparse import csr_matrix
from .multi_feature_hashing_cy import multi_feature_hashing_cy

class MultiFeatureHasher():
    def __init__(self, feature_names, n_features, n_hashes = 1, seed = 0):
        """
        Multi-feature hashing class that hashes multiple features into a fixed number of features.
        :param feature_names: List of feature names
        :param n_features: Number of features to hash to
        :param n_hashes: Number of hashes to use
        :param seed: Base seed for hashing
        """
        self.feature_names = list(feature_names)
        self.n_features = n_features
        self.signs = self._hash_signs(seed)
        self.hashes = np.array([self._hash_features(seed * i) for i in range(1, n_hashes + 1)])

    def _hash_signs(self, seed):
        """
        Generate random signs for hashing
        :param seed: Seed for random number generator
        :return: Array of random signs (-1 or 1)
        """
        np.random.seed(seed)
        return np.array(np.random.choice([-1, 1], size=len(self.feature_names)), dtype=np.int64)
    
    def _hash_features(self, seed):
        """
        Hash feature names
        :param seed: Seed for hashing
        :return: Array of hashed feature names
        """
        return np.array([mmh3.hash(f.encode('utf-8'), seed, signed=False) for f in self.feature_names], dtype=np.int64)
    
    def hash_X(self, X):
        """
        Hash multiple features into a fixed number of features
        :param X: List of feature names and values
        :return: Hashed feature matrix
        """
        return csr_matrix(
                multi_feature_hashing_cy(
                X,
                self.n_features,
                self.hashes,
                self.signs,
                self.feature_names
            )
        )
    
def dense_X(bow, feature_names):
    """
    Prepare feature matrix for hashing
    :param bow: Bag of words matrix
    :return: List of feature names and values
    """
    X = []
    
    for i in range(bow.shape[0]):
        # Get non-zero indices and values for the row i
        row_indices = bow[i].indices
        row_values = bow[i].data

        # Map feature names to their values
        row_vals = [(feature_names[j], row_values[idx]) for idx, j in enumerate(row_indices)]
        X.append(row_vals)

    return X

def apply_mfh(mfh: MultiFeatureHasher, X):
    """
    Apply multi-feature hashing to a feature matrix
    :param mfh: Multi-feature hasher
    :param X: List of feature names and values
    :return: Hashed feature matrix
    """
    return mfh.hash_X(X)
