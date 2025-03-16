import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

def mfh_transform(
    list X_hasher_input,
    int n_features,
    np.ndarray[ITYPE_t, ndim=2] hashes,     # Shape: (n_hashes, n_features)
    np.ndarray[ITYPE_t, ndim=1] signs,      # Shape: (n_features,)
    list feature_names                      
):
    cdef:
        int i, j, k
        int n_samples = len(X_hasher_input)
        int n_hashes = hashes.shape[0]
        float scaling_factor = (1 / np.sqrt(n_hashes))
        np.ndarray[DTYPE_t, ndim=2] X_hashed = np.zeros((n_samples, n_features), dtype=np.float64)
        DTYPE_t[:, :] X_hashed_view = X_hashed
        ITYPE_t[:, :] hashes_view = hashes
        ITYPE_t[:] signs_view = signs
        
        list row
        tuple feature_value
        object feature
        int feat_idx
        DTYPE_t value
        ITYPE_t hash_val
        
        dict feature_map = {name: idx for idx, name in enumerate(feature_names)}
    
    cdef int n_features_mask = n_features - 1
    
    for i in range(n_samples):
        row = X_hasher_input[i]
        
        for j in range(len(row)):
            feature_value = row[j]
            feature = feature_value[0]
            feat_idx = feature_map[feature]

            value = feature_value[1] * signs_view[feat_idx] * scaling_factor
            
            for k in range(n_hashes):
                hash_val = hashes_view[k, feat_idx] & n_features_mask
                X_hashed_view[i, hash_val] += value
    
    return X_hashed