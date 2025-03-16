%%cython
import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

def ccfh_transform(
    list X_hasher_input,
    int n_features,
    np.ndarray[ITYPE_t, ndim=1] hashes_1,
    np.ndarray[ITYPE_t, ndim=1] hashes_2,
    np.ndarray[ITYPE_t, ndim=1] hashes_H,
    np.ndarray[ITYPE_t, ndim=1] signs,   
    np.ndarray[DTYPE_t, ndim=1] q,       
    np.ndarray[DTYPE_t, ndim=1] v,
    list feature_names      
):
    cdef:
        int i, j, h1, h2, H, sign
        int n_samples = len(X_hasher_input)
        float w

        np.ndarray[DTYPE_t, ndim=2] X_hashed = np.zeros((n_samples, n_features), dtype=np.float64)
        DTYPE_t[:, :] X_hashed_view = X_hashed
        ITYPE_t[:] hashes_1_view = hashes_1
        ITYPE_t[:] hashes_2_view = hashes_2
        ITYPE_t[:] hashes_H_view = hashes_H
        ITYPE_t[:] signs_view = signs
        DTYPE_t[:] q_view = q
        DTYPE_t[:] v_view = v
        
        list row
        tuple feature_value
        object feature
        int feat_idx
        DTYPE_t value

        int n_q = len(q)
        int n_v = len(v)
        
        dict feature_map = {name: idx for idx, name in enumerate(feature_names)}
        
    for i in range(n_samples):
        row = X_hasher_input[i]
        
        for j in range(len(row)):
            feature_value = row[j]
            feature = feature_value[0]
            value = feature_value[1]
            feat_idx = feature_map[feature]

            h1 = hashes_1_view[feat_idx] % n_v
            h2 = hashes_2_view[feat_idx] % n_v
            H = hashes_H_view[feat_idx] % n_q
            sign = signs_view[feat_idx]
                        
            X_hashed_view[i, h1] += q_view[H] * value * sign
            X_hashed_view[i, h2] += (1 - q_view[H]) * value * sign
    
    return X_hashed

def ccfh_train(
    list X_train,
    list y_train,
    int n_features,
    float learning_rate,
    int n_iter,
    np.ndarray[ITYPE_t, ndim=1] hashes_1,
    np.ndarray[ITYPE_t, ndim=1] hashes_2,
    np.ndarray[ITYPE_t, ndim=1] hashes_H,
    np.ndarray[ITYPE_t, ndim=1] signs,   
    np.ndarray[DTYPE_t, ndim=1] q,       
    np.ndarray[DTYPE_t, ndim=1] v,
    list feature_names,
    int batch_size=256
):
    cdef:
        int i, j, h1, h2, H, sign
        int n_samples = len(X_train)
        float w, y_pred, error, dL_dw, epoch_loss

        ITYPE_t[:] hashes_1_view = hashes_1
        ITYPE_t[:] hashes_2_view = hashes_2
        ITYPE_t[:] hashes_H_view = hashes_H
        ITYPE_t[:] signs_view = signs
        
        list row
        tuple feature_value
        object feature
        int feat_idx
        DTYPE_t value

        int n_q = len(q)
        int n_v = len(v)

        dict feature_map = {name: idx for idx, name in enumerate(feature_names)}
    
    np.random.seed(0)
    
    for epoch in range(n_iter):
        epoch_loss = 0.0
        
        # Shuffle the data at the beginning of each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            grad_v_h1 = np.zeros_like(v)
            grad_v_h2 = np.zeros_like(v)
            grad_q_H = np.zeros_like(q)
            
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            for i in batch_indices:
                row = X_train[i]

                for j in range(len(row)):
                    feature_value = row[j]
                    feature = feature_value[0]
                    value = feature_value[1]
                    feat_idx = feature_map[feature]

                    h1 = hashes_1_view[feat_idx] % n_v
                    h2 = hashes_2_view[feat_idx] % n_v
                    H = hashes_H_view[feat_idx] % n_q
                    sign = signs_view[feat_idx]
                    
                    w = (q[H] * v[h1] + (1 - q[H]) * v[h2]) * sign
                    y_pred = w * value
                    error = y_pred - y_train[i]
                    
                    dL_dw = error * value
                    grad_v_h1[h1] += dL_dw * q[H] * sign
                    grad_v_h2[h2] += dL_dw * (1 - q[H]) * sign
                    grad_q_H[H] += dL_dw * sign * (v[h1] - v[h2])

                    epoch_loss += 0.5 * error ** 2
            
            # Update parameters using the gradients from the current batch
            v -= learning_rate * grad_v_h1
            v -= learning_rate * grad_v_h2
            q = np.clip(q - learning_rate * grad_q_H, 0.0, 1.0)

        epoch_loss /= n_samples
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.3f}')
            
    return q, v