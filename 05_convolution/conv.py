"""Stage 5: convolution and max-pooling, implemented from scratch.

Single-channel (grayscale) convolution: kernel K has shape
(n_filters, k, k), input X has shape (batch, H, W). Forward pass is
cross-correlation (no kernel flip), matching the standard "conv
layer" convention everywhere in practice:

    Z[b,i,j,f] = sum_{u,v} K[f,u,v] * X[b,i+u,j+v] + b[f]

Uses numpy's sliding_window_view to extract every k x k patch at
once (a legitimate numpy tool, not an autodiff shortcut - the
underlying math is exactly the derivation in the conversation, just
reshaped so numpy's BLAS backend does the arithmetic instead of a
slow Python quadruple-nested loop).

Backward pass implements the three gradients derived by hand:

    dL/dK[f,u,v] = sum_{b,i,j} G[b,i,j,f] * X[b,i+u,j+v]
    dL/db[f]     = sum_{b,i,j} G[b,i,j,f]
    dL/dX[b,p,q] = sum_{f,u,v} G[b,p-u,q-v,f] * K[f,u,v]

where G = dL/dZ is the upstream gradient. The dX formula is a
genuine convolution (flipped kernel) even though the forward pass is
cross-correlation - see notes.md for why.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def extract_patches(X, k):
    """All k x k patches of X, shape (batch, H, W) ->
    (batch, out_h, out_w, k, k)."""
    return sliding_window_view(X, (k, k), axis=(1, 2))


def conv_forward(X, K, b):
    """X: (batch,H,W). K: (n_filters,k,k). b: (n_filters,).
    Returns (patches, Z) where Z has shape
    (batch, out_h, out_w, n_filters) - patches are cached for use
    in conv_backward, same explicit-intermediate style as earlier
    stages' forward/backward pairs.
    """
    k = K.shape[1]
    patches = extract_patches(X, k)
    z = np.einsum("bijhw,fhw->bijf", patches, K) + b
    return patches, z


def conv_backward(patches, g, K, x_shape):
    """g = dL/dZ, shape (batch, out_h, out_w, n_filters).
    Returns (grad_K, grad_b, grad_X).
    """
    k = K.shape[1]
    grad_k = np.einsum("bijf,bijhw->fhw", g, patches) / g.shape[0]
    grad_b = g.sum(axis=(0, 1, 2)) / g.shape[0]

    pad = k - 1
    g_padded = np.pad(g, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    k_flipped = K[:, ::-1, ::-1]
    g_patches = extract_patches_multi(g_padded, k)
    grad_x = np.einsum("bijfhw,fhw->bij", g_patches, k_flipped)
    grad_x = grad_x.reshape(x_shape) / g.shape[0]

    return grad_k, grad_b, grad_x


def extract_patches_multi(g, k):
    """Like extract_patches, but for an array with a trailing
    filter axis: (batch, H, W, n_filters) ->
    (batch, out_h, out_w, k, k, n_filters).
    """
    return sliding_window_view(g, (k, k), axis=(1, 2))


def relu(z):
    return np.maximum(0, z)


def relu_backward(g, z):
    return g * (z > 0)


def maxpool_forward(a, size=2):
    """a: (batch, H, W, n_filters), H and W divisible by size.
    Returns (pooled, mask_info) where mask_info is everything
    maxpool_backward needs to route gradients back.
    """
    batch, h, w, f = a.shape
    hp, wp = h // size, w // size
    a_reshaped = a.reshape(batch, hp, size, wp, size, f)
    pooled = a_reshaped.max(axis=(2, 4))
    return pooled, (a_reshaped, pooled)


def maxpool_backward(g_pooled, mask_info):
    """g_pooled: (batch, hp, wp, f). Routes each pooled position's
    gradient back to the position(s) that achieved the max in its
    window (split evenly across ties, which correctly preserves the
    gradient sum in the rare case of an exact tie).
    """
    a_reshaped, pooled = mask_info
    batch, hp, size1, wp, size2, f = a_reshaped.shape
    mask = a_reshaped == pooled.reshape(batch, hp, 1, wp, 1, f)
    counts = mask.sum(axis=(2, 4), keepdims=True)
    g_reshaped = mask * (g_pooled.reshape(batch, hp, 1, wp, 1, f) / counts)
    return g_reshaped.reshape(batch, hp * size1, wp * size2, f)
