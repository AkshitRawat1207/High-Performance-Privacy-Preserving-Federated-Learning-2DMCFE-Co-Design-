"""
gradient_encoder.py
===================
Bridge between floating-point CNN gradients and Z_p integers required
by 2DMCFE.

The problem
-----------
2DMCFE works over Z_p (integers mod a large prime p).
CNN gradients are IEEE 754 floats in roughly [-1, 1].

We need a lossless round-trip:
    float gradient  →  Z_p integer  →  (encrypt / aggregate) →  Z_p sum  →  float sum

Encoding scheme
---------------
We use fixed-point quantisation with a scale factor Q:

    encode(x)  =  round(x * Q)  mod p       ∈ Z_p
    decode(z)  =  centre(z) / Q             ∈ ℝ

where centre(z) maps Z_p back to the signed range [-p/2, p/2]:

    centre(z) = z if z < p/2 else z - p

This works because:
    - Individual gradients are small floats (say |x| < 10).
    - After aggregating M≤20 gradients the sum is still < 200.
    - With Q=10^6 we keep 6 decimal places of precision.
    - The sum fits comfortably in Z_p since Q·M·max_grad ≪ p.

Overflow check
--------------
p ≈ 2^61 ≈ 2.3×10^18
Max encoded value per coordinate: Q × |max_grad| × M ≈ 10^6 × 10 × 20 = 2×10^8
Safety factor: 2×10^8 / 2.3×10^18 ≈ 10^-10  →  no overflow possible.

Compression (optional)
-----------------------
Real FL systems compress gradients before encryption to reduce
communication cost. We implement top-k sparsification: keep only
the k coordinates with the largest absolute gradient values and
zero the rest. This directly reduces n (the encrypted vector length).

The paper does not use compression in the base 2DMCFE scheme but it
is a natural extension.
"""

import numpy as np
from typing import Tuple, Optional
from crypto_utils import P, mod

# Default quantisation scale
DEFAULT_Q = 10 ** 8


# ---------------------------------------------------------------------------
# Encode / Decode
# ---------------------------------------------------------------------------

"""
Bridge between floating-point CNN gradients and Z_p integers required by 2DMCFE.
Fully vectorized for speed, using stable multi-precision objects to prevent arithmetic overflow.
"""

def encode_gradient(
    grad: np.ndarray,
    Q: int = DEFAULT_Q,
    p: int = P,
) -> np.ndarray:
    """
    Convert a float gradient vector into a Z_p integer vector without loops.
    """
    # Clip extreme values to avoid catastrophic quantisation error
    grad_clipped = np.clip(grad, -1e4, 1e4)
    
    # Scale and round all gradients at once using NumPy vectorization
    scaled = np.round(grad_clipped * Q).astype(object)
    
    # Reduce mod p (object type handles negative integers automatically and correctly)
    encoded = scaled % p
    return encoded

def decode_gradient(
    z_vec: np.ndarray,
    Q: int = DEFAULT_Q,
    p: int = P,
) -> np.ndarray:
    """
    Convert a Z_p integer vector back to floats with fast vectorized centering.
    """
    half_p = p // 2
    
    # Ensure vector elements are cleanly reduced mod p
    vi = z_vec % p
    
    # Vectorized centering: replaces the slow element-wise python loop
    vi_centered = np.where(vi > half_p, vi - p, vi)
    
    # Cast back to float64 for neural weights
    return vi_centered.astype(np.float64) / Q

def decode_sum(
    z: int,
    Q: int = DEFAULT_Q,
    p: int = P,
) -> float:
    """Decode a scalar Z_p sum."""
    half_p = p // 2
    zi = int(z) % p
    if zi > half_p:
        zi -= p
    return zi / Q

def topk_sparsify(
    grad: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k sparsification: keep k largest-magnitude coordinates."""
    top_idx = np.argsort(np.abs(grad))[-k:]
    sparse = np.zeros_like(grad)
    sparse[top_idx] = grad[top_idx]
    return sparse, top_idx

def pack_gradient(
    grad: np.ndarray,
    compress: bool = False,
    k: Optional[int] = None,
    Q: int = DEFAULT_Q,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if compress and k is not None:
        sparse, top_idx = topk_sparsify(grad, k)
        grad_to_encode = sparse[top_idx]
        return encode_gradient(grad_to_encode, Q=Q), top_idx
    else:
        return encode_gradient(grad, Q=Q), None

def unpack_gradient(
    z_vec: np.ndarray,
    total_len: int,
    top_idx: Optional[np.ndarray] = None,
    Q: int = DEFAULT_Q,
) -> np.ndarray:
    decoded = decode_gradient(z_vec, Q=Q)
    if top_idx is not None:
        full = np.zeros(total_len, dtype=np.float64)
        full[top_idx] = decoded
        return full
    return decoded


# ---------------------------------------------------------------------------
# Sanity test
# ---------------------------------------------------------------------------

def _selftest():
    print("=== Gradient encoder self-test ===")
    rng = np.random.default_rng(0)

    # Typical CNN gradient values
    grad = rng.normal(0, 0.1, size=20)
    print(f"  Original grad[:5]  : {grad[:5]}")

    encoded = encode_gradient(grad)
    print(f"  Encoded [:5]       : {[int(v) for v in encoded[:5]]}")

    decoded = decode_gradient(encoded)
    print(f"  Decoded [:5]       : {decoded[:5]}")

    max_err = np.max(np.abs(grad - decoded))
    print(f"  Max encoding error : {max_err:.2e}  (should be < 1e-6)")
    assert max_err < 1e-6, "Encoding round-trip error too large"

    # Test aggregation: sum of 3 gradients
    grads = [rng.normal(0, 0.1, size=20) for _ in range(3)]
    encoded_sum = np.array(
        [mod(sum(int(encode_gradient(g)[i]) for g in grads)) for i in range(20)],
        dtype=object
    )
    true_sum = sum(grads)
    decoded_sum = decode_gradient(encoded_sum)
    max_err2 = np.max(np.abs(true_sum - decoded_sum))
    print(f"  Aggregated sum error: {max_err2:.2e}  (should be < 1e-6)")
    assert max_err2 < 1e-6

    # Test negative gradient handling
    neg_grad = np.array([-0.5, -0.1, 0.3])
    enc = encode_gradient(neg_grad)
    dec = decode_gradient(enc)
    assert np.allclose(neg_grad, dec, atol=1e-6), f"Negative grad failed: {dec}"
    print("  Negative gradient round-trip: ✓")
    print("  PASS ✓\n")


if __name__ == "__main__":
    _selftest()
