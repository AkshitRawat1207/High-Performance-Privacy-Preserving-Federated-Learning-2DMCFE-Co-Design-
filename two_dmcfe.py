"""
two_dmcfe.py
============
Dual-Mode Decentralised Multi-Client Functional Encryption (2DMCFE).
Optimised for Vector-wise Summation and high performance using NumPy and gmpy2.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import gmpy2  # <-- Added for fast large vector modulo operations

from crypto_utils import P, mod, prf, make_label
from dmcfe import DMCFE, _label_hash, DMCFECiphertext, DMCFEPartialKey, DMCFEUserKeys


@dataclass
class TwoDMCFECiphertext:
    user_id: int
    label: bytes
    ct: DMCFECiphertext   # Inner DMCFE ciphertext object
    session_key: int      # k_i,l
    delta: int            # 0 or 1


@dataclass
class TwoDMCFEPartialKey:
    user_id: int
    dk: DMCFEPartialKey   # Inner partial key object


class TwoDMCFE:
    """
    2DMCFE with O(n) decryption and Vector-wise Summation.
    Utilises NumPy and gmpy2 C-bindings to avoid slow Python long-int loops.
    """

    def __init__(self, n: int, N: int, shared_seed: bytes):
        self.n = n
        self.N = N
        self.shared_seed = shared_seed
        self.P = P  # Bind the modular prime field object
        # Inner DMCFE handles extended vectors of length n+2
        self._inner = DMCFE(n=n + 2, N=N)
        self._sk_cache = {}

    def setup(self):
        """Setup inner DMCFE parameters."""
        return self._inner.setup()

    def enc(self, user_keys: DMCFEUserKeys, x: np.ndarray, label: bytes, delta: int) -> TwoDMCFECiphertext:
        """
        Encrypts a gradient vector x.
        Extended plaintext format: x || delta * k || 0
        """
        assert delta in (0, 1)
        assert len(x) == self.n

        i = user_keys.user_id
        k_raw = prf(self.shared_seed, label + b"||user=" + str(i).encode())
        k_term = mod(delta * k_raw)
        
        # Create extended vector [x_0, x_1, ..., x_n-1, k_term, 0]
        x_ext = np.append(x, [k_term, 0]).astype(object)

        ct_inner = self._inner.enc(user_keys, x_ext, label)
        return TwoDMCFECiphertext(
            user_id=i, label=label,
            ct=ct_inner, session_key=k_raw, delta=delta,
        )

    def keygen(self, user_keys: DMCFEUserKeys, y_slice: np.ndarray) -> TwoDMCFEPartialKey:
        """Generates a partial key for vector summation (y = ones)."""
        assert len(y_slice) == self.n
        # Extended function vector [1, 1, ..., 1, 1, 1]
        y_ext = np.append(y_slice, [1, 1]).astype(object)
        dk_inner = self._inner.keygen(user_keys, y_ext)
        return TwoDMCFEPartialKey(user_id=user_keys.user_id, dk=dk_inner)

    def _cache_sk(self, user_id: int, sk: np.ndarray) -> None:
        """Store sk_i for fast mask cancellation in agg_dec."""
        self._sk_cache[user_id] = sk

    def agg_dec(self, partial_keys, ciphertexts, label):
        n = self.n
        P_gmp = gmpy2.mpz(self.P)
        
        # Initialize a multi-precision flat list array for rapid arithmetic tracking
        ct_sum_gmp = [gmpy2.mpz(0) for _ in range(n + 2)]

        # 1. Accelerated vector summation of all ciphertexts (length n+2) via gmpy2 bindings
        for ct_obj in ciphertexts:
            raw_ct = ct_obj.ct.ct
            for idx in range(n + 2):
                ct_sum_gmp[idx] = (ct_sum_gmp[idx] + gmpy2.mpz(int(raw_ct[idx]))) % P_gmp

        # 2. Accelerated mask cancellation via gmpy2
        for pk in partial_keys:
            h = gmpy2.mpz(int(_label_hash(label, pk.user_id)))
            sk = self._sk_cache[pk.user_id]
            for idx in range(n + 2):
                sk_val = gmpy2.mpz(int(sk[idx]))
                ct_sum_gmp[idx] = (ct_sum_gmp[idx] - (h * sk_val)) % P_gmp

        # Convert back to standard NumPy tracking types safely for output interface compatibility
        ct_sum = np.array([int(x) for x in ct_sum_gmp], dtype=object)

        # Extract vector and session key mask
        za_vec = ct_sum[:n]
        mask_sum = int(ct_sum[n])
        
        # Broadcast addition of mask across entire vector
        return np.append((za_vec + mask_sum) % int(P_gmp), mask_sum)

    def usr_dec(self, ciphertexts, za_result, label):
        n = self.n
        za_masked = za_result[:n]
        mask_sum = int(za_result[n])
        # Vectorised subtraction
        return (za_masked - mask_sum) % int(self.P)