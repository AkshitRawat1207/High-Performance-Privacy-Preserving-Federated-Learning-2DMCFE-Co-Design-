"""
dmcfe.py
========
Decentralised Multi-Client Functional Encryption (DMCFE) for inner product.

This implements the scheme from Chotard et al. (ASIACRYPT 2018) [35],
which is the foundation that 2DMCFE is built on top of.

Paper Section III-D defines the four algorithms:
    Setup  → (mpk, {ekᵢ, skᵢ}ᵢ)
    Enc    → ctᵢ,ℓ          (label-bound ciphertext)
    KeyGen → dkᵢ,y          (partial decryption key, user-generated)
    Dec    → z              (inner product result)

Key properties that directly block Attack A1
--------------------------------------------
• Every ciphertext carries a label ℓ (the round ID).
• Dec only combines ciphertexts that share the SAME label.
• Mixing ct from round t with ct from round t+1 → wrong label → fails.

Simulation note
---------------
Real DMCFE uses bilinear pairings over elliptic curve groups.  Here we
simulate the algebraic structure with integer arithmetic mod p.  The
security intuition is preserved; only the hardness assumption changes.

Concretely we implement the "sum-of-shares" inner-product FE variant:
    - Each user i holds a secret row  skᵢ ∈ Z_p^n  (their share of msk).
    - Encryption of xᵢ under label ℓ:
          ctᵢ,ℓ  =  xᵢ  +  H(ℓ, i) · skᵢ   (component-wise mod p)
      where H is a random oracle (hash function) binding the label.
    - Partial key for function y:
          dkᵢ,y  =  <skᵢ, yᵢ>   (inner product of user's secret with
                                   their slice of the function vector)
    - Decryption:
          Σᵢ <ctᵢ,ℓ, yᵢ>  −  Σᵢ dkᵢ,y · H(ℓ, i)
        = Σᵢ <xᵢ, yᵢ>   +   Σᵢ H(ℓ,i)·<skᵢ,yᵢ>  −  Σᵢ <skᵢ,yᵢ>·H(ℓ,i)
        = <x, y>   ✓
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from crypto_utils import (
    P, add, inner_product, make_label, mod, neg, prf, rand_vec, sub, vec_mod
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DMCFEPublicParams:
    """mpk — public parameters, broadcast to everyone."""
    n: int          # dimension of each user's message vector
    N: int          # number of users
    p: int = P      # field prime


@dataclass
class DMCFEUserKeys:
    """(ekᵢ, skᵢ) — per-user key pair."""
    user_id: int
    ek: np.ndarray  # encryption key  (= sk in this simplified scheme)
    sk: np.ndarray  # secret key used for partial KeyGen
    # In real DMCFE, ek and sk are related but not equal.  Here we set
    # ek = sk for simplicity; the security argument still holds for the
    # simulation because we model the PRF as a random oracle.


@dataclass
class DMCFECiphertext:
    """ctᵢ,ℓ — label-bound ciphertext produced by user i."""
    user_id: int
    label: bytes
    ct: np.ndarray   # encrypted vector, length n


@dataclass
class DMCFEPartialKey:
    """dkᵢ,y — partial decryption key from user i for function y."""
    user_id: int
    dk: int          # scalar  <skᵢ, yᵢ>  mod p


# ---------------------------------------------------------------------------
# Label → scalar binding  (simulates random oracle H(ℓ, i))
# ---------------------------------------------------------------------------

def _label_hash(label: bytes, user_id: int) -> int:
    """
    H(ℓ, i) → Z_p

    Binds a label and user index to a pseudorandom scalar.
    This is what ties each ciphertext to its round — changing the label
    gives a completely different scalar, so cross-round mixing decrypts
    to garbage.
    """
    raw = hashlib.sha256(label + b"||user=" + str(user_id).encode()).digest()
    return mod(int.from_bytes(raw, "big"))


# ---------------------------------------------------------------------------
# DMCFE algorithms
# ---------------------------------------------------------------------------

class DMCFE:
    """
    DMCFE for inner product.

    Usage
    -----
    >>> scheme = DMCFE(n=4, N=3)
    >>> mpk, user_keys = scheme.setup()
    >>> label = make_label(round_id=1)
    >>> ct0 = scheme.enc(user_keys[0], x0, label)
    >>> dk0 = scheme.keygen(user_keys[0], y)
    >>> # collect cts and dks from all users, then:
    >>> result = scheme.dec(partial_keys, ciphertexts, y, label)
    """

    def __init__(self, n: int, N: int):
        self.n = n   # per-user vector dimension
        self.N = N   # number of users

    # ------------------------------------------------------------------
    # Setup(1^κ, 1^n, 1^N) → (mpk, {ekᵢ, skᵢ}ᵢ)
    # ------------------------------------------------------------------
    def setup(self) -> Tuple[DMCFEPublicParams, List[DMCFEUserKeys]]:
        """
        Paper Section III-D-1:
            "Setup(1^κ, 1^n, 1^N) → (mpk, {ekᵢ, skᵢ}ᵢ∈[N])"

        Each user independently samples a secret vector skᵢ ∈ Z_p^n.
        In real DMCFE there is an interactive setup where users exchange
        public keys to derive correlated randomness.  We simulate that
        by having each user sample independently (the inter-user
        correlation is captured implicitly through the shared label hash).
        """
        mpk = DMCFEPublicParams(n=self.n, N=self.N)
        user_keys = []
        for i in range(self.N):
            sk = rand_vec(self.n)   # skᵢ ← Z_p^n
            ek = sk.copy()          # ekᵢ = skᵢ (simplified)
            user_keys.append(DMCFEUserKeys(user_id=i, ek=ek, sk=sk))
        return mpk, user_keys

    # ------------------------------------------------------------------
    # Enc(ekᵢ, xᵢ, ℓ) → ctᵢ,ℓ
    # ------------------------------------------------------------------
    def enc(
        self,
        user_keys: DMCFEUserKeys,
        x: np.ndarray,
        label: bytes,
    ) -> DMCFECiphertext:
        """
        Paper Section III-D-1:
            "Enc(ekᵢ, xᵢ, ℓ) → ctᵢ,ℓ"

        Encryption formula:
            ctᵢ,ℓ  =  xᵢ  +  H(ℓ, i) · skᵢ   mod p

        The additive term H(ℓ, i)·skᵢ is a label-dependent mask.
        The label ℓ is literally encoded into the ciphertext structure,
        so the Dec algorithm can verify it.  Cross-label combinations
        produce incoherent sums → Attack A1 is foiled.
        """
        assert len(x) == self.n, f"Expected vector of length {self.n}, got {len(x)}"
        i = user_keys.user_id
        h = _label_hash(label, i)                             # H(ℓ, i)
        mask = vec_mod(h * user_keys.sk)                      # H(ℓ,i)·skᵢ
        ct = vec_mod(np.array(x, dtype=object) + mask)        # xᵢ + mask
        return DMCFECiphertext(user_id=i, label=label, ct=ct)

    # ------------------------------------------------------------------
    # KeyGen(skᵢ, y) → dkᵢ,y
    # ------------------------------------------------------------------
    def keygen(
        self,
        user_keys: DMCFEUserKeys,
        y_slice: np.ndarray,
    ) -> DMCFEPartialKey:
        """
        Paper Section III-D-1:
            "KeyGen(skᵢ, y) → dkᵢ,y"

        Partial key formula:
            dkᵢ,y  =  <skᵢ, yᵢ>   mod p

        where yᵢ is user i's slice of the global function vector y.
        Each user generates their own partial key independently — no TPA.
        The full decryption key is the collection {dkᵢ,y}ᵢ.

        This is what achieves G2 (no TPA): there is no central authority
        holding msk.  Each user contributes their piece.
        """
        assert len(y_slice) == self.n
        dk = inner_product(user_keys.sk, y_slice)
        return DMCFEPartialKey(user_id=user_keys.user_id, dk=dk)

    # ------------------------------------------------------------------
    # Dec({dkᵢ,y}ᵢ, {ctᵢ,ℓ}ᵢ, y, ℓ) → z
    # ------------------------------------------------------------------
    def dec(
        self,
        partial_keys: List[DMCFEPartialKey],
        ciphertexts: List[DMCFECiphertext],
        y: np.ndarray,       # full function vector length n*N
        label: bytes,
    ) -> int:
        """
        Paper Section III-D-1:
            "Dec({dkᵢ,y}ᵢ, {ctᵢ,ℓ}ᵢ, y, ℓ) → z"

        Decryption formula:
            z  =  Σᵢ <ctᵢ,ℓ, yᵢ>  −  Σᵢ dkᵢ · H(ℓ, i)   mod p

        Expanding <ctᵢ,ℓ, yᵢ>:
            = <xᵢ + H(ℓ,i)·skᵢ, yᵢ>
            = <xᵢ, yᵢ>  +  H(ℓ,i)·<skᵢ, yᵢ>
            = <xᵢ, yᵢ>  +  H(ℓ,i)·dkᵢ

        Summing over i and subtracting Σ dkᵢ·H(ℓ,i):
            z = Σ<xᵢ, yᵢ> = <x, y>   ✓

        Label check: if any ciphertext's label ≠ ℓ, we abort.
        This is the enforcement mechanism against Attack A1.
        """
        # --- Label consistency check (prevents A1) ---
        for ct in ciphertexts:
            if ct.label != label:
                raise ValueError(
                    f"Label mismatch for user {ct.label}: "
                    f"expected {label!r}, got {ct.label!r}. "
                    "This is exactly Attack A1 — aborting."
                )

        # Sort by user_id to guarantee consistent ordering
        ciphertexts = sorted(ciphertexts, key=lambda c: c.user_id)
        partial_keys = sorted(partial_keys, key=lambda k: k.user_id)

        assert len(ciphertexts) == self.N
        assert len(partial_keys) == self.N

        # --- Step 1: compute Σᵢ <ctᵢ,ℓ, yᵢ> ---
        total = 0
        for i, ct in enumerate(ciphertexts):
            y_slice = y[i * self.n : (i + 1) * self.n]   # yᵢ
            total = add(total, inner_product(ct.ct, y_slice))

        # --- Step 2: subtract Σᵢ dkᵢ · H(ℓ, i) ---
        for pk in partial_keys:
            h = _label_hash(label, pk.user_id)
            total = sub(total, mod(pk.dk * h))

        return total   # = <x, y> mod p


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _selftest():
    print("=== DMCFE self-test ===")
    N, n = 3, 4
    scheme = DMCFE(n=n, N=N)
    mpk, ukeys = scheme.setup()

    # User messages
    xs = [
        np.array([1, 2, 3, 4], dtype=object),
        np.array([5, 6, 7, 8], dtype=object),
        np.array([9, 10, 11, 12], dtype=object),
    ]
    # Function vector: sum all (y = ones)
    y = np.ones(n * N, dtype=object)

    label = make_label(round_id=1)

    # Encrypt
    cts = [scheme.enc(ukeys[i], xs[i], label) for i in range(N)]

    # Key generation (decentralised — each user does this independently)
    dks = [scheme.keygen(ukeys[i], y[i*n:(i+1)*n]) for i in range(N)]

    # Decrypt
    result = scheme.dec(dks, cts, y, label)
    expected = mod(sum(int(v) for x in xs for v in x))  # <x, ones>

    print(f"  Expected : {expected}")
    print(f"  Got      : {result}")
    assert result == expected, "DMCFE self-test FAILED"
    print("  PASS ✓\n")

    # Demonstrate A1 is blocked
    label2 = make_label(round_id=2)
    xs2 = [np.array([10, 20, 30, 40], dtype=object)] * N
    cts2 = [scheme.enc(ukeys[i], xs2[i], label2) for i in range(N)]

    print("  Attempting cross-label decryption (Attack A1 simulation)...")
    mixed = [cts[0], cts2[1], cts2[2]]   # round-1 ct0 + round-2 ct1, ct2
    try:
        scheme.dec(dks, mixed, y, label)
        print("  ERROR: Should have been blocked!")
    except ValueError as e:
        print(f"  Blocked ✓ → {e}\n")


if __name__ == "__main__":
    _selftest()
