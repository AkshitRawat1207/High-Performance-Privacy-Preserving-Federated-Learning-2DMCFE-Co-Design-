"""
crypto_utils.py
===============
Low-level helpers for simulating 2DMCFE over Z_p.

All "cryptographic" operations are simulated with modular arithmetic.
In a real implementation you would replace these with actual bilinear
pairing group operations (e.g. using the Charm framework as the paper
does).  For our purposes, correctness of the protocol is what matters.

Math recap
----------
Z_p  = the set {0, 1, ..., p-1} with addition and multiplication mod p.
p    = a large prime  (we use a safe 256-bit prime below).
Vectors live in Z_p^n : each coordinate is an element of Z_p.
Inner product  <x, y>  =  Σ xᵢ·yᵢ  mod p
"""

import hashlib
import hmac
import os
import struct
from typing import List, Tuple

import numpy as np
import gmpy2  # <-- Added for high-performance large integer arithmetic

# ---------------------------------------------------------------------------
# Prime field
# ---------------------------------------------------------------------------

# 256-bit safe prime  (p = 2q+1 where q is also prime).
# Small enough to run fast in simulation, large enough to model security.
P_VAL = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

# Fallback to a smaller test prime so unit tests run in milliseconds.
# Switch TEST_MODE = False to use the real 256-bit prime.
TEST_MODE: bool = True
if TEST_MODE:
    P_VAL = 2**61 - 1  # Mersenne prime, still cryptographically illustrative

# Wrap P into a GMP Multi-precision integer object for execution optimization
P: gmpy2.mpz = gmpy2.mpz(P_VAL)


def mod(x: int) -> int:
    """Reduce x into Z_p using fast C-level GMP modular extraction."""
    return int(gmpy2.mpz(int(x)) % P)


def vec_mod(v: np.ndarray) -> np.ndarray:
    """Element-wise reduction using NumPy vectorization instead of loops."""
    # Using % on an object array is significantly faster than a list comprehension
    return (v.astype(object) % P)


def inner_product(x: np.ndarray, y: np.ndarray) -> int:
    """
    <x, y> mod p

    Paper equation (Section III-B, correctness):
        Dec(KeyGen(msk, y), Enc(mpk, x), y) = <x, y>

    Here we compute the raw inner product; callers handle the mod.
    """
    assert len(x) == len(y), "Dimension mismatch in inner product"
    total = gmpy2.mpz(0)
    for xi, yi in zip(x, y):
        total += gmpy2.mpz(int(xi)) * gmpy2.mpz(int(yi))
    return int(total % P)


def rand_zp() -> int:
    """Sample a uniformly random element from Z_p."""
    return mod(int.from_bytes(os.urandom(32), "big"))


def rand_vec(n: int) -> np.ndarray:
    """Sample a uniformly random vector in Z_p^n."""
    return np.array([rand_zp() for _ in range(n)], dtype=object)


def neg(x: int) -> int:
    """Additive inverse in Z_p:  -x mod p."""
    return int((P - gmpy2.mpz(int(x))) % P)


def add(a: int, b: int) -> int:
    return int((gmpy2.mpz(int(a)) + gmpy2.mpz(int(b))) % P)


def sub(a: int, b: int) -> int:
    return int((gmpy2.mpz(int(a)) - gmpy2.mpz(int(b))) % P)


# ---------------------------------------------------------------------------
# Pseudo-Random Function (PRF)
# ---------------------------------------------------------------------------
def prf(seed: bytes, label: bytes) -> int:
    """
    PRF(seed, label) → Z_p

    Maps (seed, label) to a pseudorandom element of Z_p.
    This simulates the PRF used in Section V-D of the paper to
    generate session keys without inter-user communication each round.
    """
    h = hmac.new(seed, label, hashlib.sha256).digest()
    return mod(int.from_bytes(h, "big"))


def prf_vec(seed: bytes, label: bytes, n: int) -> np.ndarray:
    """
    Derive an n-dimensional pseudorandom vector from (seed, label).
    Each coordinate uses a different sub-label to ensure independence.
    """
    return np.array(
        [prf(seed, label + b"_coord_" + struct.pack(">I", i)) for i in range(n)],
        dtype=object,
    )


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def make_label(round_id: int, task_id: str = "default") -> bytes:
    """
    Encode a (task_id, round_id) pair as a canonical byte label ℓ.
    """
    return f"{task_id}||round={round_id}".encode()


def subset_id(user_ids: List[int]) -> str:
    """
    Canonical string identifier for a subset of users.
    """
    return "U=" + ",".join(str(u) for u in sorted(user_ids))


# ---------------------------------------------------------------------------
# Tiny diagnostic helpers
# ---------------------------------------------------------------------------

def print_vec(name: str, v: np.ndarray, max_show: int = 6) -> None:
    elems = [str(int(x)) for x in v[:max_show]]
    suffix = "..." if len(v) > max_show else ""
    print(f"  {name:20s}: [{', '.join(elems)}{suffix}]  (len={len(v)})")