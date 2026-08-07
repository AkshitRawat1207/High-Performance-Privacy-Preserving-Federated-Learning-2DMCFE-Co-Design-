import os
import time
import csv
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from cnn_model import build_model, TinyCNN
from crypto_utils import make_label, subset_id
from gradient_encoder import decode_gradient, encode_gradient
from two_dmcfe import TwoDMCFE, TwoDMCFECiphertext, TwoDMCFEPartialKey

# ---------------------------------------------------------------------------
# DYNAMIC HARDWARE & INTERNALS CONFIGURATION (STABLE VERSION)
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    import os

    # 1. Dynamically detect CPU Core layout
    logical_processors = os.cpu_count() or 4
    intra_threads = max(1, logical_processors // 2)
    inter_threads = max(1, logical_processors // 8)

    # Apply detected thread layout dynamically
    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    
    print(f"[HARDWARE] Detected {logical_processors} logical processors.")
    print(f"[HARDWARE] Dynamic Threading: intra_op={intra_threads}, inter_op={inter_threads}")

    # 2. Precision Policy Selection
    from tensorflow.keras import mixed_precision
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))
        print("[HARDWARE] GPU detected. Activated mixed_float16 policy.")
    else:
        mixed_precision.set_global_policy(mixed_precision.Policy('float32'))
        print("[HARDWARE] CPU environment. Forcing stable native float32 execution.")
except ImportError:
    pass

@dataclass
class RoundStats:
    """Container for per-round benchmarking and metrics."""
    round_id: int
    subset: List[int]
    enc_time_s: float = 0.0
    dec_time_s: float = 0.0
    train_time_s: float = 0.0
    loss: float = 0.0
    accuracy: float = 0.0
    gradient_dim: int = 0
    masked_to_aggregator: bool = True

class PPFLWithCNN:
    """
    Main PPFL Orchestrator implementing the 2DMCFE protocol 
    from Chang et al. (2023) with High-Intensity Local Training.
    """

    def __init__(
        self,
        N: int,
        T: int,
        M: int,
        compress_k: int = 0,
        lr: float = 0.1,
        batch_size: int = 64,
        local_epochs: int = 5,  
        momentum: float = 0.9,
        lr_decay: float = 0.99,
        input_shape: tuple = (32, 32, 3),
        num_classes: int = 10
    ):
        self.N = N
        self.T = T
        self.M = M
        self.compress_k = compress_k
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        self._shared_seed = os.urandom(32)
        self.stats: List[RoundStats] = []
        self._user_cnns: List[TinyCNN] = []
        self._schemes: Dict[str, TwoDMCFE] = {}
        self._agg_pkeys: Dict[str, List[TwoDMCFEPartialKey]] = {}
        self._user_keys: Dict[int, Dict[str, object]] = {}

    def setup(self, shards: List[Tuple[np.ndarray, np.ndarray]], X_test: np.ndarray, y_test: np.ndarray):
        """Initialises model parameters and cryptographic instances."""
        self._shards = shards
        self._X_test = X_test
        self._y_test = y_test

        dummy = build_model(self.input_shape, self.num_classes)
        h, w, c = self.input_shape
        dummy_grad, _ = dummy.compute_gradient(np.zeros((1, h, w, c)), np.array([0]))
        self.gradient_dim = len(dummy_grad)
        self._enc_dim = self.gradient_dim # Always encrypt full dimension to support decentralized top-k updates

        print(f"\nSETUP: {self.num_classes} classes | Params: {self.gradient_dim:,} | Enc Dim: {self._enc_dim:,}")

        self._user_cnns = [build_model(self.input_shape, self.num_classes) for _ in range(self.N)]

        for combo in combinations(range(self.N), self.M):
            sid = subset_id(list(combo))
            scheme = TwoDMCFE(n=self._enc_dim, N=self.M, shared_seed=self._shared_seed)
            self._schemes[sid] = scheme
            _, ukeys = scheme.setup()
            
            y_ones = np.ones(self._enc_dim, dtype=object)
            agg_pkeys = []
            for idx, uid in enumerate(combo):
                ukey = ukeys[idx]
                ukey.user_id = uid
                scheme._cache_sk(uid, ukey.sk)
                if uid not in self._user_keys: self._user_keys[uid] = {}
                self._user_keys[uid][sid] = ukey
                
                dk = scheme.keygen(ukey, y_ones)
                dk.user_id = uid
                agg_pkeys.append(dk)
            self._agg_pkeys[sid] = agg_pkeys

    def train(self) -> List[RoundStats]:
        """Executes the federated training loop with detailed status reporting."""
        global_params = self._user_cnns[0].get_flat_params().copy()
        rng = np.random.default_rng(42)

        print("\n" + "="*62)
        print(f"  TRAINING LOOP  (T={self.T}, M={self.M}, steps={self.local_epochs}, lr={self.lr})")
        print("="*62)

        for t in range(1, self.T + 1):
            delta = 0 if t == self.T else 1 
            label = make_label(t)
            selected = sorted(rng.choice(self.N, size=self.M, replace=False).tolist())
            sid = subset_id(selected)
            scheme = self._schemes[sid]
            pkeys = self._agg_pkeys[sid]

            print(f"\n  ── Round {t}/{self.T}  subset={selected}  δ={delta} ──")
            
            stat = RoundStats(round_id=t, subset=selected, gradient_dim=self._enc_dim, masked_to_aggregator=(delta == 1))
            t0_train = time.perf_counter()
            local_grads = []
            total_loss = 0.0
            total_steps = 0

            # Step 1: Local training
            for uid in selected:
                cnn = self._user_cnns[uid]
                cnn.set_flat_params(global_params.copy())
                X_u, y_u = self._shards[uid]
                params_start = cnn.get_flat_params().copy()

                for _ in range(self.local_epochs):
                    perm = rng.permutation(len(y_u))
                    for start_idx in range(0, len(y_u), self.batch_size):
                        end_idx = min(start_idx + self.batch_size, len(y_u))
                        idx = perm[start_idx:end_idx]
                        
                        g, loss = cnn.compute_gradient(X_u[idx], y_u[idx])
                        cnn.apply_gradient_with_momentum(g, lr=self.lr, momentum=self.momentum)
                        total_loss += loss
                        total_steps += 1
                
                effective_grad = ((params_start - cnn.get_flat_params()) / self.lr).astype(np.float64)
                local_grads.append(effective_grad)

            stat.train_time_s = time.perf_counter() - t0_train
            stat.loss = total_loss / total_steps

            # Step 1 Enhancement: Adaptive Quantization Scale calculation
            g_max = np.max([np.max(np.abs(g)) for g in local_grads])
            current_Q = int(min(10**12, max(10**5, int(10**6 / (g_max + 1e-9)))))
            print(f"    [ADAPTIVE QUANT] Round Max Gradient: {g_max:.6f} | Dynamic Q Factor: {current_Q:,}")

            # Steps 2 & 3: Combined Parallelized SECURE LOCAL Sparsification & Encryption Pipeline
            t0_enc = time.perf_counter()

            # Concurrent user data execution function executing secure edge calculations
            def parallel_client_pipeline(task_args):
                uid, grad_vector = task_args
                
                # --- OPTIMIZATION FEATURE: SECURE LOCAL TOP-K SPARSIFICATION ---
                if self.compress_k > 0 and self.compress_k < len(grad_vector):
                    # Find the threshold magnitude value for top-k selection locally
                    top_indices = np.argsort(np.abs(grad_vector))[-self.compress_k:]
                    sparse_grad = np.zeros_like(grad_vector)
                    sparse_grad[top_indices] = grad_vector[top_indices]
                else:
                    sparse_grad = grad_vector
                
                # Quantize and modulate the secure update locally
                encoded_g = encode_gradient(sparse_grad, Q=current_Q)
                return scheme.enc(self._user_keys[uid][sid], encoded_g, label, delta)

            # Process secure edge pipeline computations concurrently
            pipeline_tasks = list(zip(selected, local_grads))
            with ThreadPoolExecutor(max_workers=intra_threads) as executor:
                cts = list(executor.map(parallel_client_pipeline, pipeline_tasks))
                
            stat.enc_time_s = time.perf_counter() - t0_enc
            
            # Step 4: Aggregation & Decryption
            t0_dec = time.perf_counter()
            za_result = scheme.agg_dec(pkeys, cts, label)
            
            if delta == 1:
                grad_sum_encoded = scheme.usr_dec(cts, za_result, label)
            else:
                print("    AggDec PLAINTEXT  (δ=0, final round)")
                grad_sum_encoded = za_result[:scheme.n]
            
            grad_sum_f = decode_gradient(grad_sum_encoded, Q=current_Q)
            
            gs_norm = np.linalg.norm(grad_sum_f)
            print(f"    grad_sum norm (before avg): {gs_norm:.4f}")
            print(f"    avg_grad norm (÷{self.M})  : {(gs_norm/self.M):.4f}")
            
            avg_grad = grad_sum_f / self.M
            stat.dec_time_s = time.perf_counter() - t0_dec

            # Step 5: Global Update
            global_params -= self.lr * avg_grad
            
            # Step 6: Evaluation
            for cnn in self._user_cnns: cnn.set_flat_params(global_params.copy())
            n_eval = min(2000, len(self._y_test))
            stat.accuracy = self._user_cnns[0].accuracy(self._X_test[:n_eval], self._y_test[:n_eval])
            
            print(f"    Next Round LR: {self.lr * self.lr_decay:.4f}")
            print(f"    Train  : {stat.train_time_s:.2f}s  Enc: {stat.enc_time_s:.3f}s  Dec: {stat.dec_time_s:.3f}s")
            print(f"    Loss   : {stat.loss:.4f}  Acc: {stat.accuracy*100:.1f}%")
            
            self.lr *= self.lr_decay
            self.stats.append(stat)

        print("\n" + "="*62)
        print(f"  DONE — Final accuracy: {self.stats[-1].accuracy*100:.1f}%")
        print("="*62 + "\n")

        return self.stats

def print_benchmark(stats: List[RoundStats]):
    print("\n" + "═"*66)
    print("  BENCHMARK  (Final Report)")
    print("═"*66)
    print(f"{'Rnd':>4}   {'Subset':<10}   {'Enc':>6}   {'Dec':>6}   {'Train':>7}   {'Loss':>7}   {'Acc%':>6}   {'Hidden':>10}")
    print("  ───   ────────   ──────   ──────   ───────   ───────   ──────   ──────────")
    
    total_enc, total_dec, total_train = 0, 0, 0
    for s in stats:
        hidden_str = "YES" if s.masked_to_aggregator else "NO(final)"
        subset_str = "{" + ",".join(map(str, s.subset[:3])) + "...}"
        print(f"{s.round_id:>4}   {subset_str:<10}   {s.enc_time_s:>6.3f}   {s.dec_time_s:>6.3f}   {s.train_time_s:>7.2f}   {s.loss:>7.4f}   {s.accuracy*100:>5.1f}%   {hidden_str:>10}")
        total_enc += s.enc_time_s
        total_dec += s.dec_time_s
        total_train += s.train_time_s

    overhead = ((total_enc + total_dec) / (total_enc + total_dec + total_train)) * 100
    print(f"\n  Total enc   : {total_enc:.3f}s")
    print(f"  Total dec   : {total_dec:.3f}s")
    print(f"  Total train : {total_train:.2f}s")
    print(f"  Crypto OH   : {overhead:.1f}%")
    print(f"  Final acc   : {stats[-1].accuracy*100:.1f}%")
    print("═"*66)

def save_results_to_csv(stats: List[RoundStats], dataset_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ppfl_results_{dataset_name}_{timestamp}.csv"
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Round", "Accuracy", "Loss", "Enc_Time", "Dec_Time", "Train_Time", "Grad_Dim", "Subset"])
        for s in stats:
            w.writerow([
                s.round_id, 
                f"{s.accuracy * 100:.2f}", 
                f"{s.loss:.4f}", 
                f"{s.enc_time_s:.3f}", 
                f"{s.dec_time_s:.3f}", 
                f"{s.train_time_s:.2f}", 
                s.gradient_dim, 
                s.subset
            ])
    print(f"\n[FILE] Results saved cleanly to: {filename}")