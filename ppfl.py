"""
cnn_model.py
============
Paper-exact CNN architecture using TensorFlow / Keras.

Architecture — Chang et al. (2023) Section VII-A-2
----------------------------------------------------
Input  : (28, 28, 1)   — MNIST full resolution
Conv1  : 32 filters, 5×5, VALID padding, ReLU
Pool1  : 2×2 Max Pooling                        → (12, 12, 32)
Conv2  : 64 filters, 5×5, VALID padding, ReLU
Pool2  : 2×2 Max Pooling                        → (4, 4, 64)
Flatten:                                         → 1024
Dense1 : 512 units, ReLU
Dense2 : 10 units, Softmax

Parameter count (verified == paper's 582,026):
    Conv1 W: 32×1×5×5 = 800,   b: 32       →    832
    Conv2 W: 64×32×5×5 = 51200, b: 64      →  51264
    Dense1 W: 512×1024 = 524288, b: 512    → 524800
    Dense2 W: 10×512 = 5120,   b: 10       →   5130
    Total                                   → 582,026  ✓

Training — tf.GradientTape
--------------------------
Each local training step uses GradientTape to compute per-sample
gradients and returns them as a flat numpy vector of length 582,026.
This vector is passed directly to the 2DMCFE encryption pipeline.

Gradient layout (same order as model.trainable_variables):
    [conv1_kernel (800), conv1_bias (32),
     conv2_kernel (51200), conv2_bias (64),
     dense1_kernel (524288), dense1_bias (512),
     dense2_kernel (5120), dense2_bias (10)]
    total: 582,026

NumPy fallback
--------------
If TensorFlow is not installed the module falls back to the pure-NumPy
TinyCNN from the previous implementation.  The fallback has the same
public API so ppfl_cnn.py works unchanged regardless of backend.
The fallback is 14×14 scaled-down; the Keras model is 28×28 full-size.
A warning is printed so the user knows which backend is active.

Public API (same for both backends)
-------------------------------------
    build_model(seed)           → KerasCNN | TinyCNN
    get_flat_params(model)      → np.ndarray shape (582026,)
    set_flat_params(model, flat)
    compute_gradient(model, X, y) → (flat_grad, loss)  X:(B,1,28,28)
    apply_gradient(model, flat_grad, lr)
    evaluate_accuracy(model, X, y) → float
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
    print("[cnn_model] Using TensorFlow/Keras backend "
          f"(tf {tf.__version__})")
except ImportError:
    _TF_AVAILABLE = False
    print("[cnn_model] TensorFlow not found — using NumPy fallback CNN.")


# ---------------------------------------------------------------------------
# Keras backend
# ---------------------------------------------------------------------------

if _TF_AVAILABLE:
    import tensorflow as tf

    def build_model(seed: int = 42) -> "tf.keras.Model":
        """
        Build the paper-exact CNN using Keras Sequential API.

        Returns a compiled model with:
            optimizer = SGD (lr set per-round by ppfl_cnn)
            loss      = sparse_categorical_crossentropy
            metrics   = accuracy

        The model uses channels-last input (28, 28, 1).
        ppfl_cnn passes channels-first arrays (1, 28, 28) which are
        transposed inside compute_gradient before the Keras call.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = tf.keras.Sequential([
            # Input layer — channels-last (H, W, C)
            tf.keras.layers.Input(shape=(28, 28, 1)),

            # Conv1: 32 filters, 5×5, VALID, ReLU  → (24, 24, 32)
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=5, padding="valid",
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                bias_initializer="zeros",
                name="conv1",
            ),
            # Pool1: 2×2 Max                        → (12, 12, 32)
            tf.keras.layers.MaxPooling2D(pool_size=2, name="pool1"),

            # Conv2: 64 filters, 5×5, VALID, ReLU  → (8, 8, 64)
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=5, padding="valid",
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                bias_initializer="zeros",
                name="conv2",
            ),
            # Pool2: 2×2 Max                        → (4, 4, 64)
            tf.keras.layers.MaxPooling2D(pool_size=2, name="pool2"),

            # Flatten: 4×4×64 = 1024
            tf.keras.layers.Flatten(name="flatten"),

            # Dense1: 512 units, ReLU
            tf.keras.layers.Dense(
                512, activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                bias_initializer="zeros",
                name="dense1",
            ),

            # Dense2: 10 units, Softmax
            tf.keras.layers.Dense(
                10, activation="softmax",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                bias_initializer="zeros",
                name="dense2",
            ),
        ], name="paper_cnn")

        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def get_flat_params(model: "tf.keras.Model") -> np.ndarray:
        """
        Flatten all 582,026 trainable parameters into a 1-D numpy vector.

        Variable order follows model.trainable_variables:
            conv1/kernel, conv1/bias,
            conv2/kernel, conv2/bias,
            dense1/kernel, dense1/bias,
            dense2/kernel, dense2/bias
        """
        parts = [v.numpy().ravel() for v in model.trainable_variables]
        flat = np.concatenate(parts)
        assert len(flat) == 582026, f"Expected 582026, got {len(flat)}"
        return flat

    def set_flat_params(model: "tf.keras.Model", flat: np.ndarray) -> None:
        """
        Restore all weights from a 582,026-element flat numpy vector.
        Inverse of get_flat_params.
        """
        assert len(flat) == 582026, f"Expected 582026, got {len(flat)}"
        idx = 0
        for v in model.trainable_variables:
            n = v.numpy().size
            v.assign(flat[idx:idx + n].reshape(v.shape))
            idx += n

    def _channels_first_to_last(X: np.ndarray) -> np.ndarray:
        """
        Convert (N, 1, H, W) → (N, H, W, 1) for Keras channels-last.
        The rest of the pipeline stores images as channels-first to keep
        the crypto code framework-agnostic.
        """
        return np.transpose(X, (0, 2, 3, 1))

    def compute_gradient(
        model: "tf.keras.Model",
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute the mean gradient over a mini-batch using tf.GradientTape.

        Parameters
        ----------
        model : Keras model (paper CNN)
        X     : (B, 1, 28, 28)  float32  channels-first
        y     : (B,)             int32    class labels

        Returns
        -------
        flat_grad : (582026,)  float64  mean gradient over the batch
        loss      : float      scalar cross-entropy loss

        Implementation note
        -------------------
        GradientTape records all ops involving trainable_variables
        inside the `with` block.  We compute the mean loss over the
        batch in one forward pass, then call tape.gradient() once to
        get the full gradient vector.  This is equivalent to computing
        per-sample gradients and averaging — but faster because TF
        batches the ops.
        """
        # Keras expects channels-last
        X_tf = tf.constant(_channels_first_to_last(X), dtype=tf.float32)
        y_tf = tf.constant(y, dtype=tf.int32)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        with tf.GradientTape() as tape:
            probs = model(X_tf, training=True)       # (B, 10)
            loss  = loss_fn(y_tf, probs)             # scalar

        grads = tape.gradient(loss, model.trainable_variables)

        # Flatten and concatenate all gradient tensors
        flat_grad = np.concatenate(
            [g.numpy().ravel() for g in grads]
        )
        return flat_grad.astype(np.float64), float(loss.numpy())

    def apply_gradient(
        model: "tf.keras.Model",
        flat_grad: np.ndarray,
        lr: float,
    ) -> None:
        """
        Plain SGD update: θ ← θ − lr · ∇
        Applied directly to model weights (not via model.optimizer).
        This gives ppfl_cnn full control over the learning rate schedule.
        """
        idx = 0
        for v in model.trainable_variables:
            n = v.numpy().size
            g = flat_grad[idx:idx + n].reshape(v.shape)
            v.assign(v - lr * g)
            idx += n

    def evaluate_accuracy(
        model: "tf.keras.Model",
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Run inference on X and return fraction of correct predictions."""
        X_tf = tf.constant(_channels_first_to_last(X), dtype=tf.float32)
        probs = model(X_tf, training=False).numpy()   # (N, 10)
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == y))

    # Alias: ppfl_cnn calls TinyCNN() — wrap in a compatible class
    class TinyCNN:
        """
        Thin wrapper around a Keras model that exposes the same API
        used by ppfl_cnn.py so the orchestration layer is unchanged.

        Internally delegates to the free functions above.
        """

        PARAM_COUNT = 582_026
        INPUT_SHAPE = (1, 28, 28)   # channels-first convention

        def __init__(self, seed: int = 42):
            self._model = build_model(seed=seed)
            self._seed  = seed

        def count_params(self):
            return {
                "Conv1 W+b":  832,
                "Conv2 W+b":  51264,
                "Dense1 W+b": 524800,
                "Dense2 W+b": 5130,
                "Total":      582026,
            }

        def get_flat_params(self) -> np.ndarray:
            return get_flat_params(self._model)

        def set_flat_params(self, flat: np.ndarray) -> None:
            set_flat_params(self._model, flat)

        def compute_gradient(
            self,
            X: np.ndarray,
            y: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            """X: (B, 1, 28, 28)  y: (B,)"""
            return compute_gradient(self._model, X, y)

        def apply_gradient(self, flat_grad: np.ndarray, lr: float) -> None:
            apply_gradient(self._model, flat_grad, lr)

        def apply_gradient_with_momentum(
            self,
            gradient: np.ndarray,
            lr: float = 0.1,
            momentum: float = 0.9,
        ) -> None:
            """SGD + momentum, applied directly to Keras weights."""
            if not hasattr(self, "_mom"):
                self._mom = np.zeros_like(gradient)
            self._mom = momentum * self._mom + gradient
            apply_gradient(self._model, self._mom, lr)

        def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
            return evaluate_accuracy(self._model, X, y)

        def predict(self, x: np.ndarray) -> int:
            """x: (1, 28, 28) single image."""
            probs = self._model(
                tf.constant(_channels_first_to_last(x[np.newaxis]), dtype=tf.float32),
                training=False,
            ).numpy()
            return int(np.argmax(probs[0]))


# ---------------------------------------------------------------------------
# NumPy fallback backend  (14×14 scaled-down, same public API)
# ---------------------------------------------------------------------------

else:
    # Keep the full NumPy implementation available when TF is absent.
    # This is NOT the paper architecture — it is the scaled-down simulation
    # that was already working.  A prominent warning is printed.

    print("[cnn_model] *** WARNING *** Running scaled-down NumPy CNN (14×14).")
    print("[cnn_model]     Install TensorFlow to get the paper CNN (28×28, 582k params).")

    from typing import Dict

    def relu(x):
        return np.maximum(0, x)

    def relu_grad(x):
        return (x > 0).astype(float)

    def softmax(x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def cross_entropy_loss(probs, labels):
        n = len(labels)
        correct = probs[np.arange(n), labels]
        return -np.mean(np.log(correct + 1e-9))

    def _conv2d(x, W, b):
        C_in, H, W_in = x.shape
        C_out, _, kH, kW = W.shape
        H_out, W_out = H - kH + 1, W_in - kW + 1
        cols = np.lib.stride_tricks.as_strided(
            x,
            shape=(C_in, kH, kW, H_out, W_out),
            strides=(x.strides[0], x.strides[1], x.strides[2],
                     x.strides[1], x.strides[2]),
        ).reshape(C_in * kH * kW, H_out * W_out)
        return (W.reshape(C_out, -1) @ cols + b[:, None]).reshape(C_out, H_out, W_out)

    def _conv2d_grad(x, W, grad_out):
        C_in, H, W_in = x.shape
        C_out, _, kH, kW = W.shape
        H_out, W_out = grad_out.shape[1], grad_out.shape[2]
        grad_b = grad_out.sum(axis=(1, 2))
        cols = np.lib.stride_tricks.as_strided(
            x, shape=(C_in, kH, kW, H_out, W_out),
            strides=(x.strides[0], x.strides[1], x.strides[2],
                     x.strides[1], x.strides[2]),
        ).reshape(C_in * kH * kW, H_out * W_out)
        dout_mat = grad_out.reshape(C_out, -1)
        grad_W = (dout_mat @ cols.T).reshape(W.shape)
        dcols = W.reshape(C_out, -1).T @ dout_mat
        dcols = dcols.reshape(C_in, kH, kW, H_out, W_out)
        grad_x = np.zeros_like(x)
        for i in range(kH):
            for j in range(kW):
                grad_x[:, i:i+H_out, j:j+W_out] += dcols[:, i, j, :, :]
        return grad_x, grad_W, grad_b

    def _maxpool2d(x, size=2):
        C, H, W = x.shape
        H_out, W_out = H // size, W // size
        out  = np.zeros((C, H_out, W_out))
        mask = np.zeros_like(x)
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[c, i*size:(i+1)*size, j*size:(j+1)*size]
                    out[c, i, j] = patch.max()
                    idx = np.unravel_index(patch.argmax(), patch.shape)
                    mask[c, i*size+idx[0], j*size+idx[1]] = 1
        return out, mask

    def _maxpool2d_grad(grad_out, mask, size=2):
        C, H_out, W_out = grad_out.shape
        grad_x = np.zeros_like(mask)
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    grad_x[c, i*size:(i+1)*size, j*size:(j+1)*size] += (
                        grad_out[c, i, j] * mask[c, i*size:(i+1)*size, j*size:(j+1)*size]
                    )
        return grad_x

    class TinyCNN:
        """Scaled-down 14×14 NumPy CNN (fallback when TF is absent)."""

        PARAM_COUNT = 5_850
        INPUT_SHAPE = (1, 14, 14)

        def __init__(self, seed: int = 42):
            rng = np.random.default_rng(seed)
            self.W1 = rng.normal(0, np.sqrt(2/9),   (4, 1, 3, 3))
            self.b1 = np.zeros(4)
            self.W2 = rng.normal(0, np.sqrt(2/36),  (8, 4, 3, 3))
            self.b2 = np.zeros(8)
            self.W3 = rng.normal(0, np.sqrt(2/32),  (128, 32))
            self.b3 = np.zeros(128)
            self.W4 = rng.normal(0, np.sqrt(2/128), (10, 128))
            self.b4 = np.zeros(10)
            self._mom: Dict = {}

        def count_params(self) -> Dict:
            t = self.W1.size+self.b1.size+self.W2.size+self.b2.size
            t += self.W3.size+self.b3.size+self.W4.size+self.b4.size
            return {"Total": t}

        def get_flat_params(self) -> np.ndarray:
            return np.concatenate([
                self.W1.ravel(), self.b1.ravel(),
                self.W2.ravel(), self.b2.ravel(),
                self.W3.ravel(), self.b3.ravel(),
                self.W4.ravel(), self.b4.ravel(),
            ])

        def set_flat_params(self, flat: np.ndarray) -> None:
            idx = 0
            def _g(shape):
                nonlocal idx
                n = int(np.prod(shape))
                out = flat[idx:idx+n].reshape(shape)
                idx += n
                return out
            self.W1=_g(self.W1.shape); self.b1=_g(self.b1.shape)
            self.W2=_g(self.W2.shape); self.b2=_g(self.b2.shape)
            self.W3=_g(self.W3.shape); self.b3=_g(self.b3.shape)
            self.W4=_g(self.W4.shape); self.b4=_g(self.b4.shape)

        def _forward(self, x):
            cache = {}
            z1=_conv2d(x, self.W1, self.b1); a1=relu(z1)
            p1,m1=_maxpool2d(a1)
            cache.update({"x":x,"z1":z1,"a1":a1,"p1":p1,"mask1":m1})
            z2=_conv2d(p1,self.W2,self.b2); a2=relu(z2)
            p2,m2=_maxpool2d(a2)
            cache.update({"z2":z2,"a2":a2,"p2":p2,"mask2":m2})
            flat=p2.ravel(); cache["flat"]=flat
            z3=self.W3@flat+self.b3; a3=relu(z3)
            cache.update({"z3":z3,"a3":a3})
            z4=self.W4@a3+self.b4
            probs=softmax(z4); cache["probs"]=probs
            return probs, cache

        def _backward(self, cache, label):
            probs=cache["probs"]
            dz4=probs.copy(); dz4[label]-=1.0
            dW4=np.outer(dz4,cache["a3"]); db4=dz4; da3=self.W4.T@dz4
            dz3=da3*relu_grad(cache["z3"])
            dW3=np.outer(dz3,cache["flat"]); db3=dz3; dflat=self.W3.T@dz3
            dp2=dflat.reshape(cache["p2"].shape)
            da2=_maxpool2d_grad(dp2,cache["mask2"])
            dz2=da2*relu_grad(cache["z2"])
            dp1,dW2,db2=_conv2d_grad(cache["p1"],self.W2,dz2)
            da1=_maxpool2d_grad(dp1,cache["mask1"])
            dz1=da1*relu_grad(cache["z1"])
            _,dW1,db1=_conv2d_grad(cache["x"],self.W1,dz1)
            return np.concatenate([
                dW1.ravel(),db1.ravel(),dW2.ravel(),db2.ravel(),
                dW3.ravel(),db3.ravel(),dW4.ravel(),db4.ravel()])

        def compute_gradient(self, X, y) -> Tuple[np.ndarray, float]:
            B=len(y)
            total_g=np.zeros(self.get_flat_params().shape)
            total_l=0.0
            for i in range(B):
                probs,cache=self._forward(X[i])
                total_l+=cross_entropy_loss(probs[None],y[i:i+1])
                total_g+=self._backward(cache,int(y[i]))
            return total_g/B, total_l/B

        def apply_gradient(self, g: np.ndarray, lr: float) -> None:
            self.set_flat_params(self.get_flat_params() - lr*g)

        def apply_gradient_with_momentum(
            self, gradient: np.ndarray, lr: float=0.05, momentum: float=0.9
        ) -> None:
            idx=0
            layers=[("W1",self.W1),("b1",self.b1),("W2",self.W2),("b2",self.b2),
                    ("W3",self.W3),("b3",self.b3),("W4",self.W4),("b4",self.b4)]
            for name,param in layers:
                n=param.size
                g=gradient[idx:idx+n].reshape(param.shape)
                v=self._mom.get(name, np.zeros_like(param))
                v=momentum*v+g; self._mom[name]=v; param-=lr*v; idx+=n

        def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
            correct=sum(
                int(np.argmax(self._forward(X[i])[0]))==int(y[i])
                for i in range(len(y))
            )
            return correct/len(y)

        def predict(self, x: np.ndarray) -> int:
            probs,_=self._forward(x)
            return int(np.argmax(probs))

    # Free-function aliases so ppfl_cnn can call them without a model object
    def build_model(seed: int = 42) -> TinyCNN:
        return TinyCNN(seed=seed)

    def get_flat_params(model: TinyCNN) -> np.ndarray:
        return model.get_flat_params()

    def set_flat_params(model: TinyCNN, flat: np.ndarray) -> None:
        model.set_flat_params(flat)

    def compute_gradient(model: TinyCNN, X, y) -> Tuple[np.ndarray, float]:
        return model.compute_gradient(X, y)

    def apply_gradient(model: TinyCNN, flat_grad: np.ndarray, lr: float) -> None:
        model.apply_gradient(flat_grad, lr)

    def evaluate_accuracy(model: TinyCNN, X, y) -> float:
        return model.accuracy(X, y)