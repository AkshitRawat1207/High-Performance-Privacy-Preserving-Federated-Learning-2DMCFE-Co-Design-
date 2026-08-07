import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# Suppress unnecessary TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TinyCNN:
    def __init__(self, input_shape, num_classes, seed=42):
        # Set seeds for reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Performance-Optimized Architecture for RGB Datasets
        self.model = Sequential([
            # Block 1: Input & Initial Features
            Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer='he_normal'),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.2),

            # Block 2: Deep Feature Learning
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            # Block 3: Dimensionality Reduction for Encryption Efficiency
            # Using GlobalAveragePooling2D significantly reduces the flat parameter count
            # making 2DMCFE encryption much faster while maintaining accuracy.
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            GlobalAveragePooling2D(),

            # Output Block
            Dense(256, activation='relu', kernel_initializer='he_normal'),
            Dropout(0.4),
            Dense(num_classes)
        ])
        
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._mom_buffer = None # Essential for Momentum updates

    def get_flat_params(self) -> np.ndarray:
        """Extracts all trainable weights (including BN gamma/beta) into a single vector."""
        return np.concatenate([v.numpy().ravel() for v in self.model.trainable_variables])

    def set_flat_params(self, flat: np.ndarray):
        """Restores weights from a flattened vector into the model layers."""
        idx = 0
        for v in self.model.trainable_variables:
            n = v.numpy().size
            v.assign(flat[idx : idx + n].reshape(v.shape))
            idx += n

    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Calculates gradients. 
        Assumes input X is (N, C, H, W) from torchvision and transposes it for TF.
        """
        # Convert (N, C, H, W) -> (N, H, W, C)
        if X.shape[1] in [1, 3]: # Check if channels are at index 1
            X = np.transpose(X, (0, 2, 3, 1))
            
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            logits = self.model(X_tf, training=True)
            loss = self.loss_fn(y_tf, logits)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        flat_grad = np.concatenate([g.numpy().ravel() for g in grads])
        
        # Gradient Clipping for Federated Stability
        gnorm = np.linalg.norm(flat_grad)
        if gnorm > 1.0: 
            flat_grad /= gnorm
            
        return flat_grad, float(loss)

    def apply_gradient_with_momentum(self, g: np.ndarray, lr: float, momentum: float = 0.9):
        """Implements Nesterov-style momentum manually for the local update."""
        if self._mom_buffer is None:
            self._mom_buffer = np.zeros_like(g)
        
        # v = m*v + g
        self._mom_buffer = momentum * self._mom_buffer + g
        
        # w = w - lr * v
        current_weights = self.get_flat_params()
        new_weights = current_weights - lr * self._mom_buffer
        self.set_flat_params(new_weights)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Standard accuracy evaluation."""
        if X.shape[1] in [1, 3]:
            X = np.transpose(X, (0, 2, 3, 1))
            
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        logits = self.model(X_tf, training=False).numpy()
        preds = np.argmax(logits, axis=1)
        return float(np.mean(preds == y))

def build_model(input_shape=(32, 32, 3), num_classes=10, seed=42):
    """Factory function for PPFLWithCNN."""
    # Ensure shape is (H, W, C) for TensorFlow
    if input_shape[0] in [1, 3]: # If (C, H, W) was passed
        input_shape = (input_shape[1], input_shape[2], input_shape[0])
        
    return TinyCNN(input_shape=input_shape, num_classes=num_classes, seed=seed)