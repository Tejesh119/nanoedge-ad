"""Model architectures for NanoEdge-AD: SensorGate, TCN, and Baseline models."""
import tensorflow as tf
import keras
from keras import layers, Model, regularizers
import numpy as np
from sklearn.tree import DecisionTreeClassifier

@keras.saving.register_keras_serializable(package="nanoedge_ad")
class SensorGate(layers.Layer):
    """Learnable per-channel gate with L1 regularization.
    
    Computes out = x * sigmoid(4 * g), where g is a 1D vector of shape (num_channels,).
    Exposes learned gate values through get_gate_values().
    """
    def __init__(self, num_channels: int = 4, l1_reg: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.l1_reg = l1_reg

    def build(self, input_shape):
        self.gate_weights = self.add_weight(
            name="sensor_gate_weights",
            shape=(self.num_channels,),
            initializer=tf.keras.initializers.Constant(1.0),
            regularizer=regularizers.l1(self.l1_reg),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, time_steps, channels)
        gate_activation = tf.sigmoid(4.0 * self.gate_weights)  # Shape: (channels,)
        # Reshape to (1, 1, channels) for broadcast multiplication
        gate_broadcast = tf.reshape(gate_activation, (1, 1, self.num_channels))
        return inputs * gate_broadcast

    def get_gate_values(self) -> np.ndarray:
        """Return numpy array of effective gate values in [0, 1]."""
        raw_w = self.gate_weights.numpy()
        return 1.0 / (1.0 + np.exp(-4.0 * raw_w))

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_channels": self.num_channels,
            "l1_reg": self.l1_reg
        })
        return config

def residual_block_1d(x, filters: int, kernel_size: int = 3, dilation_rate: int = 1, name: str = "res_block"):
    """Causal dilated Conv1D residual block fully compatible with INT8 TFLite conversion."""
    in_channels = x.shape[-1]
    
    # Conv 1
    y = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=False,
        name=f"{name}_conv1"
    )(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)

    # Conv 2
    y = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=False,
        name=f"{name}_conv2"
    )(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    # Skip projection if channel dimensions differ
    if in_channels != filters:
        shortcut = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=f"{name}_skip_conv"
        )(x)
        shortcut = layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)
    else:
        shortcut = x

    out = layers.Add(name=f"{name}_add")([shortcut, y])
    out = layers.ReLU(name=f"{name}_out_relu")(out)
    return out

def build_tcn_model(
    input_shape: tuple = (64, 4),
    blocks_cfg: list = None,
    num_classes: int = 2,
    gate_l1: float = 0.001,
    use_gate: bool = True
) -> Model:
    """Build NanoEdge-AD Temporal Convolutional Network model."""
    if blocks_cfg is None:
        blocks_cfg = [
            {"filters": 16, "kernel_size": 3, "dilation_rate": 1},
            {"filters": 16, "kernel_size": 3, "dilation_rate": 2},
            {"filters": 24, "kernel_size": 3, "dilation_rate": 4}
        ]

    inputs = layers.Input(shape=input_shape, name="sensor_input")
    
    if use_gate:
        x = SensorGate(num_channels=input_shape[-1], l1_reg=gate_l1, name="sensor_gate")(inputs)
    else:
        x = inputs

    for i, b in enumerate(blocks_cfg):
        x = residual_block_1d(
            x,
            filters=b["filters"],
            kernel_size=b["kernel_size"],
            dilation_rate=b["dilation_rate"],
            name=f"tcn_block_{i+1}"
        )

    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = Model(inputs=inputs, outputs=outputs, name="NanoEdge_TCN")
    return model

def build_baseline_cnn(input_shape: tuple = (64, 4), num_classes: int = 2) -> Model:
    """Baseline B2: Standard FP32 1D-CNN without SensorGate or channel masking."""
    inputs = layers.Input(shape=input_shape, name="cnn_input")
    x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(24, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="Baseline_CNN")

def build_dense_autoencoder(input_shape: tuple = (64, 4)) -> Model:
    """Baseline B3: Dense reconstruction autoencoder trained on normal windows."""
    flat_dim = input_shape[0] * input_shape[1]
    inputs = layers.Input(shape=input_shape, name="ae_input")
    x = layers.Flatten()(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(flat_dim, activation="linear")(x)
    outputs = layers.Reshape(input_shape)(outputs)
    return Model(inputs=inputs, outputs=outputs, name="Dense_Autoencoder")

def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """Extract 6 per-channel summary statistics for Baseline B1 (Decision Tree):
    mean, std, min, max, peak-to-peak, mean absolute difference.
    
    Input shape: (N, 64, 4) -> Output shape: (N, 24)
    """
    n_windows, win_len, n_channels = X.shape
    features = []
    
    mean = np.mean(X, axis=1)                          # (N, 4)
    std = np.std(X, axis=1)                            # (N, 4)
    f_min = np.min(X, axis=1)                          # (N, 4)
    f_max = np.max(X, axis=1)                          # (N, 4)
    ptp = f_max - f_min                                # (N, 4)
    diff = np.abs(np.diff(X, axis=1))
    mean_abs_diff = np.mean(diff, axis=1)              # (N, 4)

    feat_matrix = np.concatenate([mean, std, f_min, f_max, ptp, mean_abs_diff], axis=1)
    return feat_matrix.astype(np.float32)

def tune_decision_tree(X_train_feat: np.ndarray, y_train: np.ndarray,
                        X_val_feat: np.ndarray, y_val: np.ndarray,
                        seed: int = 42) -> DecisionTreeClassifier:
    """Tune Baseline B1 Decision Tree max_depth on validation set."""
    from sklearn.metrics import f1_score
    best_dt = None
    best_f1 = -1.0
    for depth in [3, 4, 5, 6, 8, 10, None]:
        dt = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=seed)
        dt.fit(X_train_feat, y_train)
        preds = dt.predict(X_val_feat)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_dt = dt
    return best_dt

