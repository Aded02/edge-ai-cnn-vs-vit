"""Model definitions used in the dissertation experiment."""
import tensorflow as tf

def build_cnn():
    """Build the MobileNetV2 CNN baseline."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3), include_top=False, weights=None
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(base_model.input, output, name="mobilenetv2_cnn")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_vit():
    """Build the lightweight Vision Transformer used in the experiment."""
    inputs = tf.keras.Input(shape=(32, 32, 3))
    patches = tf.keras.layers.Conv2D(64, kernel_size=4, strides=4)(inputs)
    patches = tf.keras.layers.Reshape((-1, 64))(patches)
    x = tf.keras.layers.LayerNormalization()(patches)
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="lightweight_vit")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
