"""CIFAR-10 loading and preprocessing utilities."""
import tensorflow as tf
NUM_CLASSES = 10
BATCH_SIZE = 64

def load_cifar10(train_samples=10_000, test_samples=2_000):
    """Load the reduced CIFAR-10 split used for fast experimentation."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train[:train_samples].astype("float32") / 255.0
    x_test = x_test[:test_samples].astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train[:train_samples], NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test[:test_samples], NUM_CLASSES)
    return x_train, y_train, x_test, y_test
