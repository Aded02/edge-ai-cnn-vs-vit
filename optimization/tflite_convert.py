"""Convert trained Keras models to TensorFlow Lite."""
from pathlib import Path
import tensorflow as tf
MODEL_DIR = Path("models")

def convert_model(model_path, output_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"Saved {output_path} ({len(tflite_model)/1024**2:.4f} MB)")

if __name__ == "__main__":
    convert_model(MODEL_DIR / "cnn.keras", MODEL_DIR / "cnn.tflite")
    convert_model(MODEL_DIR / "vit.keras", MODEL_DIR / "vit.tflite")
