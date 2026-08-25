"""Train both models using the fast experimental configuration."""
from pathlib import Path
from data import load_cifar10
from models import build_cnn, build_vit
EPOCHS = 3
BATCH_SIZE = 64
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

def train_model(model, x_train, y_train, x_test, y_test):
    return model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_cifar10()
    cnn = build_cnn(); vit = build_vit()
    print("Training CNN...")
    train_model(cnn, x_train, y_train, x_test, y_test)
    cnn.save(OUTPUT_DIR / "cnn.keras")
    print("Training Vision Transformer...")
    train_model(vit, x_train, y_train, x_test, y_test)
    vit.save(OUTPUT_DIR / "vit.keras")
    print("Models saved to ./models/")
