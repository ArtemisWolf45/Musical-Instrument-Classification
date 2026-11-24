import numpy as np
import matplotlib.pyplot as plt
import keras

from pathlib import Path
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from cnn_model import create_model, compile_model

#Paths and configurable variables
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = OUTPUT_DIR / "best_model.keras"

EPOCH = 30
BATCH_SIZE = 16
RANDOM_SEED = 33

keras.utils.set_random_seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Loading processed data
print("Loading datasets...")

X_train = np.load(DATA_DIR / "train" / "X_train.npy")
y_train = np.load(DATA_DIR / "train" / "y_train.npy")

X_valid = np.load(DATA_DIR / "valid" / "X_valid.npy")
y_valid = np.load(DATA_DIR / "valid" / "y_valid.npy")

X_test = np.load(DATA_DIR / "test" / "X_test.npy")
y_test = np.load(DATA_DIR / "test" / "y_test.npy")

class_names = np.load(DATA_DIR / "class_names.npy")

print("\nDataset shapes:")
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_valid.shape, y_valid.shape)
print("Testing:", X_test.shape, y_test.shape)

print("\nClasses:", class_names)

#Compiling model
print("\nCreating CNN model...")

model = create_model()
model = compile_model(model)

model.summary()

#Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

#Training model
print("\nStarting model training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

#History of plot training
plt.figure(figsize=(12, 5))

# Accuracy graph
plt.subplot(1, 2, 1)

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss graph
plt.subplot(1, 2, 2)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "training_history.png")
plt.show()

print("\nTraining has been completed")

print("Best model saved at:", MODEL_PATH)