import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

from pathlib import Path
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay)

#Paths
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = OUTPUT_DIR / "final_model.keras"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading test dataset...")

X_test = np.load(DATA_DIR / "test" / "X_test.npy")
y_test = np.load(DATA_DIR / "test" / "y_test.npy")

class_names = np.load(
    DATA_DIR / "class_names.npy"
).tolist()

print("Test samples:", X_test.shape[0])
print("Classes:", class_names)

#Loads trained model
print("\nLoading trained model...")

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")

print("\nEvaluating model...")

# Compile only for evaluation
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nGenerating predictions...")

predictions = model.predict(X_test, verbose=1)
y_pred = np.argmax(predictions, axis=1)

#Classification report
report_text = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report_text)

# Save report as text and CSV
with open(OUTPUT_DIR / "classification_report.txt", "w") as file:
    file.write(report_text)

report_dict = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

print("\nGenerating confusion matrix...")

fig, ax = plt.subplots(figsize=(9, 8))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=np.arange(len(class_names)),
    display_labels=class_names,
    xticks_rotation=45,
    cmap="Blues",
    ax=ax
)

plt.title("Confusion Matrix - Final Model")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix_final.png",
    dpi=300
)
plt.show()


with open(OUTPUT_DIR / "evaluation_summary.txt", "w") as file:
    file.write("Final Model Evaluation\n")
    file.write("======================\n")
    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    file.write(f"Test Samples: {len(y_test)}\n")

print("\nEvaluation completed!")
print("Results saved in:", OUTPUT_DIR)
