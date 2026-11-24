import numpy as np
import matplotlib.pyplot as plt

X = np.load("processed_data/train/X_train.npy", mmap_mode="r")
y = np.load("processed_data/train/y_train.npy")

class_names = np.load("processed_data/class_names.npy")

# Display one spectrogram from each class
plt.figure(figsize=(15, 10))

for label in range(6):

    index = np.where(y == label)[0][0]

    plt.subplot(2, 3, label + 1)

    plt.imshow(
        X[index, :, :, 0],
        aspect="auto",
        origin="lower"
    )

    plt.title(class_names[label])
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency")

plt.tight_layout()
plt.show()