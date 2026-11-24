import tensorflow as tf
import keras
from keras import layers, models, regularizers

#Model configuration
INPUT_SHAPE = (128, 251, 1)
NUM_CLASSES = 6

def create_model():
    model = models.Sequential([
        #Input layer
        layers.Input(shape=INPUT_SHAPE),
        #Normalize the Mel spectrogram values (-80 to 0 dB)
        layers.Rescaling(scale=1.0/80.0, offset=1.0),
        
        #Convolutional block 1
        layers.Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(0.001)
        ),

        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        #Convolutional block 2
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(0.001)
        ),
        
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        #Convolutional block 3
        layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(0.001)
        ),

        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        #Classification layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        #Output layer
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    return model

def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    model = create_model()
    model = compile_model(model)
    model.summary()