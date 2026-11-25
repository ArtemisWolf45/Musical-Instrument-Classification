import numpy as np
import librosa
import keras

from pathlib import Path

#Configuration
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "outputs" / "final_model.keras"
DATA_DIR = BASE_DIR / "processed_data"

SAMPLE_RATE = 16000
DURATION = 4
AUDIO_LENGTH = SAMPLE_RATE * DURATION

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

#Loading the model
print("Loading model...")

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

class_names = np.load(DATA_DIR / "class_names.npy")

print("Model loaded successfully")
print("Classes:", class_names)

#Preprocess audio
def preprocess_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Fix audio length to 4 seconds
    if len(audio) < AUDIO_LENGTH:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
    else:
        audio = audio[:AUDIO_LENGTH]

    # Generate Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert to decibels
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Add batch and channel dimensions
    mel_db = np.expand_dims(mel_db, axis=(0, -1))

    return mel_db

#Predict the instrument
def predict_instrument(audio_path):

    spectrogram = preprocess_audio(audio_path)

    probabilities = model.predict(spectrogram, verbose=0)[0]

    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]

    print("\nPrediction Results")
    print("Predicted Instrument:", predicted_class)
    print("\nClass Probabilities:")

    sorted_indices = np.argsort(probabilities)[::-1]
    for index in sorted_indices:
        print(
            f"{class_names[index]}: "
            f"{probabilities[index] * 100:.2f}%"
        )

if __name__ == "__main__":
    audio_path = input(
        "Enter the path of a WAV audio file: "
    ).strip().strip('"')

    if not Path(audio_path).is_file():
        print("Error: Audio file not found.")

    else:
        predict_instrument(audio_path)
