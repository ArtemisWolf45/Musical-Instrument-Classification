import os
import json
import numpy as np
import pandas as pd
import librosa
from pathlib import Path

#Configurable variables
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = Path(r"Dataset")
OUTPUT_PATH = BASE_DIR / "processed_data"

SELECTED_FAMILIES = ["flute", "guitar", "keyboard", "string", "brass", "organ"]

SAMPLES_PER_SPLIT = {
    "train": 1500,
    "valid": 200,
    "test": 100
}

RANDOM_STATE = 33

SAMPLE_RATE = 16000
DURATION = 4
AUDIO_LENGTH = SAMPLE_RATE * DURATION

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

N_FRAMES = 1 + AUDIO_LENGTH // HOP_LENGTH

def load_metadata(split):
    json_path = os.path.join(DATASET_PATH, f"nsynth-{split}", "examples.json")

    print(f"\nLoading {split} metadata...")

    with open(json_path, "r") as file:
        data = json.load(file)

    df = pd.DataFrame.from_dict(data, orient="index")

    print("Total samples:", len(df))

    return df

def select_samples(df, split):
    print(f"\nSelecting samples for {split}...")

    df = df[df["instrument_family_str"].isin(SELECTED_FAMILIES)]
    
    print("\nAvailable samples:")
    print(df["instrument_family_str"].value_counts())

    samples_per_family = SAMPLES_PER_SPLIT[split]

    counts = df["instrument_family_str"].value_counts()

    for family in SELECTED_FAMILIES:
        if counts.get(family, 0) < samples_per_family:
            raise ValueError(
                f"Not enough {family} samples in {split} split."
            )

    df = (
        df.groupby("instrument_family_str", group_keys=False)
        .sample(
            n=samples_per_family,
            random_state=RANDOM_STATE
        )
    )

    df = df.sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    print("\nSelected samples:")
    print(df["instrument_family_str"].value_counts())

    print("Total selected:", len(df))

    return df

def extract_mel_spectrogram(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = librosa.util.fix_length(
        audio,
        size=AUDIO_LENGTH
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db.astype(np.float32)

label_mapping = {
    family: index
    for index, family in enumerate(SELECTED_FAMILIES)
}

print("\nLabel mapping:")
print(label_mapping)

def process_split(split):
    print(f"\nProcessing {split} data...")

    df = load_metadata(split)
    df = select_samples(df, split)

    audio_folder = os.path.join(DATASET_PATH, f"nsynth-{split}", "audio")
    split_output = os.path.join(OUTPUT_PATH, split)

    os.makedirs(split_output, exist_ok=True)

    total_samples = len(df)

    X_path = os.path.join(split_output, f"X_{split}.npy")
    y_path = os.path.join(split_output, f"y_{split}.npy")

    X = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, N_MELS, N_FRAMES, 1)
    )

    y = np.zeros(
        total_samples,
        dtype=np.int64
    )

    metadata_records = []

    for i, (sample_id, row) in enumerate(df.iterrows()):
        audio_path = os.path.join(audio_folder, sample_id + ".wav")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        mel_db = extract_mel_spectrogram(audio_path)

        if mel_db.shape != (N_MELS, N_FRAMES):
            raise ValueError(
                f"Unexpected spectrogram shape for {sample_id}: "
                f"{mel_db.shape}"
            )

        X[i] = mel_db[..., np.newaxis]
        family = row["instrument_family_str"]
        y[i] = label_mapping[family]

        metadata_records.append({
            "sample_id": sample_id,
            "instrument_family": family,
            "label": label_mapping[family],
            "instrument": row["instrument_str"],
            "pitch": row["pitch"]
        })

        if (i + 1) % 100 == 0 or (i + 1) == total_samples:
            print(f"Processed {i + 1}/{total_samples} samples")

    X.flush()
    np.save(y_path, y)

    metadata_df = pd.DataFrame(metadata_records)

    metadata_df.to_csv(
        os.path.join(split_output, f"metadata_{split}.csv"),
        index=False
    )

    del X

    print(f"\n{split} preprocessing completed")
    print("Spectrogram shape:", (total_samples, N_MELS, N_FRAMES, 1))
    print("Label shape:", y.shape)

    return

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for split in ["train", "valid", "test"]:
        process_split(split)

    np.save(
        os.path.join(OUTPUT_PATH, "class_names.npy"),
        np.array(SELECTED_FAMILIES)
    )

    print("All preprocessing completed")

    print("\nInstrument labels:")
    print(label_mapping)

    print("\nProcessed data saved at:")
    print(OUTPUT_PATH)

if __name__ == "__main__":
    main()