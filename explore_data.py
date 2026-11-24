import json
import os
import pandas as pd
import matplotlib.pyplot as plt

#Dataset paths
TRAIN_PATH = r"Dataset\nsynth-train"
JSON_PATH = os.path.join(TRAIN_PATH,"examples.json")
AUDIO_PATH = os.path.join(TRAIN_PATH, "audio")

print("Loading metadata...")

with open(JSON_PATH, "r") as file:
    data = json.load(file)

df = pd.DataFrame.from_dict(data, orient="index")
print("Metadata has been successfully loaded")

#Basic dataset information
print("\nDataset shape:", df.shape)

print("\nAvailable columns:")
print(df.head())

print("\nDataset information:")
print(df.info())

#Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

print("\nInstrument family distribution:")

family_counts = df["instrument_family_str"].value_counts()

print(family_counts)

plt.figure(figsize=(12, 6))

family_counts.plot(kind="bar")

plt.title("NSynth Instrument Family Distribution")
plt.xlabel("Instrument Family")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\nNumber of individual instruments:",
      df["instrument"].nunique())

print("\nTop 10 most common instruments:")

print(df["instrument_str"].value_counts().head(10))

print("\nChecking audio files...")

sample_ids = df.index[:100]

missing_files = []

for sample_id in sample_ids:

    audio_file = os.path.join(
        AUDIO_PATH,
        sample_id + ".wav"
    )

    if not os.path.exists(audio_file):
        missing_files.append(sample_id)

print("Files checked:", len(sample_ids))
print("Missing files:", len(missing_files))

print("\nExample sample metadata:")

sample = df.iloc[0]

print("Sample ID:", df.index[0])
print("Instrument:", sample["instrument_str"])
print("Instrument Family:", sample["instrument_family_str"])
print("Pitch:", sample["pitch"])
print("Velocity:", sample["velocity"])
print("Sample Rate: 16000 Hz")

print("\n========== DATASET SUMMARY ==========")

print("Total samples:", len(df))

print("Number of instrument families:",
      df["instrument_family_str"].nunique())

print("Number of individual instruments:",
      df["instrument"].nunique())

print("\nSamples per family:")
print(family_counts)

print("\nExploration completed!")

selected_families = ["flute", "guitar", "keyboard", "string", "brass", "organ"]

selected_df = df[df["instrument_family_str"].isin(selected_families)]

print("\nSelected family distribution:")
print(selected_df["instrument_family_str"].value_counts())

print("\nTotal selected samples:", len(selected_df))