# Musical Instrument Family Classification Using CNNs

## 1. Project Overview

This project uses a Convolutional Neural Network (CNN) to classify musical instruments into six different instrument families based on their audio recordings.

The model takes an audio recording of an isolated musical note, converts it into a Mel Spectrogram, and uses the spectrogram as input to predict the corresponding instrument family.

The project uses the **NSynth (Neural Synthesizer) Dataset** for training, validation, and testing.

### Instrument Families

The model classifies audio into the following six categories:

- Flute
- Guitar
- Keyboard
- String
- Brass
- Organ

## 2. Technologies and Libraries Used

- Python
- TensorFlow
- Keras
- Librosa
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## 3. Project Structure

```
Deep Learning PBL/
│
├── processed_data/
│   ├── test/
│   │   ├── X_test.npy
│   │   └── y_test.npy
│   └── class_names.npy
│
├── sample_audio/
│   ├── flute/
│   ├── guitar/
│   ├── keyboard/
│   ├── string/
│   ├── brass/
│   └── organ/
│
├── outputs/
│   ├── final_model.keras
│   ├── training_history.png
│   ├── confusion_matrix_final.png
│   ├── classification_report.txt
│   └── evaluation_summary.txt
│
├── explore_data.py
├── preprocessing.py
├── visualize_data.py
├── cnn_model.py
├── train.py
├── evaluate.py
├── predict.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The `processed_data` and `outputs` folders should contain the files generated during preprocessing, training, and evaluation. The original NSynth dataset is not included in the submission because of its large size.*

## 4. Dataset

The project uses the NSynth dataset, which contains audio recordings of individual musical notes generated using different instruments.

The dataset is divided into three subsets:
- Training set
- Validation set
- Testing set

Only samples belonging to the six selected instrument families are used.

The audio recordings are processed into Mel Spectrograms before being passed to the CNN.

### Dataset Setup

1. Download the NSynth dataset from the link above.
2. Extract the downloaded files.
3. Update the dataset path in `preprocessing.py` to point to the extracted dataset directory.
4. Run the preprocessing script to generate the processed spectrograms.

**Note:** The original NSynth dataset is not included in this project submission because of its large size. The processed training and validation datasets are also excluded to reduce repository size. The processed test dataset is included to allow model evaluation without downloading the complete NSynth dataset. A small collection of WAV files from the test set is included in the `sample_audio` folder for demonstrating predictions without downloading the full dataset.

## 5. Audio Preprocessing

The preprocessing pipeline performs the following operations:

1. Loads audio files using Librosa.
2. Converts audio to mono.
3. Resamples audio to 16,000 Hz.
4. Adjusts audio duration to 4 seconds.
5. Generates Mel Spectrograms.
6. Converts the spectrograms into decibel scale.
7. Saves the processed spectrograms and labels as NumPy files.

### Preprocessing Parameters

| Parameter | Value |
|---|---|
| Sample Rate | 16,000 Hz |
| Audio Duration | 4 seconds |
| FFT Size | 1024 |
| Hop Length | 256 |
| Mel Bands | 128 |
| Spectrogram Shape | 128 × 251 × 1 |

## 6. CNN Model Architecture

The project uses a Sequential CNN architecture.

The model consists of:

- Three convolutional blocks with 32, 64, and 128 filters.
- Batch Normalization layers.
- ReLU activation functions.
- Max Pooling layers.
- Dropout layers to reduce overfitting.
- Global Average Pooling.
- A Dense layer with 128 neurons.
- A final Softmax output layer with six neurons.

The model uses the Adam optimizer and Sparse Categorical Crossentropy loss function.

## 7. Model Training

The model is trained using the processed training dataset and evaluated against the validation dataset during training.

The following callbacks are used:

- **Early Stopping:** Stops training when validation loss stops improving.
- **Model Checkpoint:** Saves the best-performing model.
- **Reduce Learning Rate on Plateau:** Reduces the learning rate when validation loss stops improving.

## 8. Experiments and Results

Three experiments were conducted to evaluate model performance.

| Experiment | Description | Test Accuracy |
|---|---|---:|
| Model 1 | Baseline CNN with a smaller training dataset | 78.50% |
| Model 2 | CNN trained using a larger dataset | 80.00% |
| Model 3 | CNN with SpecAugment data augmentation | 67.17% |

### Best Model

Model 2 achieved the highest test accuracy of **80.00%**.

Increasing the training dataset improved overall classification performance compared to the baseline model.

The SpecAugment experiment did not improve performance, suggesting that the augmentation settings used were not suitable for this particular dataset and model configuration.

## 9. Model Evaluation

The final model is evaluated using:

- Test Loss
- Test Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The classification report provides performance measurements for each of the six instrument families.

## 10. Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## 11. How to Run the Project

Run the Python files in the following order:

### Step 1: Explore the Dataset

```bash
python explore_data.py
```

### Step 2: Preprocess the Audio Dataset

```bash
python preprocessing.py
```

This generates the processed spectrograms and labels.

### Step 3: Visualize the Processed Data

```bash
python visualize_data.py
```

### Step 4: Train the CNN Model

```bash
python train.py
```

The best-performing model will be saved in the `outputs` folder.

### Step 5: Evaluate the Model

```bash
python evaluate.py
```

This generates the classification report, confusion matrix, and evaluation summary.

### Step 6: Predict an Instrument Family

```bash
python predict.py
```

The prediction script can be used to classify a new audio recording.

## 12. Important Notes

- The original NSynth dataset is not included in the project submission because of its large storage requirements.
- The preprocessing script requires access to the original NSynth audio files and metadata.
- The trained model and evaluation outputs are included so that the project results can be examined without retraining.
- The model is designed to classify isolated musical notes, not multiple instruments playing together in a complete music recording.

## 13. Conclusion

This project demonstrates how Convolutional Neural Networks can be applied to audio classification using Mel Spectrograms.

The final model achieved **80% test accuracy** across six musical instrument families. The experiments also demonstrated how training dataset size and data augmentation techniques can influence model performance.

The project provides a practical example of applying deep learning techniques to music and audio analysis.

## 14. Dataset Citation

This project uses the NSynth (Neural Synthesizer) Dataset.

**Dataset:** [NSynth Dataset – Google Magenta](https://magenta.tensorflow.org/datasets/nsynth)

**Citation:**

Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck,
  Karen Simonyan, and Mohammad Norouzi. "Neural Audio Synthesis of Musical Notes
  with WaveNet Autoencoders." 2017.

Paper link: https://arxiv.org/abs/1704.01279

The dataset and its associated research paper are acknowledged as the primary data source for this project.