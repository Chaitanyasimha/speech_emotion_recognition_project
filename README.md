# Speech Emotion Recognition Project

Welcome to the Speech Emotion Recognition Project! This document will guide you through the entire project, step by step. Whether you’re an expert or a beginner, this guide will help you understand and execute the project with ease.
first make sure to extract the project file that i've provided

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Setting Up the Environment](#setting-up-the-environment)
5. [Running the Project](#running-the-project)
6. [Detailed Explanation of Each Step](#detailed-explanation-of-each-step)
7. [Understanding the Results](#understanding-the-results)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

The Speech Emotion Recognition Project is a machine learning project designed to recognize emotions from speech. The project involves several key steps, including recording audio, extracting features, training a machine learning model, and predicting emotions from new audio samples.

### What Will You Learn?

- How to preprocess audio data.
- How to extract meaningful features from audio files.
- How to build, train, and evaluate a machine learning model.
- How to make predictions based on new audio recordings.

This project is built with Python and various machine learning libraries, but don’t worry if you’re not familiar with them. This guide will walk you through everything!

---

## Project Structure

Understanding the project structure will help you navigate the files and folders. Here’s a breakdown of what each folder contains:

Speech_Emotion_Recognition_Project/
│
├── data/
│ ├── raw/ # Raw audio files (.wav)
│ ├── processed/ # Processed data, including extracted features
│ └── README.md # Explanation of the data folder
│
├── notebooks/
│ ├── 1_preprocessing.ipynb # Notebook for data preprocessing
│ ├── 2_feature_extraction.ipynb # Notebook for feature extraction
│ ├── 3_model_building.ipynb # Notebook for model training
│ ├── 4_model_evaluation.ipynb # Notebook for model evaluation
│ └── 5_prediction.ipynb # Notebook for making predictions
│
├── models/
│ ├── emotion_recognition_model.pkl # Trained model
│ └── README.md # Explanation of the models folder
│
├── results/
│ ├── audio_files.csv # List of audio files and labels
│ ├── features.csv # Extracted features from audio
│ ├── confusion_matrix.png # Confusion matrix of model evaluation
│ └── README.md # Explanation of the results folder
│
├── README.md # This file
├── requirements.txt # List of required Python packages
└── LICENSE.md # Licensing information


---

## Prerequisites

### What Do You Need?

- **Basic Computer Skills**: If you can navigate folders and files on your computer, you’re good to go!
- **Python**: Python is the programming language used in this project. If you don’t have Python installed, don’t worry—we’ll guide you through it.
- **Internet Connection**: You’ll need an internet connection to download necessary packages and the required audio files.

### Downloading the Audio Files

Since the audio files are not included in this repository, you need to download them manually:

**Download Audio Files:** [RAVDESS Audio Files](https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1)

1. Click the link above to download the audio files.
2. Extract the contents of the zip file.
3. Move the extracted files into the `data/raw/` directory of this project.

This will prepare the necessary audio data for the project.

# Running the Project

### Step 1: Preprocessing

1. **Open the `notebooks/1_preprocessing.ipynb` notebook**:
   - You’ll use this notebook to preprocess the raw audio data.
   - Run each cell in the notebook by pressing `Shift + Enter` in Jupyter Notebook.

2. **What Does This Do?**
   - Loads the raw audio files.
   - Converts audio formats if needed.
   - Saves the processed data for the next steps.

### Step 2: Feature Extraction

1. **Open the `notebooks/2_feature_extraction.ipynb` notebook**:
   - This notebook will extract important features from the audio files.
   - Again, run each cell to execute the steps.

2. **What Does This Do?**
   - Extracts features like MFCCs, pitch, and zero-crossing rate from the audio.
   - Saves these features in a CSV file for model training.

### Step 3: Model Building

1. **Open the `notebooks/3_model_building.ipynb` notebook**:
   - Here, you’ll train the machine learning model.
   - Execute each cell to train the model.

2. **What Does This Do?**
   - Uses the extracted features to train a machine learning model.
   - Saves the trained model to the `models/` folder.

### Step 4: Model Evaluation

1. **Open the `notebooks/4_model_evaluation.ipynb` notebook**:
   - This notebook evaluates the trained model’s performance.
   - Run the cells to see how well the model performs.

2. **What Does This Do?**
   - Generates evaluation metrics like accuracy.
   - Creates a confusion matrix to visualize the model’s performance.

### Step 5: Prediction

1. **Open the `notebooks/5_prediction.ipynb` notebook**:
   - This notebook is used to make predictions on new audio samples.
   - Run the cells to record a new audio sample and predict the emotion.

2. **What Does This Do?**
   - Records a new audio sample using your microphone.
   - Predicts the emotion based on the recorded audio.

# Troubleshooting

### Common Issues and Solutions

1. **Python Not Found**:
   - Ensure Python is installed and added to PATH. Reinstall if necessary.

2. **Package Installation Errors**:
   - Run `pip install -r requirements.txt` to install missing packages.

3. **Recording Errors**:
   - Ensure your microphone is connected and working. Check your system’s audio settings.

4. **Model Not Loading**:
   - Verify that the `emotion_recognition_model.pkl` file is in the `models/` folder.
