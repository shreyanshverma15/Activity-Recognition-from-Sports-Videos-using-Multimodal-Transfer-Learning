**Enhanced Action Recognition using Multimodal Transfer Learning**

This project implements a Two-Stream (Spatial + Temporal) CNN-LSTM model for human action recognition on the UCF-101 dataset.

Project Structure

The project is divided into 5 sequential notebooks to handle the large dataset and computational requirements on free-tier cloud platforms (Colab/Kaggle).

01_preprocessing.ipynb:

Downloads the raw UCF-101 dataset.

Extracts Spatial (RGB) frames and Temporal (Optical Flow) frames.

Saves processed .npy files to Google Drive.

Note: This is a resumable script that may need to be run multiple times.

02_copy_data.ipynb:

Copies the 50GB+ pre-processed dataset from Google Drive to the local Colab runtime for high-speed training.

03_train_model.ipynb:

Builds the Two-Stream Fusion model (MobileNetV2 + Custom CNN + LSTMs).

Trains the model for 20 epochs.

Saves the best model checkpoint to Google Drive.

04_test_model.ipynb:

Loads the trained model.

Allows you to input a video URL (YouTube) or upload a file to test prediction on new videos.

05_evaluation.ipynb:

Generates the confusion matrix and classification report for all 63 classes on the test set.

Requirements

Python 3.x

TensorFlow 2.x

OpenCV

NumPy

Matplotlib

Seaborn

