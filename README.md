# Fire Detection using Convolutional Neural Network (CNN)

A deep learning project that classifies images into "Fire" and "Non-Fire" categories using a Convolutional Neural Network implemented with TensorFlow/Keras.

## Project Overview

This project implements a CNN-based image classification system that can distinguish between fire and non-fire scenes. The model processes input images and classifies them into two categories, making it potentially useful for early fire detection systems, forest monitoring, or safety applications.

## Model Architecture

The CNN model consists of the following layers:
- Two Convolutional layers with 32 filters (3x3) and ReLU activation
- Two MaxPooling layers for dimensionality reduction
- Flatten layer to convert 2D features to 1D
- Three Dense layers (100, 40, and 2 neurons) with ReLU activation
- Final Softmax activation for binary classification

## Dataset

The model is trained on a dataset containing:
- 1500+ images total
- Two categories: "Fire" and "Non-Fire"
- Images are resized to 32x32 pixels and normalized (0-1 range)

## Technical Features

- **Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV
- **Data Splitting**: 80-20 train-test split
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: Categorical Crossentropy
- **Training**: 25 epochs with batch size of 32

## Results

The model achieves good classification accuracy on both training and validation sets. The training progress can be visualized through accuracy plots showing both training and validation performance over 25 epochs.

## Potential Applications

- Forest fire detection systems
- Industrial safety monitoring
- Smart home safety systems
- Surveillance and security applications

## Note

This is a research/educational project. For real-world fire detection systems, consider using higher-resolution images, more sophisticated architectures, and comprehensive safety validation.
