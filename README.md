# Lung Cancer Detection

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Web Application](#web-application)
4. [Usage](#usage)
5. [Results](#results)

## Project Overview

This project applies Deep Learning and Machine Learning techniques for Lung Cancer Detection using CT scan images. Various models have been implemented and evaluated, including:

- Sequential models for binary and multi-class classification
- Convolutional Neural Networks (CNNs) with and without regularization (VGG)
- Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units
- Autoencoders and Variational Autoencoders (VAEs)

## Dataset

The project utilizes a lung cancer dataset collected from the Iraq-Oncology Teaching Hospital/National Centre for Cancer Diseases (IQ-OTH/NCCD). The dataset comprises CT scans from 110 patients, categorized into three classes: malignant, benign, and normal.

### Dataset Details

- Total Images: 1097
- Classes:
  - Malignant: 561 images
  - Benign: 120 images
  - Normal: 416 images

## Web Application

The web application provides a user-friendly interface with the following functionalities:

- **Home:** General information about Lung Cancer and detection methods.
- **Data:** Explanation of the utilized lung cancer dataset and its characteristics.
- **Classes:** Detailed information about the different classes (normal, benign, malignant) within the dataset.
- **Models:** Descriptions of the implemented models for lung cancer detection.
- **Model Executor:** Allows users to select and run a specific model, potentially visualizing the output.

## Usage

To run the web application locally:

1. Clone the repository.
2. Install dependencies listed in `requirements.txt`.
3. Run `app.py` and navigate to the local server in your browser.

## Results

The models achieved competitive performance metrics with an average accuracy of XX% across different experiments. Detailed results can be found in the respective model sections.
