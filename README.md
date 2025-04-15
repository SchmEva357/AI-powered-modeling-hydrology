# Satellite Imagery Water Body Detection and Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://www.tensorflow.org/)

A comprehensive workflow for processing satellite imagery and predicting water bodies using a CNN-LSTM model that combines remote sensing data with exogenous factors. This project focuses on monitoring water bodies in the riverbed of the Isar in the Alps.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Workflow](#workflow)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview

This project provides an end-to-end solution for water body detection and prediction using satellite imagery and hydrological data. The main components include:

- **Satellite Imagery Processing**: Retrieval and preprocessing of Sentinel-2 satellite imagery from Google Earth Engine, calculation of the Normalized Difference Water Index (NDWI), and snow/ice masking.
- **Water Mask Creation**: Generation of binary water masks using custom thresholding techniques optimized for the study area.
- **Exogenous Data Integration**: Incorporation of hydrological data, including precipitation and discharge measurements, to enhance prediction capabilities.
- **CNN-LSTM Model**: A deep learning model that combines Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal modeling.
- **Visualization Tools**: Components for visualizing predicted water masks and analyzing river shifts over time.

## Installation

### Prerequisites
- Python 3.11 or higher
- GDAL library installed on your system

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/SchmEva357/ds-capstone-AI-powered.git
   cd ds-capstone-AI-powered
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Google Earth Engine:
   ```python
   import ee
   ee.Authenticate()  # Follow the authentication steps
   ee.Initialize()
   ```

## Data Sources

### Satellite Imagery
- **Sentinel-2**: Multi-spectral imagery obtained from Google Earth Engine, with a spatial resolution of 10 meters.
- **Time Period**: 2017-04-24 to 2025-03-31
- **Cloud Filtering**: Images with less than 30% cloud coverage

### Exogenous Factors
- **Precipitation Data**: Obtained from Gewässerkundlicher Dienst Bayern
- **Discharge Measurements**: Provided by Gewässerkundlicher Dienst Bayern
- **Temporal Resolution**: Daily measurements
- **Dataset Name**: `final_dataset.csv`

## Workflow

The project workflow consists of five main steps:

### 1. Satellite Imagery Retrieval and Preprocessing
- Retrieve Sentinel-2 imagery from Google Earth Engine
- Filter images by date range, cloud coverage, and snow/ice presence
- Calculate the Normalized Difference Water Index (NDWI)
- Export processed images and metadata to local storage

### 2. Water Mask Creation
- Process NDWI images to create binary water masks
- Implement adaptive thresholding to account for seasonal variations
- Save water masks as GeoTIFF files for further analysis

### 3. Data Preparation for Model Training
- Load water masks and add dimensions required for model input
- Process exogenous factors (precipitation, discharge)
- One-hot encode temporal features (e.g., month)
- Split data into training, validation, and test sets
- Create data generators for model training

### 4. Model Building and Training
- Implement a CNN-LSTM architecture to capture both spatial and temporal patterns
- Incorporate exogenous factors as additional inputs
- Train the model with binary cross-entropy loss and IoU metric
- Save the trained model for later use

### 5. Visualization and Analysis
- Predict water masks for test data
- Create heatmaps to visualize river shifts
- Analyze prediction accuracy and model performance

## Model Architecture

The model uses a hybrid CNN-LSTM architecture:

- **CNN Component**: Processes satellite imagery through 3D convolutional layers to extract spatial features
  - 3D Convolutional layers with small kernel sizes (3x3x3)
  - Max pooling layers with small pool sizes to preserve spatial details
  - 'Valid' padding to preserve data integrity

- **LSTM Component**: Processes time-series data of exogenous factors
  - Single LSTM layer with 16 units

- **Future Exogenous Input**: Additional input for future exogenous factors at prediction time
  - Dense layer with four units

- **Output**: Dense layer reshaped to match the original image dimensions
  - Sigmoid activation for binary water mask prediction

### Model Summary:
- Input: 5 consecutive satellite images + exogenous factors
- Output: Predicted water mask for the next time step
- Optimization: Adam optimizer
- Loss Function: Binary cross-entropy
- Metrics: Accuracy, Intersection over Union (IoU)

## Results

The model successfully predicts water body extents in the Isar riverbed, capturing seasonal variations and responses to extreme weather events. Key findings include:

- **Spatial Accuracy**: The model accurately captures the spatial distribution of water bodies, particularly in areas with well-defined river channels.
- **Temporal Dynamics**: Integrating exogenous factors improves the model's ability to predict changes in water extent due to precipitation events or discharge variations.
- **Limitations**: The model's performance is constrained by the resolution of Sentinel-2 imagery (10 meters), which may not capture very narrow water channels.

## Future Improvements

The following improvements could enhance the project:

- **Loss Function**: Implement Dice Loss or a combination of Dice Loss and binary cross-entropy to better handle imbalanced datasets.
- **Multi-Sensor Data Fusion**: Incorporate Sentinel-1 SAR data for cloud-penetration capabilities.
- **Advanced Architectures**: Explore U-Net or Transformer-based models for improved spatial feature extraction.
- **Uncertainty Quantification**: Implement methods to quantify prediction uncertainty.
- **Enhanced Hydrological Components**: Include cumulative precipitation/discharge and lagged variables to capture broader trends.
- **Explainability**: Add techniques like saliency maps to improve model interpretability.
- **Resolution Enhancement**: Investigate super-resolution techniques to improve the effective resolution of water body detection.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite: https://github.com/SchmEva357


## Acknowledgments

- Gewässerkundlicher Dienst Bayern for providing discharge data
- Open-Meteo for climate data
- Google Earth Engine for satellite imagery access
