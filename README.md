# 🌍 Satellite Imagery Preprocessing & Water Body Prediction with CNN-LSTM

This repository presents a full pipeline for satellite image analysis and time-series prediction of water body extents using Sentinel-2 imagery and exogenous environmental factors (e.g., discharge, precipitation). It integrates Earth observation data, machine learning, and geospatial processing into a structured and reproducible workflow.

## 📌 Project Objectives

1. **Satellite Image Acquisition & Preprocessing**  
   - Access Sentinel-2 imagery via Google Earth Engine (GEE).
   - Calculate NDWI (Normalized Difference Water Index).
   - Apply cloud and snow filtering.
   - Export NDWI layers and metadata for further use.

2. **Water Mask Creation**  
   - Convert NDWI images to binary water masks.
   - Save results as GeoTIFF for spatial modeling.

3. **Exogenous Data Integration**  
   - Load external factors like discharge and precipitation from:
     - [Open-Meteo](https://open-meteo.com/)
     - [Gewässerkundlicher Dienst Bayern](https://www.gkd.bayern.de/)
   - Combined dataset stored as `final_dataset.csv`.
   - Data cleaning and feature prep in `df_exogenous.ipynb`.

4. **Model Training**  
   - Use a CNN-LSTM model (Keras/TensorFlow) for temporal prediction of water masks.
   - Evaluate with metrics like IoU and MSE.

5. **Visualization & Evaluation**  
   - Visualize predicted masks and compare with actual.
   - Generate plots for insights on performance and error patterns.

## 📂 Repository Structure

```bash
├── Workflow.ipynb               # Main pipeline notebook
├── df_exogenous.ipynb           # External factor preprocessing
├── final_dataset.csv            # Combined dataset of exogenous features
├── requirements.txt             # Python environment dependencies
├── README.md                    # Project overview
```

## ⚙️ Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/satellite-ml-waterbody.git
cd satellite-ml-waterbody

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Authenticate with Google Earth Engine
python
>>> import ee
>>> ee.Authenticate()
>>> ee.Initialize()
```

## 📈 Model & Approach

The CNN-LSTM model combines:
- **CNN layers** to extract spatial features from satellite imagery.
- **LSTM layers** to model temporal dependencies across image sequences.

## 📊 Sample Outputs

- NDWI maps and binary masks
- Visual comparison: prediction vs. ground truth
- Evaluation charts (loss curves, IoU plots, etc.)

## 💡 Inspiration & Use Cases

- River morphodynamics
- Flood monitoring
- Water resource planning
- Environmental change analysis

## 🤝 Acknowledgements

- Sentinel-2 data via [Google Earth Engine](https://earthengine.google.com/)
- Hydrological data from [GKD Bayern](https://www.gkd.bayern.de/)
- Weather data from [Open-Meteo](https://open-meteo.com/)

## 📬 Contact

Created by ** Project Satellite Team ** – feel free to reach out via https://github.com/SchmEva357/ds-capstone-AI-powered

## 📜 License

MIT License — feel free to use, share, and build upon this work! If you use this code in your research, please cite: https://github.com/SchmEva357
