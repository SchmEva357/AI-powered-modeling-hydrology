import streamlit as st
from streamlit_folium import folium_static
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import folium
from folium.raster_layers import ImageOverlay
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from osgeo import gdal
import logging

# --- Layout ---
st.set_page_config(page_title="River Shift Predictor", layout="wide", initial_sidebar_state="expanded")
custom_css = """
<style>
/* Clean blue-green theme */
body {
    background-color: #f0f7f9;
    color: #333333;
}
h1, h2, h3, h4 {
    color: #245c67;
}
.sidebar .sidebar-content {
    background-color: #dceef2;
}
.stButton>button {
    background-color: #2a9d8f;
    color: white;
    border-radius: 0.5rem;
}
.stButton>button:hover {
    background-color: #21867a;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🌊 River Shift Settings")

ndwi_folder = st.sidebar.text_input("Path to the NDWI-Images", value="data/ndwi_images")
exo_csv_path = st.sidebar.text_input("Path to exogenous factors", value="data/exo_factors.csv")
timestamps_csv_path = st.sidebar.text_input("Path to image timestamps", value="data/image_timestamps.csv")
model_path = "model/advanced_model.h5"

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Parameters")

delta_precip = st.sidebar.number_input("Cumulative precipitation [mm]", value=0.0)
delta_discharge = st.sidebar.number_input("Average discharge [m³/s]", value=0.0)
prec_extreme = st.sidebar.number_input("Extreme precipitation events", value=0)
disc_extreme = st.sidebar.number_input("Extreme discharge events", value=0)
forecast_horizon_days = st.sidebar.slider("Forecast Horizon (days)", 1, 500, value=14)

run_prediction = st.sidebar.button("Start Prediction")

# --- App Title ---
st.markdown("""
# River Shift Prediction App
            
*Explore predicted changes in river courses based on remote sensing and hydrological factors by applying different scenarios.*  
           
*The output is a heatmap which indicates the future river shift. The predicted direction of shift is blue and the current or past location red.*  
*The intensity of the color reflects the likelihood of change.* 
""")

# --- Extract ID ---
def extract_id_from_filename(filename):
    return int(filename.split('_')[-1].split('.')[0])

# --- Create Water Masks ---
def process_images(folder):
    masks = []
    filenames = sorted([f for f in os.listdir(folder) if f.endswith('.tif')], key=extract_id_from_filename)
    for fname in filenames:
        path = os.path.join(folder, fname)
        dataset = gdal.Open(path)
        ndwi = dataset.GetRasterBand(1).ReadAsArray()
        threshold = np.percentile(ndwi, 93)
        if np.isnan(threshold) or threshold < -0.1 or threshold > 0.1:
            threshold = np.clip(threshold, -0.1, 0.1)
        mask = ndwi > threshold
        mask = np.expand_dims(mask.astype(np.float32), axis=-1)
        masks.append(mask)
    return np.array(masks)

# --- Prepare exogenous factors - consider steps ---
def prepare_exo_data(path, forecast_horizon, step_days=14):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_diff_days'] = df['timestamp'].diff().dt.days
    df['month'] = df['timestamp'].dt.month
    month_dummies = pd.get_dummies(df['month'], prefix='month')
    df = pd.concat([df, month_dummies], axis=1).drop(columns=['month'])

    total_precip = delta_precip
    avg_discharge = delta_discharge
    total_prec_extreme = prec_extreme
    total_disc_extreme = disc_extreme

    steps = forecast_horizon // step_days
    last_ts = df['timestamp'].max()
    last_id = df['image_id'].max()
    monthly_cols = month_dummies.columns

    per_step_precip = total_precip / steps if steps else 0
    remaining_prec_extreme = total_prec_extreme
    remaining_disc_extreme = total_disc_extreme

    future_rows = []

    for i in range(steps):
        new_ts = last_ts + timedelta(days=step_days * (i + 1))
        new_id = last_id + i + 1

        new_row = {
            'timestamp': new_ts,
            'image_id': new_id,
            'discharge': avg_discharge,
            'precipitation': per_step_precip,
            'prec_extreme': 1 if remaining_prec_extreme > 0 else 0,
            'disc_extreme': 1 if remaining_disc_extreme > 0 else 0,
            'time_diff_days': step_days
        }

        for col in monthly_cols:
            new_row[col] = 1 if int(col.split('_')[-1]) == new_ts.month else 0

        if remaining_prec_extreme > 0:
            remaining_prec_extreme -= 1
        if remaining_disc_extreme > 0:
            remaining_disc_extreme -= 1

        future_rows.append(new_row)

    future_df = pd.DataFrame(future_rows)
    full_df = pd.concat([df, future_df], ignore_index=True)
    full_df.fillna(0, inplace=True)

    cols = ['discharge', 'precipitation', 'prec_extreme', 'disc_extreme', 'time_diff_days'] + list(monthly_cols)
    return full_df, full_df[cols].values

# --- Sequences for LSTM/Model - last 5 images ---
def create_sequences(images, exo, time_steps=5):
    X_img = [images[-time_steps:]]
    X_exo = [exo[-time_steps:]]
    X_future = [exo[-1]]
    return np.array(X_img), np.array(X_exo), np.array(X_future)

# --- Model is trained on forecast period of approximately 2 weeks - to capture dynamic over longer forecast period every second week a new water mask is created and will be used for future prediction ---
def predict_iteratively(images, exo_factors, model_path, forecast_horizon_steps=1, time_steps=5):
    model = load_model(model_path)
    current_images = list(images)
    current_exo = exo_factors.copy()
    predicted_images = []

    for step in range(forecast_horizon_steps):
        X_img, X_exo, X_future = create_sequences(np.array(current_images), current_exo, time_steps)
        pred = model.predict((X_img.astype(np.float32), X_exo.astype(np.float32), X_future.astype(np.float32)))
        predicted_image = pred[0]
        predicted_images.append(predicted_image)
        current_images.append(predicted_image)
        if len(current_images) > time_steps:
            current_images.pop(0)
        current_exo = np.vstack([current_exo, X_future])
    
    return predicted_images

# --- Main function ---
if run_prediction:
    st.info("Please be patient, this may take a while...")
    try:
        images = process_images(ndwi_folder)
        df, exo = prepare_exo_data(exo_csv_path, forecast_horizon_days)
        forecast_steps = forecast_horizon_days // 14
        predicted_images = predict_iteratively(images, exo, model_path, forecast_steps)
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Länge der Images: {len(images)}")
        last_pred = predicted_images[-1].squeeze()
        actual = images[-1].squeeze()
        shift = last_pred - actual
        alpha = np.where((shift < -0.15) | (shift > 0.15), 1.0, 0.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(shift, cmap='coolwarm', vmin=-1, vmax=1, alpha=alpha)
        plt.colorbar(im, ax=ax, label='Shift Intensity')

        norm_shift = (shift - shift.min()) / (shift.max() - shift.min())
        rgba_image = plt.cm.coolwarm(norm_shift)
        rgba_image[..., 3] = alpha
        output_path = "river_shift_overlay.png"
        if os.path.exists(output_path):
            os.remove(output_path)
        plt.imsave(output_path, rgba_image)

        bounds = [[47.5589, 11.4379], [47.5632, 11.4845]]
        m = folium.Map(location=[47.561, 11.461], zoom_start=14)
        ImageOverlay(image=output_path, bounds=bounds, opacity=0.9).add_to(m)
        folium.LayerControl().add_to(m)
        st.subheader("Map with Overlay")
        folium_static(m)

        m.save("river_shift_map.html")
        with open("river_shift_map.html", "rb") as f:
            st.download_button("📥 Download HTML Karte", f, file_name="river_shift_map.html")

    except Exception as e:
        st.error(f"Error: {e}")