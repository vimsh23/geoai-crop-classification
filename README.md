# GeoAI Crop Classification Pipeline

## Project Overview
This repository contains an end-to-end Earth Observation (EO) data processing and modeling pipeline designed for pixel-wise crop classification using multi-temporal Sentinel-2 satellite imagery. Developed as a comprehensive solution for agricultural monitoring, the system ingests the PASTIS dataset and applies a hybrid modeling architecture to map land-use and land-cover over space and time.

The system is built with modularity and scalability in mind, encompassing custom geospatial data loading, rigorous preprocessing (including temporal interpolation and cloud mitigation), hybrid model architectures, and an interactive analytical dashboard for visualization.

## Core Capabilities
- **Classical ML & Feature Engineering**: Highly optimized temporal and spectral feature extraction feeding into Random Forest and LightGBM algorithms. Includes an advanced Object-Based Image Analysis (OBIA) post-processing module for spatial regularization and noise reduction.
- **Deep Learning Architectures**: Implements baseline convolutional networks (U-Net, ResU-Net) and vision transformers (TransUNet) for end-to-end spatial-temporal feature extraction directly from raw multidimensional arrays.
- **Interactive Analytical Dashboard**: A robust Streamlit-based web application providing dynamic exploration of spectral profiles, temporal dynamics, model comparisons, and performance metrics.
- **Geospatial Processing Utilities**: Automated deployment pipelines that translate raw multidimensional predictions into georeferenced GeoTIFFs, complete with custom QML styling for immediate ingestion into enterprise GIS platforms (e.g., QGIS).

## Repository Structure

```
pastis-crop-classification/
├── README.md                          # This file
├── report.md                          # Analysis, interpretation, and findings
├── requirements.txt                   # Python dependencies
├── configs/
│   └── config.yaml                    # Central project configuration
├── notebooks/
│   └── ultimate_geoai_eda.ipynb             # Main EDA and exploration notebook
├── src/                               # Reusable modules
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
├── tools/                             # GIS & Presentation Utilities
│   ├── export_predictions_geotiff.py  # Advanced GeoTIFF export pipelines
│   ├── vectorize_labels.py            # Vectorization of raster predictions
│   ├── generate_slide.py              # Automated PPT slide generation
│   ├── train_rf_cloud_free.py         # Cloud-free RF experiment
│   └── ... (other geospatial utilities)
├── app/                               # Interactive Streamlit Dashboard
│   ├── streamlit_app.py               # Main entrypoint
│   ├── i18n.py                        # Internationalization support
│   └── components/                    # Modular dashboard UI tabs
│       ├── crop_map.py
│       ├── dataset_dashboard.py
│       ├── model_comparison.py
│       ├── prediction_portal.py
│       ├── spectral_explorer.py
│       └── temporal_explorer.py
├── splits/                            # Train/Val/Test patch IDs
├── outputs/                           # Generated figures, metrics, and predictions
└── data/                              # Put PASTIS_subset here
```

## Setup & Execution

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation
Ensure the `PASTIS_subset` folder is placed or symlinked into the `data/` directory:
```bash
ln -s /path/to/PASTIS_subset data/PASTIS_subset
```

### 3. Understanding the Dataset and Pipeline
The dataset consists of multi-temporal Sentinel-2 imagery stored as `.npy` (NumPy) arrays. Each patch has dimensions `(Time_Steps, Channels, Height, Width)` for the images, and `(3, Height, Width)` for the annotations.

To run the entire pipeline from data ingestion to model training and evaluation, follow these steps:

#### Step A: Exploratory Data Analysis (EDA)
The easiest way to understand how the numpy arrays are loaded and visualized is through the main notebook:
```bash
jupyter notebook notebooks/ultimate_geoai_eda.ipynb
```

#### Step B: Train Models from Scratch
If you wish to train the models entirely from scratch using the `.npy` arrays, execute the training script. The script automatically handles data loading, temporal feature engineering, and model training.

**Train Random Forest Baseline:**
```bash
python -m src.train --model rf
```
**Train Deep Learning Models (Requires GPU):**
```bash
python -m src.train --model resunet
# Other available models: unet, transunet
```

#### Step C: Evaluate and Publish Results
Once models are trained, run the evaluation script to generate predictions, calculate IoU/F1 scores, and save the metrics required for the dashboard:
```bash
python -m src.evaluate --model all
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 5. Convert to GeoTIFF and Generate Slides
To export all predictions (with OBIA smoothing for ML models and pure raw outputs for DL models) and generate comparison slides:
```bash
python tools/export_predictions_geotiff.py --obia
python tools/generate_slide.py
```

## Assumptions & Limitations
- **Georeferencing**: The provided metadata contains a bounding box for each patch. The GeoTIFF converter uses this to create an affine transform, assuming the 128x128 pixels are evenly distributed across the bounding box. This is an approximation.
- **Model Simplification**: The Deep Learning models implemented are lightweight baselines for demonstration and compute constraints, not heavily optimized.
- **Missing Classes**: Class 9 (Beet) is absent from the provided 102-patch subset and is excluded from evaluation. Class 19 (Void) is masked during training.



## System Reproduction & Deployment

To reproduce this pipeline or deploy the application in a new computational environment, follow the steps below to ensure all dependencies and data structures are correctly configured.

### 1. Repository and Asset Preparation
Ensure the project repository is instantiated and the necessary directory structures are maintained:

- **Source Code**: Ensure the `src/` and `app/` directories are present.
- **Configuration**: Ensure `configs/` contains the necessary settings.
- **Dataset**: Mount or copy the `PASTIS_subset/` dataset (including `ANNOTATIONS`, `DATA_S2`, and `metadata.geojson`) into the `data/` directory.
- **Pre-trained Weights (Optional)**: To bypass model training, download the pre-trained model weights from [Google Drive](https://drive.google.com/drive/folders/1IUiC3IfmYySwBG15TFMtXM7_dmSDufIP?usp=drive_link) and extract them into the `models/` directory. *(Note: Model artifacts are hosted externally to maintain a lightweight repository and ensure fast cloning).*

### 2. Environment Initialization
Initialize a dedicated Python virtual environment (Python 3.10+ recommended) to isolate project dependencies:

```bash
python -m venv venv
```
Activate it:

- Linux/Mac: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

### 3. Dependency Installation
Install the requisite Python packages and libraries specified in the project requirements:

```bash
pip install -r requirements.txt
```

### 4. Application Launch
With the environment configured, initialize the Streamlit dashboard to explore the data and model performance:

```bash
streamlit run app/streamlit_app.py
```

*(Note: Refer to Section 3 above if you need to retrain the models from scratch).*

#### 6. Generate Cloud-Free Data and Train RF (Experiment)
To improve Random Forest performance by mitigating cloud cover, you can generate a cloud-free version of the dataset and train a specialized model:

```bash
# 1. Generate cloud-free dataset (applies temporal interpolation)
python -m src.preprocessing_cloud_removal

# 2. Train the Cloud-Free Random Forest
python tools/train_rf_cloud_free.py
```