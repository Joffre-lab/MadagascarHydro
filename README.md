# 🇲🇬 Madagascar Watershed Explorer — Geospatial AI & Hydro-Engineering Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://madahydro.streamlit.app)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![Domain](https://img.shields.io/badge/domain-Hydrology%20%26%20Hydraulics-orange)
![AI/ML](https://img.shields.io/badge/model-Gradient%20Boosting%20Regressor-green)

**MadagascarHydro** is an end-to-end Geospatial AI and Hydro-Engineering engine designed for rapid watershed profiling, flood hazard modeling, and peak discharge prediction across Madagascar. 

By unifying high-resolution spatial datasets (FABDEM 30m, ESA WorldCover 10m, CHIRPS v2.0) with a supervised Gradient Boosting Regressor, MadaHydro bridges the gap between raw spatial data, machine learning inference, and actionable hydraulic design inputs.

---

## 🌟 Core Technical Capabilities

* **Automated Watershed Delineation:** Instant extraction of catchment boundaries, flow paths, and outlet coordinates.
* **Geospatial Morphometry Engine:** Automatic calculation of area ($S$), perimeter ($P$), Gravelius compactness index ($K_c$), average slope ($I_g$), drainage density ($D_d$), and Giandotti concentration time ($T_c$).
* **Supervised Machine Learning ($Q_{10}–Q_{100}$):** Log-transformed Gradient Boosting Regressor calibrated on gauged Malagasy basins to predict peak return discharge ($Q_{10}, Q_{25}, Q_{50}, Q_{100}$).
* **Land Cover & Climatology Analysis:** Pixel-level land-use extractions powered by ESA WorldCover (10m) and 40+ year Gumbel statistical rainfall fitting using CHIRPS v2.0 datasets.
* **"What-If" Sensitivity Engine:** Real-time scalar simulation of environmental changes (deforestation, urbanization) and climate change extreme rainfall multipliers ($P_{24\text{h}}$).
* **Automated Hydro-Engineering Reports:** 1-click generation of publication-ready PDF engineering reports complete with morphometric tables, rainfall-runoff data, and disclaimers.

---

## 🛠️ Tech Stack & Data Sources

### Geospatial & ML Architecture
* **Frontend Framework:** [Streamlit](https://streamlit.io/)
* **Geospatial Processing:** `geopandas`, `rasterio`, `shapely`, `pyproj`
* **Machine Learning:** `scikit-learn`, `joblib`, `huggingface_hub`
* **PDF Generation Engine:** `reportlab`
* **Containerization:** `Docker`

### Underlying Data Infrastructure
* **Altimetry & Digital Elevation Model:** Copernicus / FABDEM 30m (Forest And Buildings removed DEM).
* **Land Cover:** ESA WorldCover v200 (10m resolution).
* **Precipitation:** CHIRPS v2.0 (Climate Hazards Group InfraRed Precipitation with Station data).

---

## 📂 Project Structure

```text
.
├── .streamlit/
│   └── secrets.toml          # API keys & PRO licenses (ignored by Git)
├── MadaHydro.py               # Main Streamlit app & scenario engine
├── hydrology_pdf.py          # ReportLab PDF report generator
├── Dockerfile                # Deployment container config
├── requirements.txt          # Python dependencies
├── .gitignore                # Environment exclusions
└── README.md                 # Project documentation