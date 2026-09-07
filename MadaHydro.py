import os
import streamlit as st

# --- 1. Configuration GDAL pour streaming HTTP Range / COG ---
os.environ.setdefault("GDAL_HTTP_USERAGENT", "MadaHydro/1.0")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "2")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "20")

# Une seule plage HTTP à la fois : plus robuste avec les endpoints CDN/Xet de HF.
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "NO")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")

# Cache VSI limité pour éviter une explosion de RAM sur Streamlit Cloud.
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "16777216")
os.environ.setdefault("GDAL_CACHEMAX", "32")
os.environ.setdefault("GDAL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import time
import gc
import psutil
import pandas as pd
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
from pysheds.grid import Grid
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_window, geometry_mask
from rasterio.windows import transform as window_transform
from affine import Affine
import io
import tempfile
import zipfile
import re
import joblib
from huggingface_hub import hf_hub_download
from hydrology_pdf import create_pdf_report

# --- 2. Ressources distantes Hugging Face ---
HF_REPO_ID = "Reynolds002/MadaHydro-data"
HF_REPO_TYPE = "dataset"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

def hf_vsi_url(filename: str) -> str:
    encoded = filename.replace("\\", "/")
    return f"/vsicurl/{HF_BASE_URL}/{encoded}"

FLOW_DIR_PATH        = hf_vsi_url("FlowDir_cog.tif")
FLOW_ACC_PATH        = hf_vsi_url("FlowAcc_cog.tif")
DEM_PATH             = hf_vsi_url("DemMada_cog.tif")
P10_RAINFALL_PATH    = hf_vsi_url("P10_24max.tif")
ANNUAL_RAINFALL_PATH = hf_vsi_url("CHIRPS_Annual_Mada.tif")
WORLD_COVER_PATHS    = [
    hf_vsi_url("WorldCover_1_COG.tif"),
    hf_vsi_url("WorldCover_2_COG.tif"),
]

MODEL_HF_FILENAME = "Q10_Model_Prediction/Q10_global_logq10.joblib"
VALID_PRO_KEYS = ["HYDRO-PRO-2026", "MADA-HYDRO-PRO", "EXUTOIRE-2026"]

# Limite fonctionnelle pour protéger Streamlit Cloud sans modifier
# la résolution des rasters ni la logique hydrologique.
MAX_CATCHMENT_AREA_KM2 = 20_000.0
MAX_DELINEATION_ITERATIONS = 3


# --- Modèle ML & Classes ---
class Q10FeatureEngineer:
    def __init__(self):
        self.feature_names_out_ = [
            "log_Surface_km2", "Indice_Pluvio_Hmm", "Indice_Pente_m_km",
            "Indice_Exonde", "Coef_Imperm", "Indice_Veg",
        ]
    def fit(self, X, y=None): return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=["Surface_km2", "Indice_Pluvio_Hmm", "Indice_Pente_m_km", "Indice_Exonde", "Coef_Imperm", "Indice_Veg"])
        X = X[["Surface_km2", "Indice_Pluvio_Hmm", "Indice_Pente_m_km", "Indice_Exonde", "Coef_Imperm", "Indice_Veg"]].copy()
        X["log_Surface_km2"] = np.log(np.maximum(X["Surface_km2"].astype(float), 1e-8))
        return X[self.feature_names_out_].values
    def get_feature_names_out(self, input_features=None): return np.array(self.feature_names_out_)

@st.cache_resource
def load_q10_model():
    try:
        local_model = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_HF_FILENAME,
            repo_type=HF_REPO_TYPE,
        )
        model_data = joblib.load(local_model)
        if isinstance(model_data, dict) and "model" in model_data:
            return model_data["model"], True
        if hasattr(model_data, "predict"):
            return model_data, True
    except Exception:
        pass
    return None, False

def get_q10_model():
    return load_q10_model()

def get_madagascar_utm_epsg(longitude):
    return 32739 if longitude >= 48.0 else 32738

def compute_passini_tc(area_km2, l_rect_km, slope_m_km):
    """Calcul du temps de concentration de Passini (en heures).
    Pente en m/m = slope_m_km / 1000.0.
    """
    safe_slope_m_m = max(float(slope_m_km) / 1000.0, 0.0001)
    safe_length    = max(float(l_rect_km), 0.1)
    safe_area      = max(float(area_km2), 0.01)
    
    tc_hours = 0.108 * ((safe_area * safe_length) ** (1.0 / 3.0)) / np.sqrt(safe_slope_m_m)
    return max(round(tc_hours, 2), 0.10)

def detect_versant(lat, lon):
    if lat > -16.0:
        return "Nord_Est" if lon >= 48.0 else "Nord_Ouest"
    elif lat < -22.0:
        return "Sud_Est" if lon >= 46.5 else "Sud"
    else:
        if lon >= 47.5:
            return "Centre_Est"
        elif lon <= 45.5:
            return "Sud_Ouest" if lat < -20.0 else "Nord_Ouest"
        else:
            return "Hautes_Terres"

# --- Fonctions raster & calculs mis en CACHE ---
@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def get_local_raster_mean_cached(gdf_json: str, raster_path: str, fallback_value: float, max_dim: int = 512):
    """Lit une statistique raster locale avec sortie scalaire uniquement.

    Le cache ne conserve jamais le tableau raster : seulement la moyenne finale.
    Cela limite fortement la RAM tout en évitant les lectures répétées pour une
    même géométrie.
    """
    data = None
    inside = None
    vals = None
    try:
        gdf_polygon = gpd.read_file(io.StringIO(gdf_json))
        with rasterio.open(raster_path) as src:
            gdf_proj = gdf_polygon.to_crs(src.crs)
            shapes = [geom for geom in gdf_proj.geometry if geom is not None and not geom.is_empty]
            if not shapes:
                return fallback_value

            win = geometry_window(src, shapes, pad_x=0, pad_y=0)
            out_h = min(max_dim, max(1, int(win.height)))
            out_w = min(max_dim, max(1, int(win.width)))

            data = src.read(
                1,
                window=win,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.average,
                masked=False,
            ).astype(np.float32, copy=False)

            out_transform = window_transform(win, src.transform) * Affine.scale(
                win.width / out_w,
                win.height / out_h,
            )

            inside = geometry_mask(
                shapes,
                out_shape=(out_h, out_w),
                transform=out_transform,
                invert=True,
            )

            nodata = src.nodata
            valid = inside & np.isfinite(data) & (data > 0)
            if nodata is not None:
                valid &= data != nodata

            vals = data[valid]
            if vals.size:
                return float(np.mean(vals))

    except Exception:
        return fallback_value
    finally:
        if vals is not None:
            del vals
        if inside is not None:
            del inside
        if data is not None:
            del data
        gc.collect()

    return fallback_value


@st.cache_data(ttl=1800, max_entries=16, show_spinner=False)
def extract_local_landcover_cached(gdf_json: str, area_km2: float):
    gdf_polygon = gpd.read_file(io.StringIO(gdf_json))
    fallback = (0.80, 0.80, 0.50, 0.00, False, pd.DataFrame())
    raster_files = WORLD_COVER_PATHS

    class_names = {
        10: "Forêt", 20: "Arbustes", 30: "Herbacé", 40: "Cultures (Rizières)", 
        50: "Bâti/Route", 60: "Sol nu", 70: "Neige", 80: "Eau libre", 
        90: "Zones humides", 95: "Mangroves"
    }
    e_weights = {10: 1.0, 20: 1.0, 30: 1.0, 40: 0.20, 50: 1.0, 60: 1.0, 70: 1.0, 80: 0.10, 90: 0.15, 95: 0.15}
    g_weights = {10: 0.60, 20: 0.70, 30: 0.75, 40: 0.85, 50: 1.0, 60: 0.90, 70: 0.5, 80: 1.00, 90: 0.95, 95: 0.90}
    v_weights = {10: 0.90, 20: 0.70, 30: 0.50, 40: 0.60, 50: 0.05, 60: 0.05, 70: 0.0, 80: 0.05, 90: 0.80, 95: 0.85}

    total_pixel_counts = {}

    for r_path in raster_files:
        data = inside = valid_pixels = unique = counts = None
        try:
            with rasterio.open(r_path) as src:
                gdf_proj = gdf_polygon.to_crs(src.crs)
                shapes = [geom for geom in gdf_proj.geometry]
                try:
                    win = geometry_window(src, shapes, pad_x=0, pad_y=0)
                    out_h = min(512, max(1, int(win.height)))
                    out_w = min(512, max(1, int(win.width)))
                    data = src.read(
                        1,
                        window=win,
                        out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.nearest,
                        masked=False,
                    )
                    out_transform = window_transform(win, src.transform) * Affine.scale(
                        win.width / out_w, win.height / out_h
                    )
                    inside = geometry_mask(
                        shapes,
                        out_shape=(out_h, out_w),
                        transform=out_transform,
                        invert=True,
                    )
                    nodata = src.nodata if src.nodata is not None else 0
                    valid_pixels = data[inside & (data != nodata)]

                    if len(valid_pixels) > 0:
                        unique, counts = np.unique(valid_pixels, return_counts=True)
                        for u, c in zip(unique, counts):
                            total_pixel_counts[u] = total_pixel_counts.get(u, 0) + c
                except ValueError:
                    continue
        except Exception:
            continue
        finally:
            if counts is not None:
                del counts
            if unique is not None:
                del unique
            if valid_pixels is not None:
                del valid_pixels
            if inside is not None:
                del inside
            if data is not None:
                del data
            gc.collect()

    total_pixels = sum(total_pixel_counts.values())
    if total_pixels == 0:
        return fallback

    lc_data = []
    e_sum, g_sum, v_sum = 0, 0, 0
    wet_pixels = 0

    for code, count in total_pixel_counts.items():
        if code in class_names:
            pct = count / total_pixels
            area_class = pct * area_km2
            lc_data.append({"Type de sol": class_names[code], "Surface (km²)": area_class, "Proportion (%)": pct * 100})
            e_sum += count * e_weights.get(code, 0.8)
            g_sum += count * g_weights.get(code, 0.8)
            v_sum += count * v_weights.get(code, 0.5)
            if code in [40, 80, 90, 95]:
                wet_pixels += count

    df_lc = pd.DataFrame(lc_data).sort_values("Surface (km²)", ascending=False)
    return float(e_sum / total_pixels), float(g_sum / total_pixels), float(v_sum / total_pixels), float(wet_pixels / total_pixels), True, df_lc

def calibrer_indices_egv(g_raw, v_raw, e_raw, pct_wet, area_km2, versant):
    if pct_wet > 0.30 and e_raw > 0.50 and area_km2 > 50:
        e_cal = max(e_raw * (1.0 - 0.5 * pct_wet), 0.15)
    else: e_cal = e_raw
    return g_raw, v_raw, e_cal

@st.cache_data(ttl=1800, max_entries=32, show_spinner=False)
def calculate_local_rainfall_cached(gdf_json: str):
    p_annuelle_mm = get_local_raster_mean_cached(gdf_json, ANNUAL_RAINFALL_PATH, 1450.0)
    p10_mm        = get_local_raster_mean_cached(gdf_json, P10_RAINFALL_PATH, 142.0)

    monthly_ratios = [0.26, 0.22, 0.16, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.03, 0.08, 0.14]
    p_mensuelles = [float(p_annuelle_mm * ratio) for ratio in monthly_ratios]

    p_design = {
        10:  round(p10_mm, 1),
        25:  round(p10_mm * 1.146, 1),
        50:  round(p10_mm * 1.254, 1),
        100: round(p10_mm * 1.362, 1),
    }
    return float(p_annuelle_mm), p_mensuelles, p_design

def compute_q10_ml(model, area_km2, p10_mm, slope_m_km, e_cal, g_cal, v_cal, p_design):
    log_surface = float(np.log(max(area_km2, 0.01)))
    X = pd.DataFrame([{
        "log_Surface_km2"  : log_surface, "Indice_Pluvio_Hmm": float(p10_mm),
        "Indice_Pente_m_km": float(slope_m_km), "Indice_Exonde": float(e_cal),
        "Coef_Imperm": float(g_cal), "Indice_Veg": float(v_cal),
    }])
    log_q10_specific = float(model.predict(X)[0])
    Q10              = max(float(np.exp(log_q10_specific)) * area_km2, 0.0)
    p10 = max(p_design[10], 1.0)
    q_dict = {10: Q10}
    for T in [25, 50, 100]: q_dict[T] = Q10 * (p_design[T] / p10) ** 1.39
    return q_dict

# --- Moteur d'Auto-Expansion Dynamique (Optimisé & Performant) ---
def get_dynamic_catchment(target_lon, target_lat, initial_buffer, threshold):
    """Délimitation exacte avec auto-expansion rapide et nettoyage mémoire RAM."""
    
    xmin = target_lon - initial_buffer
    xmax = target_lon + initial_buffer
    ymin = target_lat - initial_buffer
    ymax = target_lat + initial_buffer

    d8_esri = (64, 128, 1, 2, 4, 8, 16, 32)
    MAX_DELINEATION_ITERATIONS = 3

    sub_grid = None
    sub_fdir = None
    sub_acc = None
    catchment = None

    for iteration in range(1, MAX_DELINEATION_ITERATIONS + 1):
        bbox = (xmin, ymin, xmax, ymax)

        current_grid = None
        current_fdir = None
        current_acc = None
        current_catchment = None

        try:
            current_grid = Grid.from_raster(FLOW_DIR_PATH, window=bbox)

            current_fdir = current_grid.read_raster(
                FLOW_DIR_PATH,
                window=bbox,
                d8_mapping=d8_esri,
                dtype=np.uint8,
            )

            current_acc = current_grid.read_raster(
                FLOW_ACC_PATH,
                window=bbox,
                dtype=np.float32,
            )

            stream_mask = current_acc > threshold
            if not np.any(stream_mask):
                raise ValueError(
                    f"Aucun cours d'eau trouvé avec un seuil d'accumulation de {threshold}. "
                    "Diminuez le seuil."
                )

            x_snap, y_snap = current_grid.snap_to_mask(
                stream_mask,
                (target_lon, target_lat),
            )

            current_catchment = current_grid.catchment(
                x=x_snap,
                y=y_snap,
                fdir=current_fdir,
                d8_mapping=d8_esri,
                xytype="coordinate",
            )

            # 🔍 Cast explicite en numpy array natif pour éviter l'ambiguïté de vérité PySheds
            touch_north = bool(np.asarray(current_catchment[0:4, :]).any())
            touch_south = bool(np.asarray(current_catchment[-4:, :]).any())
            touch_west  = bool(np.asarray(current_catchment[:, 0:4]).any())
            touch_east  = bool(np.asarray(current_catchment[:, -4:]).any())
            touches_boundary = touch_north or touch_south or touch_west or touch_east

            # Si le bassin est totalement englobé
            if not touches_boundary or iteration == MAX_DELINEATION_ITERATIONS:
                sub_grid = current_grid
                sub_fdir = current_fdir
                sub_acc = current_acc
                catchment = current_catchment
                current_grid = current_fdir = current_acc = current_catchment = None
                break

            # 🚀 PAS D'EXPANSION PUISSANT 
            # Garantit un saut d'au moins 0.8° à 1.0° dès le premier dépassement
            step = max(float(initial_buffer) * 1.2, 0.8)
            print(f"[Auto-Expansion] Itération {iteration}: bassin coupé, agrandissement de la fenêtre (+{step:.2f}°)")

            if touch_west:
                xmin -= step
            if touch_east:
                xmax += step
            if touch_south:
                ymin -= step
            if touch_north:
                ymax += step

        finally:
            # Purge stricte des objets temporaires pour éviter les fuites RAM
            if current_catchment is not None: del current_catchment
            if current_acc is not None: del current_acc
            if current_fdir is not None: del current_fdir
            if current_grid is not None: del current_grid
            gc.collect()

    if sub_grid is None or catchment is None:
        raise ValueError("Impossible de délimiter le bassin versant aux coordonnées indiquées.")

    sub_dem = sub_grid.read_raster(DEM_PATH, window=bbox, dtype=np.float32)
    return sub_grid, catchment, sub_fdir, sub_acc, sub_dem

@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def delineate_catchment_geometry_cached(target_lon: float, target_lat: float, initial_buffer: float, threshold: int):
    """Cache uniquement la géométrie finale ; jamais FlowDir/FlowAcc."""
    grid_obj = catchment = sub_fdir = sub_acc = None
    try:
        grid_obj, catchment, sub_fdir, sub_acc, features = get_dynamic_catchment(
            target_lon,
            target_lat,
            initial_buffer,
            threshold,
        )
        return features
    finally:
        if sub_acc is not None:
            del sub_acc
        if sub_fdir is not None:
            del sub_fdir
        if catchment is not None:
            del catchment
        if grid_obj is not None:
            del grid_obj
        gc.collect()


def extract_river_network_ondemand(target_lon, target_lat, buffer_deg, threshold):
    """Extraction à la demande du réseau hydrographique pour limiter la RAM."""
    try:
        grid_obj, catchment, sub_fdir, sub_acc, _ = get_dynamic_catchment(target_lon, target_lat, buffer_deg, threshold)
        stream_mask = ((sub_acc > threshold) & catchment).astype(bool)
        branches = grid_obj.extract_river_network(sub_fdir, stream_mask)
        
        del grid_obj, catchment, sub_fdir, sub_acc
        gc.collect()

        if len(branches["features"]) > 0:
            stream_gdf_raw = gpd.GeoDataFrame.from_features(branches).set_crs("EPSG:4326")
            return stream_gdf_raw
    except Exception as e:
        print(f"[River Extraction] Erreur : {e}")
    return None

def test_remote_raster(raster_path: str):
    t0 = time.perf_counter()
    with rasterio.open(raster_path) as src:
        info = {
            "width": src.width, "height": src.height,
            "crs": str(src.crs), "dtype": src.dtypes[0],
            "driver": src.driver,
        }
        w = min(32, src.width)
        h = min(32, src.height)
        sample = src.read(1, window=((0, h), (0, w)))
        info["sample_shape"] = tuple(sample.shape)
    info["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return info

# --- 3. Configuration & State ---
st.set_page_config(layout="wide", page_title="MadaHydro", page_icon="🇲🇬", initial_sidebar_state="expanded")
APP_VERSION = "HF-COG-STREAM-v3-RAM25K"


def show_recovery_message(exc, context="analyse"):
    """Message utile pour les erreurs récupérables ; les OOM système restent non capturables."""
    msg = str(exc).lower()
    if isinstance(exc, MemoryError) or "out of memory" in msg or "cannot allocate" in msg:
        st.error(
            "⚠️ MadaHydro n’a pas pu terminer cette analyse faute de mémoire disponible. "
            "Réduisez la fenêtre de recherche ou relancez l’analyse après un redémarrage de l’application."
        )
    elif "25,000" in str(exc) or "25 000" in str(exc) or "25000" in msg:
        st.warning(str(exc))
    else:
        st.error(f"⚠️ MadaHydro n’a pas pu terminer la phase {context}. {exc}")


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def create_shapefile_zip_cached(bassin_json: str, reseau_json: str = None):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        with tempfile.TemporaryDirectory() as tmp_dir:
            gdf_bassin = gpd.read_file(io.StringIO(bassin_json))
            gdf_bassin.to_file(os.path.join(tmp_dir, "bassin.shp"), driver="ESRI Shapefile")
            
            if reseau_json:
                gdf_reseau = gpd.read_file(io.StringIO(reseau_json))
                gdf_reseau.to_file(os.path.join(tmp_dir, "reseau.shp"), driver="ESRI Shapefile")
                
            for fname in os.listdir(tmp_dir):
                zf.write(os.path.join(tmp_dir, fname), arcname=fname)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def create_pdf_report_cached(m_data: dict, center_coords: list):
    return create_pdf_report(m_data, center_coords)

def parse_coordinate(coord_str, is_latitude=True):
    if not coord_str or not isinstance(coord_str, str): raise ValueError("Format invalide")
    s = coord_str.strip()
    s = re.sub(r"[’′`‘]", "'", s)         
    s = re.sub(r'[”″“]', '"', s)          
    s = s.replace("''", '"').replace(',', '.')               

    direction = None
    dir_match = re.search(r'(?i)([NSEW])', s)
    if dir_match:
        direction = dir_match.group(1).upper()
        s = re.sub(r'(?i)[NSEW]', '', s)   

    numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', s)
    if not numbers: raise ValueError("Format invalide")

    try:
        first_str = numbers[0]
        is_negative = first_str.startswith('-') or (direction in ['S', 'W'])
        deg = abs(float(first_str))

        if len(numbers) == 1: dd = deg
        elif len(numbers) == 2: dd = deg + (float(numbers[1]) / 60.0)
        else: dd = deg + (float(numbers[1]) / 60.0) + (float(numbers[2]) / 3600.0)

        return -dd if is_negative else dd
    except Exception:
        raise ValueError("Format invalide")

for _key, _default in [
    ("last_coords", None), ("pending_coords", None), ("last_click_key", None),
    ("analysis_requested", False), ("hydro_computed", False), ("catchment_gdf", None),
    ("stream_gdf", None), ("network_attempted", False), ("network_error", None),
    ("metrics", None), ("center_coords", [-18.7669, 46.8691]),
    ("map_center", [-18.7669, 46.8691]), ("map_zoom", 6),
    ("map_bounds", None), ("is_pro", False),
]:
    if _key not in st.session_state: st.session_state[_key] = _default

if st.query_params.get("key", "") in VALID_PRO_KEYS:
    st.session_state.is_pro = True

# --- 4. Interface Utilisateur & CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --mh-blue: #0ea5e9; --mh-blue-dark: #0369a1; --mh-cyan: #06b6d4;
    --mh-green: #10b981; --mh-red: #ef4444; --mh-text: #0f172a;
    --mh-muted: #64748b; --mh-border: rgba(148, 163, 184, 0.25);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.5rem !important; padding-bottom: 2rem; max-width: 1800px; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; padding-bottom: 1.1rem; }
[data-testid="stMetricValue"] { font-size: 1.13rem !important; font-weight: 750 !important; letter-spacing: -0.4px !important; }
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; font-weight: 700 !important; opacity: 0.78; white-space: nowrap !important; }
[data-testid="metric-container"] { padding: 0.25rem 0.15rem !important; }
[data-testid="stExpander"] { border: 1px solid var(--mh-border) !important; border-radius: 14px !important; overflow: hidden; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.035); margin-bottom: 0.7rem; }
[data-testid="stExpander"] summary { font-weight: 700 !important; font-size: 0.92rem !important; }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button { border-radius: 10px !important; font-weight: 700 !important; min-height: 2.55rem; }
[data-testid="stSidebar"] [data-testid="stButton"] button { width: 100%; }
[data-testid="stDataFrame"] { border-radius: 10px; }
h1, h2, h3, h4, h5, h6 { letter-spacing: -0.3px; }

.mh-hero {
    border: 1px solid rgba(14, 165, 233, 0.18); border-radius: 18px;
    padding: 1.0rem 1.15rem 0.9rem 1.15rem; margin: 0.5rem 0 0.95rem 0 !important;
    background: linear-gradient(135deg, rgba(14,165,233,0.08), rgba(6,182,212,0.035) 52%, rgba(255,255,255,0.0));
    overflow: visible !important;
}
.mh-brand-row { display: flex; align-items: center; gap: 0.8rem; }
.mh-logo { width: 44px; height: 44px; border-radius: 13px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #0284c7, #06b6d4); box-shadow: 0 7px 18px rgba(2, 132, 199, 0.18); font-size: 1.4rem; }
.mh-title { font-size: 1.65rem; line-height: 1.3 !important; font-weight: 800; color: var(--mh-text); padding-top: 4px !important; }
.mh-subtitle { margin-top: 0.28rem; font-size: 0.78rem; color: var(--mh-muted); font-weight: 600; }
.mh-header-spacer { flex: 1; }
.mh-pill { font-size: 0.63rem; letter-spacing: 0.9px; font-weight: 800; padding: 0.36rem 0.55rem; border-radius: 999px; color: #0369a1; background: rgba(14,165,233,0.11); border: 1px solid rgba(14,165,233,0.16); }
.mh-description { margin-top: 0.65rem; font-size: 0.79rem; line-height: 1.5; color: #475569; max-width: 980px; }
.mh-side-status { display: flex; align-items: center; gap: 0.65rem; padding: 0.7rem 0.75rem; border-radius: 12px; border: 1px solid var(--mh-border); margin: 0.4rem 0 0.8rem 0; }
.mh-side-status .status-dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }
.mh-side-status.pro { background: rgba(16,185,129,0.07); }
.mh-side-status.pro .status-dot { background: var(--mh-green); box-shadow: 0 0 0 4px rgba(16,185,129,0.10); }
.mh-side-status.free { background: rgba(14,165,233,0.06); }
.mh-side-status.free .status-dot { background: var(--mh-blue); box-shadow: 0 0 0 4px rgba(14,165,233,0.10); }
.mh-side-status strong, .mh-side-status span { display: block; }
.mh-side-status strong { font-size: 0.78rem; }
.mh-side-status span { margin-top: 0.08rem; font-size: 0.66rem; color: var(--mh-muted); }
.mh-section-title { display: flex; gap: 0.58rem; align-items: center; margin: 0.1rem 0 0.55rem 0; }
.mh-section-title > span { width: 33px; height: 33px; display: flex; align-items: center; justify-content: center; border-radius: 10px; background: rgba(14,165,233,0.08); border: 1px solid rgba(14,165,233,0.13); }
.mh-section-title strong, .mh-section-title small { display: block; }
.mh-section-title strong { font-size: 0.92rem; }
.mh-section-title small { font-size: 0.66rem; color: var(--mh-muted); margin-top: 0.04rem; }
.mh-help-wrap { margin-top: 1.2rem; }
@media (max-width: 900px) { .mh-pill { display: none; } .mh-title { font-size: 1.4rem; } }
</style>
<div class="mh-hero">
    <div class="mh-brand-row">
        <div class="mh-logo">MG</div>
        <div>
            <div class="mh-title">Madagascar Watershed Explorer</div>
            <div class="mh-subtitle">Hydrologie appliquée · Madagascar</div>
        </div>
        <div class="mh-header-spacer"></div>
    </div>
    <div class="mh-description">
        Délimitez un bassin versant à partir d'un exutoire, puis explorez ses indicateurs hydrologiques et ses estimations de crue dans un environnement de travail orienté ingénierie.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Barre latérale ---
st.sidebar.markdown("## Navigation")
st.sidebar.caption("Préparez l'exutoire, contrôlez la vue puis lancez l'analyse.")

status_label = "PRO · Analyse complète" if st.session_state.is_pro else "GRATUIT · Délimitation"
status_class = "pro" if st.session_state.is_pro else "free"
st.sidebar.markdown(
    f'<div class="mh-side-status {status_class}"><div class="status-dot"></div>'
    f'<div><strong>{status_label}</strong><span>{"Hydrologie et IA disponibles" if st.session_state.is_pro else "Bassin et géométrie disponibles"}</span></div></div>',
    unsafe_allow_html=True,
)

with st.sidebar.expander("📌 1 · Exutoire", expanded=True):
    input_mode = st.radio(
    "Méthode de sélection",
    ["Saisie manuelle", "Clic sur la carte"],
    index=0,
    horizontal=False,
    label_visibility="collapsed"
)
    target_lat, target_lon = None, None
    do_phase_1 = False

    if input_mode == "Saisie manuelle":
        st.caption("Coordonnées en degrés décimaux, degrés/minutes ou DMS.")
        lat_input = st.text_input("Latitude", value="-18.8792", key="lat_input")
        lon_input = st.text_input("Longitude", value="47.5079", key="lon_input")
        if st.button("🚀 Délimiter le bassin", type="primary", width="stretch"):
            try:
                target_lat = parse_coordinate(lat_input, is_latitude=True)
                target_lon = parse_coordinate(lon_input, is_latitude=False)
                st.session_state.pending_coords = (target_lat, target_lon)
                st.session_state.analysis_requested = True
                st.session_state.network_attempted = False
                st.session_state.network_error = None
            except Exception:
                st.error("Coordonnées invalides.")
    else:
        st.info("Cliquez directement sur l'exutoire souhaité dans la carte.")
        if st.session_state.last_coords is not None:
            lat0, lon0 = st.session_state.last_coords
            st.caption(f"Dernier exutoire : **{lat0:.5f}, {lon0:.5f}**")

with st.sidebar.expander("⚙️ 2 · Paramètres de délimitation", expanded=True):
    st.caption("Ces paramètres contrôlent la fenêtre de recherche et le réseau extrait.")
    buffer_deg = st.slider(
    "Fenêtre initiale (°)",
    min_value=0.30,
    max_value=1.20,
    value=0.40,
    step=0.05,
    help=(
        "Correspondance recommandée selon la surface du bassin :\n\n"
        "• **0.30°** : Petit / Micro-bassin (≤ 100 km²)\n"
        "• **0.50° - 0.60°** : Bassin moyen (100 à 5 000 km²)\n"
        "• **0.70° - 1°** : Grand bassin (5 000 à 20 000 km² max)"
    )
)
    accumulation_threshold = st.slider(
    "Seuil d'accumulation",
    min_value=100,
    max_value=15000,
    value=200,
    step=100,
    help=(
        "Sensibilité d'extraction du cours d'eau :\n\n"
        "• **100 - 500** : Ravines et ruisseaux (S ≤ 100 km²)\n"
        "• **1 000 - 5 000** : Rivières secondaires (100 à 5 000 km²)\n"
        "• **10 000+** : Fleuves principaux (S > 10 000 km²)"
    )
)

with st.sidebar.expander("🗺️ 3 · Affichage de la carte", expanded=False):
    map_basemap = st.selectbox("Fond cartographique", ["HYBRID", "SATELLITE", "ROADMAP", "TERRAIN"], index=0)
    show_catchment = st.checkbox("Afficher le bassin", value=True)
    show_network = st.checkbox("Afficher le réseau hydrographique", value=False, help="Calculé à la demande pour préserver les ressources.")
    map_height = st.select_slider("Hauteur de la carte", options=[620, 680, 740, 800], value=740)

with st.sidebar.expander("📌 4 · Contrôles rapides", expanded=False):
    if st.session_state.last_coords is not None:
        lat0, lon0 = st.session_state.last_coords
        st.metric("Exutoire", f"{lat0:.4f}° / {lon0:.4f}°")
    else:
        st.caption("Aucun exutoire sélectionné.")

    if st.session_state.metrics is not None:
        st.metric("Surface", f"{st.session_state.metrics['area_km2']:,.1f} km²")

    if st.button("↺ Réinitialiser l'analyse", width="stretch"):
        for _k, _v in [
            ("last_coords", None), ("pending_coords", None), ("last_click_key", None),
            ("analysis_requested", False), ("hydro_computed", False),
            ("catchment_gdf", None), ("stream_gdf", None),
            ("network_attempted", False), ("network_error", None), ("metrics", None),
            ("center_coords", [-18.7669, 46.8691]), ("map_center", [-18.7669, 46.8691]),
            ("map_zoom", 6), ("map_bounds", None),
        ]:
            st.session_state[_k] = _v
        gc.collect()
        st.rerun()

    if st.button("🔎 Tester le streaming GDAL", width="stretch", help="Test lit seulement une petite fenêtre de FlowDir distant."):
        try:
            diag = test_remote_raster(FLOW_DIR_PATH)
            st.success(f"GDAL OK · {diag['width']:,}×{diag['height']:,} · échantillon {diag['sample_shape']} · {diag['elapsed_s']} s")
        except Exception as e:
            st.error(f"Test GDAL échoué : {type(e).__name__}: {e}")

    # --- Bloc Diagnostic RAM (à coller à la ligne 752) ---
        st.divider()
        ram_used = psutil.Process().memory_info().rss / (1024 * 1024)
        ram_limit = 2700.0  # Limite Streamlit Cloud (Mo)
        pct_used = (ram_used / ram_limit) * 100

        st.caption("📊 **Utilisation Mémoire RAM**")
        st.metric(
            label="RAM consommée", 
            value=f"{ram_used:.1f} Mo", 
            delta=f"{ram_limit - ram_used:.1f} Mo libres"
        )
        st.progress(min(pct_used / 100, 1.0))

# --- Conteneurs principaux ---
col_map, col_panel = st.columns([1.7, 1.5], gap="large")

with col_map:
    st.markdown('<div class="mh-section-title"><span>🗺️</span><div><strong>Carte d’exploration</strong><small>Délimitation interactive du bassin versant</small></div></div>', unsafe_allow_html=True)
    m = leafmap.Map(center=st.session_state.map_center, zoom=st.session_state.map_zoom)
    try: m.add_basemap(map_basemap)
    except Exception: m.add_basemap("HYBRID")

    if show_catchment and st.session_state.catchment_gdf is not None:
        m.add_gdf(st.session_state.catchment_gdf, layer_name="Bassin versant", style={"color": "#ef4444", "weight": 3, "fillColor": "#ef4444", "fillOpacity": 0.20})
    
    # Calcul à la demande du réseau hydrographique si l'utilisateur coche la case
    if show_network and st.session_state.catchment_gdf is not None and st.session_state.last_coords is not None:
        if st.session_state.stream_gdf is not None and not st.session_state.stream_gdf.empty:
            m.add_gdf(st.session_state.stream_gdf, layer_name="Réseau hydrographique", style={"color": "#06b6d4", "weight": 2})
        elif not st.session_state.network_attempted:
            # Une seule extraction par bassin : un pan/zoom ne relance jamais le calcul.
            with st.spinner("Extraction à la demande du réseau hydrographique..."):
                lon_e, lat_e = st.session_state.last_coords[1], st.session_state.last_coords[0]
                eff_acc = max(accumulation_threshold, int(buffer_deg * 25000))
                try:
                    st.session_state.stream_gdf = extract_river_network_ondemand(lon_e, lat_e, buffer_deg, eff_acc)
                except Exception as e:
                    st.session_state.stream_gdf = None
                    st.session_state.network_error = f"{type(e).__name__}: {e}"
                st.session_state.network_attempted = True
            if st.session_state.stream_gdf is not None and not st.session_state.stream_gdf.empty:
                m.add_gdf(st.session_state.stream_gdf, layer_name="Réseau hydrographique", style={"color": "#06b6d4", "weight": 2})
        
    if st.session_state.map_bounds is not None:
        m.fit_bounds(st.session_state.map_bounds)
        st.session_state.map_bounds = None 

    # IMPORTANT : seuls les clics doivent remonter à Streamlit.
    # En limitant returned_objects à "last_clicked", les événements Leaflet
    # de pan/zoom (center, zoom, bounds) ne modifient pas la valeur du composant
    # et ne provoquent donc pas de rerun Streamlit.
    map_data = st_folium(
    m,
    height=map_height,
    width="100%",
    key="madagascar_map",
    returned_objects=["last_clicked"],
    return_on_hover=False,
)
    st.caption("💡 Cliquez sur un exutoire pour lancer la délimitation. Le pan et le zoom n’interrompent pas le calcul.")

# Seul un nouveau clic sur la carte crée une demande de délimitation.
# Le pan/zoom n'est volontairement pas lu ici : il ne provoque aucun rerun
# avec returned_objects=["last_clicked"].

# Seul un nouveau clic sur la carte crée une demande de délimitation.
# Le pan/zoom n'est volontairement pas lu ici : il ne provoque aucun rerun
# avec returned_objects=["last_clicked"].
if map_data and map_data.get("last_clicked"):
    lat = float(map_data["last_clicked"]["lat"])
    lon = float(map_data["last_clicked"]["lng"])
    click_key = (round(lat, 7), round(lon, 7))

    if input_mode == "Clic sur la carte":
        # Déclenchement uniquement s'il s'agit d'un VÉRITABLE nouveau clic
        if st.session_state.get("last_click_key") != click_key:
            st.session_state.last_click_key = click_key
            st.session_state.last_coords = (lat, lon)
            st.session_state.pending_coords = (lat, lon)
            st.session_state.analysis_requested = True
            st.session_state.network_attempted = False
            st.session_state.network_error = None
    else:
        # En mode Saisie manuelle : mémoriser la clé du clic sans activer analysis_requested
        st.session_state.last_click_key = click_key

with col_panel:
    st.markdown('<div class="mh-section-title"><span>📊</span><div><strong>Analyse</strong><small>Résultats et indicateurs</small></div></div>', unsafe_allow_html=True)

    if st.session_state.metrics is None:
        with st.container(border=True):
            st.write("La plateforme calculera automatiquement la géométrie du bassin à partir du Modèle Numérique de Surface.")
    else:
        m_data = st.session_state.metrics
        with st.container(border=True):
            st.markdown("#### 📐 Bassin Versant")
            c1, c2 = st.columns(2)
            c1.metric("Surface", f"{m_data['area_km2']:,.1f} km²")
            c2.metric("Périmètre", f"{m_data['perimeter_km']:,.1f} km")
            if st.session_state.last_coords is not None:
                lat0, lon0 = st.session_state.last_coords
                st.caption(f"Exutoire : **{lat0:.5f}°, {lon0:.5f}°**")
            if st.session_state.catchment_gdf is not None:
                bassin_json = st.session_state.catchment_gdf.to_json()
                reseau_json = st.session_state.stream_gdf.to_json() if st.session_state.stream_gdf is not None else None
                zip_bytes = create_shapefile_zip_cached(bassin_json, reseau_json)
                st.download_button("📦 Télécharger les couches.shp SIG (.ZIP)", zip_bytes, f"bassin_{st.session_state.center_coords[0]:.2f}.zip", "application/zip", width="stretch")

        if st.session_state.is_pro:
            if st.session_state.get("hydro_computed", False):
                with st.expander("🌍 Caractéristiques du bassin", expanded=False):
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Alt. min.", f"{m_data['min_elev']:.0f} m")
                    sc2.metric("Alt. max.", f"{m_data['max_elev']:.0f} m")
                    sc3.metric("Pente globale", f"{m_data['slope_m_km']:.1f} m/km")

                    sc4, sc5 = st.columns(2)
                    sc4.metric("Rectangle équiv.", f"{m_data['l_rect_km']:.1f} km")
                    sc5.metric("Gravelius Kc", f"{m_data['kc']:.2f}")

                    st.markdown("**Occupation du sol**")
                    if not m_data.get("lc_df").empty:
                        st.dataframe(m_data["lc_df"].style.format({"Surface (km²)": "{:.1f}", "Proportion (%)": "{:.1f}"}), hide_index=True, width="stretch")
                    else: st.caption("Aucune donnée d'occupation du sol disponible.")

                with st.expander("🌧️ Pluviométrie", expanded=False):
                    st.metric("Pluie moyenne annuelle", f"{m_data['p_annuelle_mm']:,.0f} mm/an", help="Moyenne spatiale.")
                    st.markdown("**Distribution mensuelle typique**")
                    df_m = pd.DataFrame([m_data["p_mensuelles"]], columns=["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sept", "Oct", "Nov", "Déc"])
                    st.dataframe(df_m.style.format("{:.1f}"), hide_index=True, width="stretch")
                    st.markdown("**Pluie max 24h ajustée par la loi de Gumbel**")
                    
                    cp1, cp2 = st.columns(2)
                    cp1.metric("P10", f"{m_data['p_design'][10]:,.0f} mm")
                    cp2.metric("P25", f"{m_data['p_design'][25]:,.0f} mm")
                    cp3, cp4 = st.columns(2)
                    cp3.metric("P50", f"{m_data['p_design'][50]:,.0f} mm")
                    cp4.metric("P100", f"{m_data['p_design'][100]:,.0f} mm")

                with st.expander("⚡ Débits de crue de projet (Q_T)· Modélisation", expanded=False):
                    in_domain = m_data.get("in_domain", False)
                    q_ml = m_data.get("q_dict")
                    st.markdown("**Prédicteur multivarié par apprentissage supervisé (Gradient Boosting)**")
                    
                    if in_domain and q_ml is not None:
                        st.success(f"🤖 Modèle d'apprentissage automatique actif · Versant : {m_data['versant']}")
                        
                        cq1, cq2 = st.columns(2)
                        cq1.metric("Q10", f"{q_ml[10]:,.1f} m³/s")
                        cq2.metric("Q25", f"{q_ml[25]:,.1f} m³/s")
                        cq3, cq4 = st.columns(2)
                        cq3.metric("Q50", f"{q_ml[50]:,.1f} m³/s")
                        cq4.metric("Q100", f"{q_ml[100]:,.1f} m³/s")
                    else:
                        st.warning("⚠️ Modèle ML indisponible. Vérifiez la connectivité réseau.")
                        cq1, cq2 = st.columns(2)
                        cq1.metric("Q10", "—")
                        cq2.metric("Q25", "—")
                        cq3, cq4 = st.columns(2)
                        cq3.metric("Q50", "—")
                        cq4.metric("Q100", "—")

                    st.markdown("**Variables d'entrée du modèle**")
                    egv = m_data.get("egv_params", {})
                    cpa, cpb, cpc = st.columns(3)
                    cpa.metric("Exondation E", f"{egv.get('E', '—')}"); cpb.metric("Imperméabilité G", f"{egv.get('G', '—')}"); cpc.metric("Végétation V", f"{egv.get('V', '—')}")
                    cpd, cpe, cpf = st.columns(3)
                    cpd.metric("Pente I", f"{egv.get('Ig_m_km', '—')} m/km"); cpe.metric("Pluie P10", f"{egv.get('P10_mm', '—')} mm"); cpf.metric("Surface S", f"{egv.get('Surface_km2', '—')} km²")

                with st.expander("📚 Sources des Données & Modélisation", expanded=False):
                    st.markdown("* **MNT & Altimétrie** · Copernicus / FABDEM 30m\n* **Occupation du sol** · ESA WorldCover v200 (10m)\n* **Pluviométrie** · CHIRPS v2.0 calibré avec 21 stations pluvio à Mada \n* **Intelligence Artificielle** · Gradient boosting regressor")

                m_data['slope_pct'] = m_data['slope_m_m'] * 100.0
                pdf_bytes = create_pdf_report_cached(m_data, st.session_state.center_coords)
                st.download_button("📄 Télécharger le rapport complet (.PDF)", pdf_bytes, f"Rapport_Hydro_{st.session_state.center_coords[0]:.2f}.pdf", "application/pdf", type="primary", width="stretch")
            else:
                st.info("🔄 Profilage hydrologique en cours…")
        else:
            with st.container(border=True):
                st.write("La version gratuite permet de **tracer et contrôler le bassin versant**. Activez PRO pour accéder au profilage hydrologique complet.")
                feature_cols = st.columns(2)
                feature_cols[0].markdown("✓ Altitudes & pente\n\n✓ Géométrie complet\n\n✓ Occupation du sol")
                feature_cols[1].markdown("✓ Pluviométrie\n\n✓ Débits de crue IA\n\n✓ Rapport PDF")

                with st.expander("💳 Activer la fonctionalité complet", expanded=False):
                    st.markdown("**Accès pour projets académiques, bureaux d'études et applications professionnelles.**")
                    st.markdown("💬 [Contacter MadaHydro](https://api.whatsapp.com/send/?phone=327829333)")
                    st.markdown("✉️ [Envoyer un e-mail](https://mail.google.com/mail/?view=cm&fs=1&to=joffrerazafimihary@gmail.com&su=Demande%20d%27acces%20MadaHydro)")
                    code_input = st.text_input("Code d'accès", key="unlock_code_panel")
                    if st.button("🔓 Accèder", type="primary", width="stretch", key="activate_panel"):
                        if code_input.strip() in VALID_PRO_KEYS:
                            st.session_state.is_pro = True
                            st.success("Licence PRO activée avec succès.")
                            st.rerun()
                        else: st.error("Code d'accès invalide ou expiré.")

st.markdown('<div class="mh-help-wrap">', unsafe_allow_html=True)
with st.expander("📖 Guide d'utilisation", expanded=False):
    st.markdown("""
    ### Utilisation en 4 étapes

    **1 · Choisir l'exutoire**  
    Cliquez sur la carte ou saisissez les coordonnées dans la barre latérale. Les formats décimaux, degrés/minutes et DMS sont pris en charge.

    **2 · Régler la délimitation**  
    Ajustez la fenêtre de recherche et le seuil d'accumulation uniquement lorsque nécessaire.

    **3 · Examiner les résultats**  
    La version gratuite présente la géométrie du bassin. Le mode PRO ouvre le profilage morphologique, l'occupation du sol, la pluviométrie et les estimations de crue par IA.

    **4 · Exporter**  
    Téléchargez les couches SIG et, en mode PRO, le rapport d'étude complet au format PDF.
    """)
st.markdown('</div>', unsafe_allow_html=True)

# --- 5. Pipelines d'Exécution Optimisés ---
do_phase_1 = bool(st.session_state.get("analysis_requested", False) and st.session_state.get("pending_coords"))
if do_phase_1:
    try:
        t_phase1_start = time.perf_counter()
        with st.spinner("1/2 Délimitation et extraction géométrique..."):
            target_lat, target_lon = st.session_state.pending_coords
            effective_acc = max(accumulation_threshold, int(buffer_deg * 25000))
            features = delineate_catchment_geometry_cached(
                round(float(target_lon), 6),
                round(float(target_lat), 6),
                round(float(buffer_deg), 2),
                int(effective_acc),
            )
            target_epsg = get_madagascar_utm_epsg(target_lon)
            gc.collect()

            if features:
                gdf_raw = gpd.GeoDataFrame.from_features(features).set_crs("EPSG:4326")
                gdf_proj = gdf_raw.to_crs(epsg=target_epsg)
                gdf_proj["geometry"] = gdf_proj.geometry.simplify(tolerance=50.0, preserve_topology=True)

                area_km2 = float(gdf_proj.geometry.area.sum() / 1e6)

                if area_km2 > MAX_CATCHMENT_AREA_KM2:
                    raise ValueError(
                        f"Le bassin délimité fait {area_km2:,.0f} km², au-delà de la "
                        f"limite de {MAX_CATCHMENT_AREA_KM2:,.0f} km² fixée pour protéger "
                        "les ressources de l'application."
                    )

                perimeter_km = float(gdf_proj.geometry.length.sum() / 1000.0)
                gdf = gdf_proj.to_crs(epsg=4326)

                kc = 0.28 * perimeter_km / np.sqrt(area_km2)
                if kc >= 1.12: l_rect_km = (kc * np.sqrt(area_km2) / 1.12) * (1.0 + np.sqrt(max(1.0 - (1.12 / kc) ** 2, 0.0)))
                else: l_rect_km = np.sqrt(area_km2)

                minx, miny, maxx, maxy = gdf.total_bounds
                dx, dy = maxx - minx, maxy - miny
                st.session_state.map_bounds = [[miny - dy * 0.25, minx - dx * 0.25], [maxy + dy * 0.25, maxx + dx * 0.25]]

                # Le mode FREE ne lit aucun DEM : valeurs par défaut légères
                min_elev = max_elev = z5_m = z95_m = 0.0
                slope_m_km = 0.5
                slope_m_m = slope_m_km / 1000.0

            st.session_state.catchment_gdf = gdf
            st.session_state.stream_gdf = None  # Calculé uniquement à la demande si demandé
            st.session_state.metrics = {
                "area_km2": area_km2, "perimeter_km": perimeter_km,
                "min_elev": min_elev, "max_elev": max_elev,
                "slope_m_km": round(slope_m_km, 2), "slope_m_m": slope_m_m,
                "kc": round(kc, 3), "l_rect_km": round(l_rect_km, 2),
                "z5_m": round(z5_m, 1), "z95_m": round(z95_m, 1),
                "main_channel_len_km": round(l_rect_km, 2),
                "target_epsg": target_epsg,
            }
            st.session_state.last_coords = (target_lat, target_lon)
            st.session_state.last_click_key = (round(target_lat, 7), round(target_lon, 7))
            st.session_state.pending_coords = None
            st.session_state.analysis_requested = False
            st.session_state.network_attempted = False
            st.session_state.network_error = None
            st.session_state.stream_gdf = None
            st.session_state.hydro_computed = False
            st.session_state.center_coords = [(miny + maxy) / 2.0, (minx + maxx) / 2.0]
            st.session_state.map_center = list(st.session_state.center_coords)
            
            t_elapsed = time.perf_counter() - t_phase1_start
            print(f"[CHRONO] Phase 1 terminée en {t_elapsed:.2f} s")
            
            st.rerun()

    except Exception as e:
        show_recovery_message(e, "de délimitation")
        st.session_state.analysis_requested = False
        st.session_state.pending_coords = None

if (st.session_state.is_pro and st.session_state.catchment_gdf is not None and not st.session_state.get("hydro_computed", False)):
    try:
        t_phase2_start = time.perf_counter()
        with st.spinner("2/2 Lecture des Rasters distants & IA..."):
            m_data = st.session_state.metrics

            # Lecture du DEM par fenêtre HTTP Range limitée
            with rasterio.open(DEM_PATH) as dem_src:
                dem_proj_gdf = st.session_state.catchment_gdf.to_crs(dem_src.crs)
                dem_shapes = [geom for geom in dem_proj_gdf.geometry if geom is not None and not geom.is_empty]
                dem_win = geometry_window(dem_src, dem_shapes, pad_x=0, pad_y=0)
                dem_h = min(512, max(1, int(dem_win.height)))
                dem_w = min(512, max(1, int(dem_win.width)))
                
                dem_arr = dem_src.read(
                    1,
                    window=dem_win,
                    out_shape=(dem_h, dem_w),
                    resampling=rasterio.enums.Resampling.average,
                    masked=False,
                ).astype(np.float32, copy=False)
                dem_transform = window_transform(dem_win, dem_src.transform) * Affine.scale(
                    dem_win.width / dem_w, dem_win.height / dem_h
                )
                dem_inside = geometry_mask(
                    dem_shapes,
                    out_shape=(dem_h, dem_w),
                    transform=dem_transform,
                    invert=True,
                )
                dem_nodata = dem_src.nodata
                valid_mask = dem_inside & np.isfinite(dem_arr)
                if dem_nodata is not None:
                    valid_mask &= dem_arr != dem_nodata
                valid_elevs = dem_arr[valid_mask]
                valid_elevs = valid_elevs[valid_elevs > -50]

            if valid_elevs.size > 0:
                pos_elevs = valid_elevs[valid_elevs > 0]
                min_elev = float(np.min(pos_elevs)) if pos_elevs.size else 0.0
                max_elev = float(np.max(valid_elevs))
                z5_m = float(np.percentile(valid_elevs, 95))
                z95_m = float(np.percentile(valid_elevs, 5))
                slope_m_km = max((z5_m - z95_m) / max(m_data["l_rect_km"], 0.1), 0.5)
            else:
                min_elev = max_elev = z5_m = z95_m = 0.0
                slope_m_km = 0.5

            m_data.update({
                "min_elev": min_elev, "max_elev": max_elev,
                "slope_m_km": round(slope_m_km, 2),
                "slope_m_m": slope_m_km / 1000.0,
                "z5_m": round(z5_m, 1), "z95_m": round(z95_m, 1),
            })

            # Correction Passini Tc
            tc_passini = compute_passini_tc(m_data["area_km2"], m_data["l_rect_km"], slope_m_km)
            m_data["tc_passini_hours"] = tc_passini

            # Libération explicite de tous les tableaux DEM temporaires.
            try:
                del dem_arr
            except NameError:
                pass
            try:
                del dem_inside
            except NameError:
                pass
            try:
                del valid_mask
            except NameError:
                pass
            try:
                del valid_elevs
            except NameError:
                pass
            try:
                del pos_elevs
            except NameError:
                pass
            gc.collect()

            catchment_json = st.session_state.catchment_gdf.to_json()
            p_annuelle_mm, p_mensuelles, p_design = calculate_local_rainfall_cached(catchment_json)
            
            lat_exu, lon_exu = st.session_state.last_coords
            versant = detect_versant(lat_exu, lon_exu)
            in_domain = True

            e_raw, g_raw, v_raw, pct_wet, lc_ok, df_lc = extract_local_landcover_cached(catchment_json, m_data["area_km2"])
            g_cal, v_cal, e_cal = calibrer_indices_egv(g_raw, v_raw, e_raw, pct_wet, m_data["area_km2"], versant)
            
            ml_model, ml_loaded = get_q10_model()
            q_dict = compute_q10_ml(ml_model, m_data["area_km2"], p_design[10], slope_m_km, e_cal, g_cal, v_cal, p_design) if (ml_loaded and ml_model is not None) else None

            st.session_state.metrics.update({
                "p_annuelle_mm": p_annuelle_mm, "p_mensuelles": p_mensuelles, "p_design": p_design,
                "q_dict": q_dict,
                "versant": versant, "in_domain": in_domain, "lc_df": df_lc,
                "egv_params": {"E": round(e_cal, 3), "G": round(g_cal, 3), "V": round(v_cal, 3), "Ig_m_km": m_data["slope_m_km"], "P10_mm": round(p_design[10], 1), "Surface_km2": round(m_data["area_km2"], 2)},
            })
            st.session_state.hydro_computed = True

            t_elapsed_p2 = time.perf_counter() - t_phase2_start
            print(f"[CHRONO] Phase 2 terminée en {t_elapsed_p2:.2f} s")

            st.rerun()
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la lecture des ressources distantes : {type(e).__name__}: {e}")
        