import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from PIL import Image
import requests
import plotly.graph_objects as pgo
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False
import gdown
from ultralytics import YOLO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Forelectra | NEOM Energy Forecast",
    layout="wide"
)


# =========================
# LANGUAGE
# =========================
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

def tr(en, ar):
    return ar if st.session_state.lang == "AR" else en

def label(en, ar):
    return f"{en}<br><small style='color:#AFA7D8'>{ar}</small>" if st.session_state.lang == "EN" else f"{ar}<br><small style='color:#AFA7D8'>{en}</small>"


# =========================
# CSS
# =========================
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(123, 44, 191, 0.28), transparent 35%),
        radial-gradient(circle at top right, rgba(0, 255, 255, 0.12), transparent 30%),
        linear-gradient(135deg, #050816 0%, #090B1F 45%, #12051F 100%);
    color: white;
}
.block-container {
    padding-top: 4rem;
    padding-bottom: 4rem;
    max-width: 1500px;
}
.main-title {
    text-align:center;
    font-size:52px;
    font-weight:900;
    color:white;
    text-shadow:0 0 25px rgba(138,92,246,.55);
}
.subtitle {
    text-align:center;
    font-size:19px;
    color:#C9C3E8;
    margin-bottom:28px;
}
.glass-card {
    background:rgba(12,16,38,.78);
    border:1px solid rgba(180,130,255,.24);
    border-radius:26px;
    padding:28px;
    box-shadow:0 18px 45px rgba(0,0,0,.35);
    backdrop-filter:blur(14px);
}
.method-card {
    background:rgba(13,17,42,.82);
    border:1px solid rgba(180,130,255,.28);
    border-radius:24px;
    padding:30px;
    min-height:190px;
    box-shadow:0 12px 35px rgba(0,0,0,.32);
}
.card-title {
    font-size:25px;
    font-weight:850;
}
.card-text {
    color:#CFC8EF;
    font-size:16px;
}
div.stButton > button {
    width:100%;
    border-radius:16px;
    min-height:56px;
    background:linear-gradient(90deg,#6D28D9,#9333EA);
    color:white;
    border:1px solid rgba(255,255,255,.18);
    font-weight:800;
}
.metric-card {
    background:rgba(8,12,30,.78);
    border:1px solid rgba(124,58,237,.35);
    border-radius:20px;
    padding:22px;
    text-align:center;
    box-shadow:0 0 25px rgba(124,58,237,.22);
}
.metric-icon {
    font-size:38px;
}
.metric-title {
    color:#BEB8E8;
    font-size:14px;
}
.metric-value {
    font-size:26px;
    font-weight:900;
    color:#7CFF9B;
}
.small-ar {
    color:#AFA7D8;
    font-size:12px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# SESSION
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "features" not in st.session_state:
    st.session_state.features = {}

def go(page):
    st.session_state.page = page
    st.rerun()


# =========================
# MODEL PATHS
# =========================
MODEL_DIR = "."

PATHS = {
    "vehicle": os.path.join(MODEL_DIR, "vehicle_detection_model.pt"),
    "crowd": os.path.join(MODEL_DIR, "crowd_count_model.h5"),
    "land_cover": os.path.join(MODEL_DIR, "land_cover_model.keras"),
    "land_cover_classes": os.path.join(MODEL_DIR, "land_cover_class_names.pkl"),
    "land_use": os.path.join(MODEL_DIR, "land_use_cnn.h5"),
    "weather": os.path.join(MODEL_DIR, "weather_cnn_model.h5"),
    "area_type": os.path.join(MODEL_DIR, "area_type_model.h5"),
}

# =========================
# DOWNLOAD LARGE MODEL
# =========================

LAND_USE_FILE_ID = "1c1f4l27Yp-2sni-UTU4-OjfXjfWr6MIu"

if not os.path.exists(PATHS["land_use"]):
    os.makedirs(MODEL_DIR, exist_ok=True)

    gdown.download(
        f"https://drive.google.com/uc?id={LAND_USE_FILE_ID}",
        PATHS["land_use"],
        quiet=False
    )

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    models = {}

    if os.path.exists(PATHS["vehicle"]):
        models["vehicle"] = YOLO(PATHS["vehicle"])

    if TF_AVAILABLE:
        for key in ["crowd", "land_cover", "land_use", "weather", "area_type"]:
            if os.path.exists(PATHS[key]):
                models[key] = load_model(PATHS[key], compile=False)
        if os.path.exists(PATHS[key]):
            models[key] = load_model(PATHS[key], compile=False)

    if os.path.exists(PATHS["land_cover_classes"]):
        models["land_cover_classes"] = joblib.load(PATHS["land_cover_classes"])

    return models

models = load_models()

# =========================
# HELPERS
# =========================
def metric_card(icon, title_en, title_ar, value, color="#7CFF9B"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-title">{title_en}</div>
        <div class="small-ar">{title_ar}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def preprocess_for_model(uploaded_img, model):
    img = Image.open(uploaded_img).convert("RGB")

    input_shape = model.input_shape
    h = input_shape[1]
    w = input_shape[2]

    img_resized = img.resize((w, h))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    return arr

def density_from_count(count, low=8, medium=18):
    if count <= low:
        return "LOW"
    elif count <= medium:
        return "MEDIUM"
    return "HIGH"


def green_density_mapping(pred):
    if pred == "forest_land":
        return "HIGH"
    elif pred in ["agriculture_land", "rangeland"]:
        return "MEDIUM"
    return "LOW"

LAND_COVER_DISPLAY = {
    "urban_land": "Urban Area",
    "agriculture_land": "Agricultural Area",
    "rangeland": "Open Natural Land",
    "forest_land": "Forest / Green Area",
    "water": "Water Body",
    "barren_land": "Barren Land",
    "unknown": "Unknown Area"
}

LAND_COVER_DISPLAY_AR = {
    "urban_land": "منطقة حضارية",
    "agriculture_land": "منطقة زراعية",
    "rangeland": "أرض طبيعية مفتوحة",
    "forest_land": "منطقة خضراء",
    "water": "مسطح مائي",
    "barren_land": "أرض جافة",
    "unknown": "منطقة غير معروفة"
}

def auto_time_features():
    now = datetime.now()
    return {
        "Hour": now.hour,
        "DayOfWeek": now.weekday(),
        "Month": now.month,
        "IsWeekend": 1 if now.weekday() >= 5 else 0,
        "IsHoliday": 0,
        "Season": "Summer" if now.month in [6,7,8] else "Winter" if now.month in [12,1,2] else "Spring/Fall"
    }

def get_live_weather(lat, lon):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,wind_speed_10m,precipitation,cloud_cover"
        )

        data = requests.get(url, timeout=10).json()
        current = data["current"]

        temp = current.get("temperature_2m", 30)
        wind = current.get("wind_speed_10m", 10)
        rain = current.get("precipitation", 0)
        clouds = current.get("cloud_cover", 0)

        if rain > 0:
            weather = "rainy"
        elif clouds > 60:
            weather = "cloudy"
        else:
            weather = "clear"

        return {
            "Weather": weather,
            "Temperature": temp,
            "Wind Speed": wind,
            "Rainfall": rain,
            "Cloud Cover": clouds
        }

    except:
        return {
            "Weather": "clear",
            "Temperature": 30,
            "Wind Speed": 10,
            "Rainfall": 0,
            "Cloud Cover": 20
        }

def estimate_live_traffic(zone_name, hour):
    busy_zones = ["THE LINE", "OXAGON", "NEOM BAY", "SINDALAH"]

    if zone_name in busy_zones and (7 <= hour <= 10 or 16 <= hour <= 21):
        return "HIGH"
    elif zone_name in busy_zones:
        return "MEDIUM"
    else:
        return "LOW"

def image_models_predict(uploaded_img):
    img = Image.open(uploaded_img).convert("RGB")
    outputs = {}

    # Land Cover
    if "land_cover" in models and "land_cover_classes" in models:
        arr_land = preprocess_for_model(uploaded_img, models["land_cover"])
        pred = models["land_cover"].predict(arr_land, verbose=0)

        print("Land Cover Raw =", pred)
        print("Land Cover Index =", np.argmax(pred))

        idx = int(np.argmax(pred))
        land_class = models["land_cover_classes"][idx]

        outputs["Land Cover Raw"] = land_class
        outputs["Land Cover"] = LAND_COVER_DISPLAY.get(land_class, land_class)
        outputs["Land Cover AR"] = LAND_COVER_DISPLAY_AR.get(land_class, land_class)
        outputs["Green Density"] = green_density_mapping(land_class)
        outputs["Land Cover Confidence"] = round(float(np.max(pred)) * 100, 2)

    # Land Use
    if "land_use" in models:
        arr_land_use = preprocess_for_model(uploaded_img, models["land_use"])
        pred = models["land_use"].predict(arr_land_use, verbose=0)

        print("Land Use Raw =", pred)
        print("Land Use Index =", np.argmax(pred))

        land_use_classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
        idx = int(np.argmax(pred))

        outputs["Land Use"] = land_use_classes[idx] if idx < len(land_use_classes) else "buildings"
        outputs["Land Use Confidence"] = round(float(np.max(pred)) * 100, 2)

    # Weather
    if "weather" in models:
        arr_weather = preprocess_for_model(uploaded_img, models["weather"])
        pred = models["weather"].predict(arr_weather, verbose=0)

        print("Weather Raw =", pred)
        print("Weather Index =", np.argmax(pred))

        idx = int(np.argmax(pred))
        weather_classes = ["cloudy", "rainy", "shine", "sunrise"]

        weather_result = weather_classes[idx] if idx < len(weather_classes) else "shine"

        img_np = np.array(img)
        brightness = img_np.mean()
        blue_channel = img_np[:, :, 2].mean()

        if brightness < 65:
            weather_result = "night"
        elif weather_result == "rainy" and brightness > 140 and blue_channel > 110:
            weather_result = "shine"

        outputs["Weather"] = weather_result
        outputs["Weather Confidence"] = round(float(np.max(pred)) * 100, 2)

    # Area Type
    if "area_type" in models:
        arr_area = preprocess_for_model(uploaded_img, models["area_type"])
        pred = models["area_type"].predict(arr_area, verbose=0)

        idx = int(np.argmax(pred))
        area_classes = ["Rural", "Suburban", "Urban"]
        area_result = area_classes[idx] if idx < len(area_classes) else "Urban"

        if outputs.get("Land Cover Raw") == "urban_land" or outputs.get("Land Use") in ["buildings", "street"]:
            area_result = "Urban"

        outputs["Area Type"] = area_result
        outputs["Area Confidence"] = round(float(np.max(pred)) * 100, 2)

    # Vehicle Detection
    if "vehicle" in models:
        temp_path = "temp_uploaded_image.jpg"
        img.save(temp_path)

        result = models["vehicle"].predict(
            source=temp_path,
            conf=0.25,
            save=False,
            verbose=False
        )

        vehicle_count = 0

        if hasattr(result[0], "obb") and result[0].obb is not None:
            vehicle_count = len(result[0].obb)
        elif hasattr(result[0], "boxes") and result[0].boxes is not None:
            vehicle_count = len(result[0].boxes)

        outputs["Vehicle Count"] = vehicle_count
        outputs["Vehicle Density"] = density_from_count(vehicle_count)

    # Crowd Model
    if "crowd" in models:
        arr_crowd = preprocess_for_model(uploaded_img, models["crowd"])
        pred = models["crowd"].predict(arr_crowd, verbose=0)

        crowd_value = float(np.max(pred))
        crowd_count = int(crowd_value * 100) if crowd_value <= 1 else int(crowd_value)
        print("Crowd Count=",crowd_count)

        outputs["Crowd Count"] = crowd_count
        outputs["Crowd Density"] = density_from_count(crowd_count, low=20, medium=60)
        if outputs.get("Area Type") == "Urban" and outputs.get("Vehicle Density") == "HIGH":
            outputs["Crowd Density"] = "HIGH"
        elif outputs.get("Area Type") == "Urban" and outputs.get("Vehicle Density") == "MEDIUM":
            outputs["Crowd Density"] = "MEDIUM"

    return img, outputs

def feature_engineering(base):
    time_f = auto_time_features()
    base.update(time_f)

    traffic = base.get("Vehicle Density", "MEDIUM")
    crowd = base.get("Crowd Density", "MEDIUM")
    green = base.get("Green Density", "MEDIUM")
    weather = base.get("Weather", "clear")
    sector = base.get("Load Sector", "Commercial")
    area = base.get("Area Type", "Urban")

    # علاقة المركبات + الأشخاص = ضغط تشغيل المنطقة
    activity_score = 0
    activity_score += {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(traffic, 2)
    activity_score += {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(crowd, 2)

    base["Energy Status"] = "Event" if activity_score >= 5 else "No Event"
    base["Current Load Level"] = "Peak" if activity_score >= 5 or sector in ["Industrial", "Commercial"] else "Normal"

    # قيم تقديرية بدل ما المستخدم يتعب
    base["Current Level (A)"] = 180 if base["Current Load Level"] == "Peak" else 95

    base["Solar PV Output (kW)"] = (
        85 if weather == "clear" else
        60 if weather == "cloudy" else
        35
    )

    base["Wind Power Output (kW)"] = (
        70 if area in ["Coastal", "Mountain", "Rural"] else
        35
    )

    # تأثير الخضرة
    base["Cooling Demand Factor"] = (
        0.85 if green == "HIGH" else
        1.0 if green == "MEDIUM" else
        1.15
    )

    return base


def simple_load_forecast(features):
    base_load = {
        "Residential": 520,
        "Commercial": 760,
        "Industrial": 1100
    }.get(features.get("Load Sector", "Commercial"), 760)

    area_factor = {
        "Urban": 1.18,
        "Suburban": 1.00,
        "Rural": 0.82,
        "Coastal": 1.05,
        "Mountain": 0.90
    }.get(features.get("Area Type", "Urban"), 1.0)

    peak_factor = 1.22 if features.get("Current Load Level") == "Peak" else 1.0
    event_factor = 1.15 if features.get("Energy Status") == "Event" else 1.0
    traffic_factor = {"LOW": 0.95, "MEDIUM": 1.05, "HIGH": 1.18}.get(features.get("Vehicle Density", "MEDIUM"), 1.05)
    crowd_factor = {"LOW": 0.95, "MEDIUM": 1.05, "HIGH": 1.16}.get(features.get("Crowd Density", "MEDIUM"), 1.05)
    cooling_factor = features.get("Cooling Demand Factor", 1.0)

    renewable_offset = (features.get("Solar PV Output (kW)", 50) + features.get("Wind Power Output (kW)", 30)) * 0.18

    predicted = base_load * area_factor * peak_factor * event_factor * traffic_factor * crowd_factor * cooling_factor
    predicted = predicted - renewable_offset

    return round(max(predicted, 0), 2)


def recommendations(features, predicted):
    tips = []

    if features.get("Vehicle Density") == "HIGH":
        tips.append("High traffic detected: increase grid monitoring around mobility corridors.")
    if features.get("Crowd Density") == "HIGH":
        tips.append("High crowd density: prepare extra energy allocation for public services.")
    if features.get("Green Density") == "LOW":
        tips.append("Low green coverage may increase cooling demand; optimize HVAC scheduling.")
    if features.get("Solar PV Output (kW)", 0) > 70:
        tips.append("Solar output is strong: shift non-critical loads to daylight hours.")
    if predicted > 1000:
        tips.append("Predicted load is high: activate peak-load management strategy.")
    if not tips:
        tips.append("Energy status looks stable. Continue normal monitoring.")

    return tips


# =========================
# HOME
# =========================
def home_page():
    top1, top2 = st.columns([5,1])
    with top2:
        if st.button("🌐 عربي / EN"):
            st.session_state.lang = "AR" if st.session_state.lang == "EN" else "EN"
            st.rerun()

    st.markdown(f"<div class='main-title'>Forelectra ⚡</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{tr('AI-powered electricity load forecasting for NEOM smart energy systems', 'نظام ذكي للتنبؤ بالحمل الكهربائي في نيوم')}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class='method-card'>
            <div class='card-title'>🗺 {tr('Interactive NEOM Map', 'خريطة نيوم التفاعلية')}</div>
            <div class='card-text'>{tr('Select a district and auto-fill scenario features.', 'اختاري منطقة وسيتم تعبئة البيانات تلقائياً.')}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(tr("Open Map", "فتح الخريطة")):
            go("map")

    with c2:
        st.markdown(f"""
        <div class='method-card'>
            <div class='card-title'>📸 {tr('AI Image Analysis', 'تحليل الصورة بالذكاء الاصطناعي')}</div>
            <div class='card-text'>{tr('Upload an image and extract features using six AI models.', 'ارفعي صورة وسيتم استخراج الخصائص من المودلات.')}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(tr("Analyze Image", "تحليل صورة")):
            go("image")

    with c3:
        st.markdown(f"""
        <div class='method-card'>
            <div class='card-title'>✍️ {tr('Manual Scenario', 'إدخال يدوي')}</div>
            <div class='card-text'>{tr('Enter all scenario inputs manually.', 'إدخال جميع بيانات السيناريو يدوياً.')}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(tr("Manual Input", "إدخال يدوي")):
            go("manual")


# =========================
# MAP PAGE
# =========================
def map_page():
    st.markdown(f"<div class='main-title'>{tr('Interactive NEOM Smart Map 🗺️', 'خريطة نيوم الذكية 🗺️')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{tr('Choose a district to auto-fill the forecast inputs', 'اختاري منطقة لتعبئة بيانات التنبؤ تلقائياً')}</div>", unsafe_allow_html=True)

    zones = {
        "THE LINE": {"lat":28.1036, "lon":35.1995, "Area Type":"Urban", "Load Sector":"Residential", "Weather":"clear", "Green Density":"MEDIUM", "Vehicle Density":"MEDIUM", "Crowd Density":"HIGH", "color":"purple", "hex":"#A855F7"},
        "OXAGON": {"lat":27.5020, "lon":35.4750, "Area Type":"Urban", "Load Sector":"Industrial", "Weather":"clear", "Green Density":"LOW", "Vehicle Density":"HIGH", "Crowd Density":"MEDIUM", "color":"blue", "hex":"#3B82F6"},
        "TROJENA": {"lat":28.5160, "lon":35.3100, "Area Type":"Mountain", "Load Sector":"Commercial", "Weather":"snowy", "Green Density":"HIGH", "Vehicle Density":"LOW", "Crowd Density":"MEDIUM", "color":"green", "hex":"#22C55E"},
        "SINDALAH": {"lat":27.9440, "lon":34.7040, "Area Type":"Coastal", "Load Sector":"Commercial", "Weather":"clear", "Green Density":"MEDIUM", "Vehicle Density":"MEDIUM", "Crowd Density":"HIGH", "color":"orange", "hex":"#F97316"},
        "NEOM BAY": {"lat":27.9270, "lon":35.2990, "Area Type":"Coastal", "Load Sector":"Commercial", "Weather":"cloudy", "Green Density":"MEDIUM", "Vehicle Density":"MEDIUM", "Crowd Density":"MEDIUM", "color":"cadetblue", "hex":"#06B6D4"},
    }

    if "selected_zone" not in st.session_state:
        st.session_state.selected_zone = "THE LINE"

    left, mid, right = st.columns([1.1, 3.2, 1.5])

    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        for z in zones:
            if st.button(z):
                st.session_state.selected_zone = z
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    selected = zones[st.session_state.selected_zone]

    with mid:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        m = folium.Map(location=[28.1,35.2], zoom_start=7, tiles="CartoDB dark_matter")
        for name, data in zones.items():
            folium.Circle(
                [data["lat"], data["lon"]],
                radius=26000,
                color=data["hex"],
                fill=True,
                fill_color=data["hex"],
                fill_opacity=.16
            ).add_to(m)
            folium.Marker(
                [data["lat"], data["lon"]],
                tooltip=name,
                icon=folium.Icon(color=data["color"], icon="location-dot", prefix="fa")
            ).add_to(m)
        st_folium(m, width=780, height=540)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"## {st.session_state.selected_zone}")
        for k in ["Area Type", "Load Sector", "Weather", "Green Density", "Vehicle Density", "Crowd Density"]:
            st.markdown(f"**{k}:** `{selected[k]}`")

        if st.button(tr("Use This Location", "استخدام هذه المنطقة")):
            zone_name = st.session_state.selected_zone

            f = dict(selected)
            f["Zone"] = zone_name

            live_weather = get_live_weather(
                selected["lat"],
                selected["lon"]
            )

            f.update(live_weather)

            now = datetime.now()
            f["Vehicle Density"] = estimate_live_traffic(zone_name, now.hour)

            if f["Vehicle Density"] == "HIGH":
                f["Crowd Density"] = "HIGH"
            elif f["Vehicle Density"] == "MEDIUM":
                f["Crowd Density"] = selected.get("Crowd Density", "MEDIUM")
            else:
                f["Crowd Density"] = "LOW"

            st.session_state.features = feature_engineering(f)
            go("result")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button(tr("Back Home", "رجوع")):
        go("home")


# =========================
# IMAGE PAGE
# =========================
def image_page():
    st.markdown(f"<div class='main-title'>{tr('AI Visual Analysis Result 📸', 'نتيجة تحليل الصورة بالذكاء الاصطناعي 📸')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{tr('Upload one image to auto-fill energy forecasting inputs', 'ارفعي صورة واحدة لتعبئة مدخلات التنبؤ تلقائياً')}</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(tr("Upload City / Satellite Image", "ارفعي صورة مدينة أو ساتلايت"), type=["jpg","jpeg","png"])

    if uploaded:
        img, outputs = image_models_predict(uploaded)

        # =========================
        # AI CONFIDENCE REVIEW
        # =========================

        if outputs.get("Land Cover Confidence", 100) < 70:

            land_options = [
                "Urban Area",
                "Agricultural Area",
                "Open Natural Land",
                "Forest / Green Area",
                "Water Body",
                "Barren Land"
            ]

            current_land = outputs.get("Land Cover", "Urban Area")

            outputs["Land Cover"] = st.selectbox(
                "Land Cover confidence is low. Review if needed",
                land_options,
                index=land_options.index(current_land)
                if current_land in land_options else 0
            )

        if outputs.get("Vehicle Count", 0) == 0 and outputs.get("Area Type") == "Urban":

            vehicle_options = ["LOW", "MEDIUM", "HIGH"]

            current_vehicle = outputs.get("Vehicle Density", "LOW")

            outputs["Vehicle Density"] = st.selectbox(
                "Vehicle detection confidence is low. Review if needed",
                vehicle_options,
                index=vehicle_options.index(current_vehicle)
                if current_vehicle in vehicle_options else 0
            )
        st.image(img, use_container_width=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"## {tr('CNN Visual Analysis', 'تحليل الصورة')}")

        c1,c2,c3 = st.columns(3)
        with c1:
            metric_card("🚗", "Vehicle Density", "كثافة المركبات", outputs.get("Vehicle Density", "N/A"))
        with c2:
            metric_card("👥", "Crowd Density", "كثافة الأشخاص", outputs.get("Crowd Density", "N/A"))
        with c3:
            metric_card("🌳", "Green Density", "كثافة المساحات الخضراء", outputs.get("Green Density", "N/A"))

        c4,c5,c6 = st.columns(3)
        with c4:
            metric_card("☁️", "Weather", "الطقس", outputs.get("Weather", "N/A"), "#8CCEFF")
        with c5:
            metric_card("🏙️", "Area Type", "نوع المنطقة", outputs.get("Area Type", "N/A"), "#FFD84D")
        with c6:
            metric_card("🌍", "Land Cover", "الغطاء الأرضي", outputs.get("Land Cover", "N/A"), "#7CFF9B")

        outputs["Load Sector"] = "Commercial"
        st.session_state.features = feature_engineering(outputs)

        if st.button(tr("Generate Energy Forecast", "توليد التنبؤ بالطاقة")):
            go("result")

        st.markdown("</div>", unsafe_allow_html=True)

    if st.button(tr("Back Home", "رجوع")):
        go("home")


# =========================
# MANUAL PAGE
# =========================
def manual_page():
    st.markdown(f"<div class='main-title'>{tr('Manual Input ✍️', 'الإدخال اليدوي ✍️')}</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)

    with c1:
        area = st.selectbox("Area Type\nنوع المنطقة", ["Urban", "Suburban", "Rural", "Coastal", "Mountain"])
        sector = st.selectbox("Load Sector\nقطاع الاستهلاك الكهربائي", ["Commercial", "Industrial", "Residential"])

    with c2:
        solar = st.number_input("Solar PV Output (kW)\nالطاقة الشمسية", value=50.0)
        wind = st.number_input("Wind Power Output (kW)\nطاقة الرياح", value=20.0)

    with c3:
        current = st.number_input("Current Level (A)\nمستوى التيار", value=100.0)
        load_level = st.selectbox("Current Load Level\n مستوى الحمل الكهربائي", ["Normal", "Peak"])

    with c4:
        weather = st.selectbox("Weather Condition\nحالة الطقس", ["clear", "cloudy", "rainy", "snowy", "stormy"])
        energy_status = st.selectbox("Energy Status\nحالة الطاقة", ["No Event", "Event"])

    vehicle_density = st.selectbox("Vehicle Density\nكثافة المركبات", ["LOW", "MEDIUM", "HIGH"])
    crowd_density = st.selectbox("Crowd Density\nكثافة الأشخاص", ["LOW", "MEDIUM", "HIGH"])
    green_density = st.selectbox("Green Density\nكثافة المساحات الخضراء", ["LOW", "MEDIUM", "HIGH"])

    if st.button(tr("Generate Forecast", "توليد التنبؤ")):
        f = {
            "Area Type": area,
            "Load Sector": sector,
            "Solar PV Output (kW)": solar,
            "Wind Power Output (kW)": wind,
            "Current Level (A)": current,
            "Current Load Level": load_level,
            "Weather": weather,
            "Energy Status": energy_status,
            "Vehicle Density": vehicle_density,
            "Crowd Density": crowd_density,
            "Green Density": green_density
        }
        st.session_state.features = feature_engineering(f)
        go("result")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(tr("Back Home", "رجوع")):
        go("home")


# =========================
# RESULT PAGE
# =========================
def result_page():
    features = st.session_state.features
    predicted = simple_load_forecast(features)
    tips = recommendations(features, predicted)

    st.markdown(f"<div class='main-title'>{tr('Forecast Result ⚡', 'نتيجة التنبؤ ⚡')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{tr('Final electricity load prediction with smart recommendations', 'التنبؤ النهائي بالحمل الكهربائي مع توصيات ذكية')}</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    metric_card("⚡", "Predicted Electricity Load", "الحمل الكهربائي المتوقع", f"{predicted} kW", "#FFD84D")

    # =========================
    # Load Forecast Graph
    # =========================
    low_load = 700
    medium_load = 1400
    high_load = 2200

    fig = pgo.Figure()

    fig.add_trace(pgo.Bar(
        x=["Low Load", "Medium Load", "High Load", "Predicted Load"],
        y=[low_load, medium_load, high_load, predicted],
        text=[low_load, medium_load, high_load, round(predicted, 2)],
        textposition="outside",
        marker=dict(
            color=["#22C55E", "#FACC15", "#F97316", "#A855F7"],
            line=dict(color="#FFFFFF", width=1)
        )
    ))

    fig.update_layout(
        title="Electricity Load Comparison | مقارنة الحمل الكهربائي",
        xaxis_title="Load Level | مستوى الحمل",
        yaxis_title="kW | كيلوواط",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,16,38,0.65)",
        font=dict(color="white"),
        height=420,
        margin=dict(l=40, r=40, t=70, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Scenario Summary")
    st.json(features)

    st.markdown("### Smart Recommendations | التوصيات الذكية")
    for t in tips:
        st.success(t)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(tr("Back Home", "الصفحة الرئيسية")):
        go("home")


# =========================
# ROUTER
# =========================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "map":
    map_page()
elif st.session_state.page == "image":
    image_page()
elif st.session_state.page == "manual":
    manual_page()
elif st.session_state.page == "result":
    result_page()
