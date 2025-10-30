import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="EV Range Predictor", layout="wide")
st.title("⚡ Electric Vehicle Range Predictor with Charging Map")

# -------------------------------
# LOAD DATASET
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("electric_vehicles_spec_2025_clean.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.sidebar.header("🔍 Model Controls")

# -------------------------------
# FEATURES & MODEL TRAINING
# -------------------------------
features = ['battery_capacity_kwh', 'motor_power_kw', 'top_speed_kmph', 'weight_kg', 'load_kg']
target = 'range_km'

if not all(f in df.columns for f in features + [target]):
    st.error("❌ Dataset missing required columns. Please check column names.")
    st.stop()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -------------------------------
# SIDEBAR SELECTION
# -------------------------------
st.sidebar.subheader("⚙️ Select Prediction Mode")
mode = st.sidebar.radio("Choose Mode:", ["Predict Range", "Predict Required Speed"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Enter EV Specifications")

battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 20, 200, 60)
motor_power = st.sidebar.slider("Motor Power (kW)", 50, 500, 150)
weight = st.sidebar.slider("Weight (kg)", 800, 3000, 1500)
load = st.sidebar.slider("Load Capacity (kg)", 100, 1000, 300)

# -------------------------------
# MODE 1: PREDICT RANGE
# -------------------------------
if mode == "Predict Range":
    top_speed = st.sidebar.slider("Top Speed (km/h)", 80, 300, 180)
    user_input = pd.DataFrame(
        [[battery_capacity, motor_power, top_speed, weight, load]],
        columns=features
    )

    scaled_input = scaler.transform(user_input)
    predicted_range = model.predict(scaled_input)[0]

    st.subheader("🚗 Predicted EV Range")
    st.success(f"Estimated Range: **{predicted_range:.2f} km**")

# -------------------------------
# MODE 2: PREDICT REQUIRED SPEED
# -------------------------------
# -------------------------------
# MODE 2: PREDICT REQUIRED SPEED (Fixed)
# -------------------------------
else:
    target_range = st.sidebar.slider("🎯 Desired Range (km)", 50, 800, 300)

    # Range of possible speeds to test
    possible_speeds = np.linspace(60, 300, 200)  # 60 to 300 kmph

    best_speed = None
    min_diff = float("inf")

    for s in possible_speeds:
        temp_input = pd.DataFrame(
            [[battery_capacity, motor_power, s, weight, load]],
            columns=features
        )
        temp_scaled = scaler.transform(temp_input)
        predicted = model.predict(temp_scaled)[0]
        diff = abs(predicted - target_range)
        if diff < min_diff:
            min_diff = diff
            best_speed = s

    st.subheader("🚘 Required Speed for Target Range")

    if best_speed is not None:
        st.success(f"Maintain approximately **{best_speed:.2f} km/h** to achieve {target_range:.0f} km range.")
    else:
        st.warning("⚠️ Could not estimate a valid speed for given parameters.")


# -------------------------------
# MODEL PERFORMANCE
# -------------------------------
st.markdown("### 📊 Model Performance")
y_pred = model.predict(scaler.transform(X_test))
st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# -------------------------------
# MAP SECTION (Charging Stations)
# -------------------------------
st.markdown("---")
st.header("🗺️ Find Nearest Charging Steations")

col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input("Enter Latitude", value=12.9716, format="%.4f", key="lat_input")
with col2:
    longitude = st.number_input("Enter Longitude", value=77.5946, format="%.4f", key="lon_input")

api_key = "c9d8a19a-4716-4798-b2c5-a33f6b8220f8"  # Replace with your actual key

@st.cache_data(show_spinner=False)
def fetch_stations(lat, lon):
    try:
        url = (
            f"https://api.openchargemap.io/v3/poi/?output=json"
            f"&latitude={lat}&longitude={lon}&distance=30&distanceunit=KM&maxresults=10"
        )
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "User-Agent": "EVRangePredictorApp/1.0"
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            st.warning(f"⚠️ API Error: {response.status_code}")
            return []
        return response.json()
    except Exception as e:
        st.error(f"Error fetching stations: {e}")
        return []

if "stations" not in st.session_state:
    st.session_state["stations"] = None

if st.button("🔍 Find Nearest Charging Stations"):
    st.session_state["stations"] = fetch_stations(latitude, longitude)

if st.session_state["stations"]:
    stations = st.session_state["stations"]
    if not stations:
        st.warning("No stations found or invalid API key.")
    else:
        m = folium.Map(location=[latitude, longitude], zoom_start=12)
        folium.Marker(
            [latitude, longitude],
            tooltip="Your Location",
            icon=folium.Icon(color='blue')
        ).add_to(m)

        cluster = MarkerCluster().add_to(m)
        for s in stations:
            lat = s.get("AddressInfo", {}).get("Latitude")
            lon = s.get("AddressInfo", {}).get("Longitude")
            name = s.get("AddressInfo", {}).get("Title", "Unknown")
            addr = s.get("AddressInfo", {}).get("AddressLine1", "")
            if lat and lon:
                folium.Marker(
                    [lat, lon],
                    tooltip=f"{name}\n{addr}",
                    icon=folium.Icon(color='green')
                ).add_to(cluster)

        st_folium(m, width=700, height=500)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Developed by Haravind | EV Sustainability Project 🌱")
