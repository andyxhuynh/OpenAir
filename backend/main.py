import math
import random
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
from scipy.ndimage import shift, gaussian_filter
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="OpenAir Lyon Predictive Pollution Backend",
    description="Microscale atmospheric dispersion engine for Lyon & Villeurbanne blocks.",
    version="1.0.0"
)

# Enable CORS for iOS apps and web testing clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# CONSTANTS & CONFIGURATION
# =====================================================================
# Center of the grid around Villeurbanne / East Lyon
LYON_CENTER_LAT = 45.76
LYON_CENTER_LNG = 4.87
GRID_SIZE = 50  # 50x50 cell internal computational matrix
GRID_RESOLUTION_KM = 0.2  # Each grid pixel represents 200 meters (~1 city block)

# European Air Quality Index (AQI) Thresholds for the 6 target pollutants
# Format: [Good_Max, Moderate_Max, Poor_Max] -> maps to Green, Yellow, Red
POLLUTANT_THRESHOLDS = {
    "pm25": {"standard": [10, 20, 25], "sensitive": [8, 15, 20]},   # µg/m³
    "pm10": {"standard": [20, 40, 50], "sensitive": [15, 30, 40]},   # µg/m³
    "o3":   {"standard": [50, 100, 140], "sensitive": [40, 80, 110]}, # µg/m³
    "no2":  {"standard": [40, 90, 120], "sensitive": [30, 70, 100]}, # µg/m³
    "so2":  {"standard": [100, 200, 350], "sensitive": [80, 150, 250]}, # µg/m³
    "co":   {"standard": [4, 7, 10], "sensitive": [3, 5, 8]},         # mg/m³
    "pb":   {"standard": [0.2, 0.4, 0.5], "sensitive": [0.1, 0.3, 0.4]} # µg/m³
}

# =====================================================================
# SIMULATED IN-MEMORY DATABASE STATE
# =====================================================================
class UserProfile(BaseModel):
    user_id: str
    name: str
    mode: str  # "standard", "young" (Elena's kids), or "elder" (Bernard)
    device_token: Optional[str] = None

# Mock database tracking physical street blocks in Villeurbanne/Lyon
mock_blocks_db: List[Dict] = []
# Mock active users matching your course personas
mock_users_db: Dict[str, UserProfile] = {
    "elena_123": UserProfile(user_id="elena_123", name="Elena Rostova", mode="young", device_token="apns_token_elena"),
    "bernard_789": UserProfile(user_id="bernard_789", name="Bernard Dupont", mode="elder", device_token="apns_token_bernard")
}

def init_mock_lyon_blocks():
    """Generates an array of synthetic street block polygons covering Villeurbanne & Lyon."""
    block_id_counter = 0
    # Generate 400 micro-blocks centered around Villeurbanne
    for i in range(-10, 10):
        for j in range(-10, 10):
            lat_offset = i * 0.003  # Roughly 300 meters step
            lng_offset = j * 0.004
            block_lat = LYON_CENTER_LAT + lat_offset
            block_lng = LYON_CENTER_LNG + lng_offset
            
            # Simple GeoJSON polygon coordinates generation for a square block
            w = 0.0015
            polygon_coordinates = [[
                [block_lng - w, block_lat - w],
                [block_lng + w, block_lat - w],
                [block_lng + w, block_lat + w],
                [block_lng - w, block_lat + w],
                [block_lng - w, block_lat - w]
            ]]
            
            name = f"Villeurbanne Block #{block_id_counter}" if j > -2 else f"Lyon District Block #{block_id_counter}"
            mock_blocks_db.append({
                "id": f"block_{block_id_counter}",
                "name": name,
                "center": {"lat": block_lat, "lng": block_lng},
                "coordinates": polygon_coordinates,
                "grid_x": int((j + 10) * (GRID_SIZE / 20)),
                "grid_y": int((i + 10) * (GRID_SIZE / 20))
            })
            block_id_counter += 1

init_mock_lyon_blocks()

# =====================================================================
# ATMOSPHERIC DISPERSION & COMPENSATORY ENGINE
# =====================================================================
class EnvironmentalEngine:
    @staticmethod
    def get_current_wind_vector():
        """
        Simulates fetching wind vectors from the Météo-France / Open-Meteo API.
        Returns wind speed in km/h and wind direction in degrees (0 = North, 90 = East).
        """
        return {"speed": 12.5, "direction": 210}  # South-SouthWest wind blowing plumes North-East across Lyon

    @staticmethod
    def simulate_sensor_grids() -> Dict[str, np.ndarray]:
        """Generates the base ground-truth matrices using simulated industrial source nodes."""
        grids = {}
        for pollutant in POLLUTANT_THRESHOLDS.keys():
            grid = np.zeros((GRID_SIZE, GRID_SIZE))
            
            # Simulated point source 1: Industrial corridor / Vallée de la Chimie (South-West)
            grid[10, 12] = 85.0 if pollutant in ["no2", "so2", "pm25"] else 5.0
            # Simulated point source 2: Factory zone near Villeurbanne outskirts (Center-East)
            grid[25, 28] = 95.0 if pollutant in ["pm10", "pb", "co"] else 12.0
            # Ambient urban baseline pollution addition
            grid += np.random.uniform(1.0, 5.0, (GRID_SIZE, GRID_SIZE))
            
            grids[pollutant] = grid
        return grids

    @classmethod
    def run_advection_diffusion_model(cls) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Applies mathematical shifts (advection) and smoothing filters (diffusion) 
        to project pollution movement across the 3-hour lookahead frame.
        """
        wind = cls.get_current_wind_vector()
        angle_rad = math.radians(wind["direction"])
        
        # Calculate coordinate step shifts based on wind vector speed
        speed_factor = wind["speed"] * 0.15 
        dx = -speed_factor * math.sin(angle_rad)
        dy = speed_factor * math.cos(angle_rad)
        
        base_grids = cls.simulate_sensor_grids()
        forecast_timeline = {p: {} for p in POLLUTANT_THRESHOLDS.keys()}
        
        for pollutant, base_grid in base_grids.items():
            # Hour 0 (Current State Snapshot)
            forecast_timeline[pollutant][0] = base_grid
            
            # Iteratively calculate looking ahead states T+1, T+2, T+3 hours
            current_state = base_grid.copy()
            for hour in [1, 2, 3]:
                # 1. Advection: Physically shift pollution downwind
                shifted = shift(current_state, shift=[dy * hour, dx * hour], cval=2.0)
                # 2. Diffusion: Dissipate the concentration as the plume expands outward
                diluted = gaussian_filter(shifted, sigma=0.8 * hour)
                
                forecast_timeline[pollutant][hour] = diluted
                current_state = diluted
                
        return forecast_timeline

    @staticmethod
    def calculate_color_status(pollutant_values: Dict[str, float], mode: str) -> str:
        """
        Calculates sub-AQI classifications, matching specific sensitivity settings 
        configured for Elena and Bernard's user profiles.
        """
        highest_severity = 0  # 0=Green, 1=Yellow, 2=Red
        
        # Decide which threshold criteria array to apply based on profile type
        tier = "sensitive" if mode in ["young", "elder"] else "standard"
        
        for pollutant, val in pollutant_values.items():
            limits = POLLUTANT_THRESHOLDS[pollutant][tier]
            if val <= limits[0]:
                severity = 0 # Green
            elif val <= limits[2]:
                severity = 1 # Yellow
            else:
                severity = 2 # Red
                
            if severity > highest_severity:
                highest_severity = severity
                
        if highest_severity == 0: return "Green"
        if highest_severity == 1: return "Yellow"
        return "Red"

# =====================================================================
# REST ENDPOINTS FOR THE MOBILE FRONTEND
# =====================================================================
@app.get("/api/map/initial-blocks")
def get_initial_blocks_map(user_id: Optional[str] = "default"):
    """
    SOLELY INITIALLY API ENDPOINT: Feeds the mobile map with clean GeoJSON map data
    of blocks immediately color-coded to adapt seamlessly to the user profile type.
    """
    # Fetch profile characteristics to handle profile modes dynamically
    user_profile = mock_users_db.get(user_id, UserProfile(user_id="default", name="Guest User", mode="standard"))
    
    # Process the 3-hour projection grid matrix
    model_output = EnvironmentalEngine.run_advection_diffusion_model()
    
    geojson_features = []
    active_alerts_count = 0
    
    for block in mock_blocks_db:
        gx, gy = block["grid_x"], block["grid_y"]
        
        # Extract the real-time calculated pollutant arrays inside our 2D grid matrix
        current_metrics = {}
        forecast_timeline = []
        
        # Build standard timelines (Current T+0 through T+3 Hours lookahead)
        for hour in [0, 1, 2, 3]:
            hour_metrics = {}
            for p in POLLUTANT_THRESHOLDS.keys():
                # Extract coordinate value safely preventing out-of-bound errors
                val = float(model_output[p][hour][max(0, min(gy, GRID_SIZE-1)), max(0, min(gx, GRID_SIZE-1))])
                hour_metrics[p] = round(val, 2)
            
            color_at_hour = EnvironmentalEngine.calculate_color_status(hour_metrics, user_profile.mode)
            
            if hour == 0:
                current_metrics = hour_metrics
                block_color = color_at_hour
            else:
                forecast_timeline.append({
                    "time": f"T+{hour}",
                    "dominant_color": color_at_hour,
                    "metrics": hour_metrics
                })
        
        if block_color == "Red":
            active_alerts_count += 1
            
        # Compile structured standard GeoJSON format to pass cleanly to iOS MapKit frameworks
        feature = {
            "type": "Feature",
            "properties": {
                "block_id": block["id"],
                "block_name": block["name"],
                "color_code": block_color,  # Dynamic visual target color mapping per block
                "current_pollutants": current_metrics,
                "forecast_timeline": forecast_timeline
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": block["coordinates"]
            }
        }
        geojson_features += [feature]
        
    # Handle proactive background notification routing triggers
    if active_alerts_count > 0:
        trigger_proactive_notifications(active_alerts_count)

    return {
        "type": "FeatureCollection",
        "user_context": {
            "name": user_profile.name,
            "mode_applied": user_profile.mode
        },
        "features": geojson_features
    }

# =====================================================================
# PUSH NOTIFICATION BACKBONE ENGINE HOOKS
# =====================================================================
def trigger_proactive_notifications(red_block_count: int):
    """
    Scans predictive grid spikes to run direct, automated notification delivery 
    rules satisfying Bernard's explicit routine requirements.
    """
    print(f"\n[BACKGROUND PUSH NOTIFICATION TRIGGERED]")
    print(f"CRITICAL STATE CHECK: {red_block_count} blocks in the Lyon area are entering critical Red Status.")
    
    for user_id, profile in mock_users_db.items():
        if profile.mode == "elder" and profile.device_token:
            print(f"-> Firing APNs Mobile Push Notification payload to Bernard ({profile.device_token}):")
            print(f"   'Alert Bernard: A high-concentration emission plume is moving across your walking perimeter within the next 2 hours.'")
        elif profile.mode == "young" and profile.device_token:
            print(f"-> Firing APNs Mobile Push Notification payload to Elena ({profile.device_token}):")
            print(f"   'OpenAir Alert: Localized pollutant spikes expanding towards nearby schools & parks.'")
    print("=========================================\n")

if __name__ == "__main__":
    import uvicorn
    # Execute backend locally on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)