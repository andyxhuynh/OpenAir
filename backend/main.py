import math
import random
import requests
from typing import List, Dict, Optional
import numpy as np
from scipy.ndimage import shift, gaussian_filter
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# =====================================================================
# DATABASE CONFIGURATION
# =====================================================================
DATABASE_URL = "sqlite:///./openair.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class AreaOfInterestTable(Base):
    __tablename__ = "areas_of_interest"
    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    layer_type = Column(String(30), default="street_block")
    coordinates = Column(JSON, nullable=False)

class PollutionSensorTable(Base):
    __tablename__ = "pollution_sensors"
    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    layer_type = Column(String(30), default="emission_source")
    emblem_type = Column(String(30), nullable=False) 
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

Base.metadata.create_all(bind=engine)

# =====================================================================
# APPLICATION INITIALIZATION
# =====================================================================
app = FastAPI(title="OpenAir Lyon Predictive Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LYON_CENTER_LAT = 45.76
LYON_CENTER_LNG = 4.87
GRID_SIZE = 50  

POLLUTANT_THRESHOLDS = {
    "pm25": {"standard": [10, 20, 25], "sensitive": [8, 15, 20]},   
    "pm10": {"standard": [20, 40, 50], "sensitive": [15, 30, 40]},   
    "o3":   {"standard": [50, 100, 140], "sensitive": [40, 80, 110]}, 
    "no2":  {"standard": [40, 90, 120], "sensitive": [30, 70, 100]}, 
    "so2":  {"standard": [100, 200, 350], "sensitive": [80, 150, 250]}, 
    "co":   {"standard": [4, 7, 10], "sensitive": [3, 5, 8]},         
    "pb":   {"standard": [0.2, 0.4, 0.5], "sensitive": [0.1, 0.3, 0.4]} 
}

class UserProfile(BaseModel):
    user_id: str
    name: str
    mode: str  

mock_users_db: Dict[str, UserProfile] = {
    "elena_123": UserProfile(user_id="elena_123", name="Elena Rostova", mode="young"),
    "bernard_789": UserProfile(user_id="bernard_789", name="Bernard Dupont", mode="elder")
}

def seed_database_placeholders():
    db = SessionLocal()
    if db.query(AreaOfInterestTable).first() is not None:
        db.close()
        return

    emitters = [
        PollutionSensorTable(id="source_factory_01", name="Vallée de la Chimie Industrial Complex", emblem_type="factory", latitude=45.76, longitude=4.878),
        PollutionSensorTable(id="source_power_02", name="Villeurbanne District Thermal Power Plant", emblem_type="power_plant", latitude=45.805, longitude=4.942)
    ]
    db.add_all(emitters)

    block_id_counter = 0
    for i in range(-10, 10):
        for j in range(-10, 10):
            lat_offset = i * 0.003  
            lng_offset = j * 0.004
            block_lat = LYON_CENTER_LAT + lat_offset
            block_lng = LYON_CENTER_LNG + lng_offset
            
            w = 0.0015
            polygon_coordinates = [[
                [block_lng - w, block_lat - w], [block_lng + w, block_lat - w],
                [block_lng + w, block_lat + w], [block_lng - w, block_lat + w],
                [block_lng - w, block_lat - w]
            ]]
            
            block_record = AreaOfInterestTable(
                id=f"block_{block_id_counter}",
                name=f"District Block #{block_id_counter}",
                layer_type="street_block",
                coordinates=polygon_coordinates
            )
            db.add(block_record)
            block_id_counter += 1
            
    db.commit()
    db.close()

seed_database_placeholders()

# =====================================================================
# ENVIRONMENTAL ENGINE (DUAL API INTEGRATION)
# =====================================================================
class EnvironmentalEngine:
    _live_data = None

    @classmethod
    def fetch_live_data(cls):
        if cls._live_data: return cls._live_data 
            
        try:
            # 1. Fetch Wind Data (Open-Meteo)
            w_url = "https://api.open-meteo.com/v1/forecast?latitude=45.76&longitude=4.87&current=wind_speed_10m,wind_direction_10m"
            weather = requests.get(w_url).json()["current"]
            
            # 2. Fetch Air Quality Data (WAQI)
            waqi_token = "cbb3446a161ed28ded4f11d5dc8b29bd2aa68713" # <--- PASTE YOUR TOKEN HERE
            a_url = f"https://api.waqi.info/feed/geo:45.76;4.87/?token={waqi_token}"
            waqi_response = requests.get(a_url).json()
            iaqi = waqi_response.get("data", {}).get("iaqi", {})

            cls._live_data = {
                "speed": weather["wind_speed_10m"] * 0.27778, 
                "direction": weather["wind_direction_10m"],
                "pm25": iaqi.get("pm25", {}).get("v", 12.0),
                "pm10": iaqi.get("pm10", {}).get("v", 20.0),
                "no2": iaqi.get("no2", {}).get("v", 15.0),
                "o3": iaqi.get("o3", {}).get("v", 40.0)
            }
            return cls._live_data
        except Exception as e:
            return {"speed": 2.5, "direction": 210, "pm25": 12.0, "pm10": 20.0, "no2": 15.0, "o3": 40.0}

    @classmethod
    def get_current_wind_vector(cls):
        data = cls.fetch_live_data()
        return {"speed": data["speed"], "direction": data["direction"]}

    @classmethod
    def simulate_sensor_grids(cls) -> Dict[str, np.ndarray]:
        live_data = cls.fetch_live_data()
        grids = {}
        
        for pollutant in POLLUTANT_THRESHOLDS.keys():
            bg_level = live_data.get(pollutant, 5.0) 
            # Cast grid to float to prevent integer/float sum crashes
            grid = np.full((GRID_SIZE, GRID_SIZE), bg_level, dtype=float)
            
            grid[10, 12] = 85.0 if pollutant in ["no2", "so2", "pm25"] else bg_level + 5.0
            grid[25, 28] = 95.0 if pollutant in ["pm10", "pb", "co"] else bg_level + 12.0
            grid += np.random.uniform(-1.0, 1.0, (GRID_SIZE, GRID_SIZE))
            grids[pollutant] = grid
            
        cls._live_data = None 
        return grids

    @classmethod
    def run_advection_diffusion_model(cls) -> Dict[str, Dict[int, np.ndarray]]:
        wind = cls.get_current_wind_vector()
        angle_rad = math.radians(wind["direction"])
        
        speed_factor = wind["speed"] * 0.15 
        dx = -speed_factor * math.sin(angle_rad)
        dy = speed_factor * math.cos(angle_rad)
        
        base_grids = cls.simulate_sensor_grids()
        forecast_timeline = {p: {} for p in POLLUTANT_THRESHOLDS.keys()}
        
        for pollutant, base_grid in base_grids.items():
            forecast_timeline[pollutant][0] = base_grid
            current_state = base_grid.copy()
            for hour in [1, 2, 3]:
                shifted = shift(current_state, shift=[dy * hour, dx * hour], cval=2.0)
                diluted = gaussian_filter(shifted, sigma=0.8 * hour)
                forecast_timeline[pollutant][hour] = diluted
                current_state = diluted
                
        return forecast_timeline

    @staticmethod
    def calculate_color_status(pollutant_values: Dict[str, float], mode: str) -> str:
        highest_severity = 0  
        tier = "sensitive" if mode in ["young", "elder"] else "standard"
        for pollutant, val in pollutant_values.items():
            limits = POLLUTANT_THRESHOLDS[pollutant][tier]
            if val <= limits[0]: severity = 0 
            elif val <= limits[2]: severity = 1 
            else: severity = 2 
            if severity > highest_severity: highest_severity = severity
        if highest_severity == 0: return "Green"
        if highest_severity == 1: return "Yellow"
        return "Red"

# =====================================================================
# REST ENDPOINTS
# =====================================================================
@app.get("/api/map/initial-blocks")
def get_initial_blocks_map(user_id: Optional[str] = "default", db: Session = Depends(get_db)):
    user_profile = mock_users_db.get(user_id, UserProfile(user_id="default", name="Guest User", mode="standard"))
    model_output = EnvironmentalEngine.run_advection_diffusion_model()
    
    geojson_features = []
    db_blocks = db.query(AreaOfInterestTable).all()
    db_sensors = db.query(PollutionSensorTable).all()
    
    for index, block in enumerate(db_blocks):
        gx = int((index % 20) * (GRID_SIZE / 20))
        gy = int((index // 20) * (GRID_SIZE / 20))
        current_metrics = {}
        for p in POLLUTANT_THRESHOLDS.keys():
            val = float(model_output[p][0][max(0, min(gy, GRID_SIZE-1)), max(0, min(gx, GRID_SIZE-1))])
            current_metrics[p] = round(val, 2)
            
        geojson_features.append({
            "type": "Feature",
            "properties": {
                "layer_type": block.layer_type,
                "block_name": block.name,
                "current_pollutants": current_metrics,
            },
            "geometry": {"type": "Polygon", "coordinates": block.coordinates}
        })
        
    live_env_data = EnvironmentalEngine.fetch_live_data()
    for source in db_sensors:
        # We simulate one factory spiking while the other reflects the real ambient WAQI reading
        geojson_features.append({
            "type": "Feature",
            "properties": {
                "layer_type": source.layer_type,
                "source_name": source.name,
                "core_pollutants": {
                    "pm25": 85.0 if "Vallée" in source.name else live_env_data.get("pm25", 12.0), 
                    "no2": 62.0 if "Vallée" in source.name else live_env_data.get("no2", 15.0)
                } 
            },
            "geometry": {"type": "Point", "coordinates": [source.longitude, source.latitude]}
        })

    return {
        "type": "FeatureCollection",
        "live_weather": EnvironmentalEngine.get_current_wind_vector(), 
        "features": geojson_features
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)