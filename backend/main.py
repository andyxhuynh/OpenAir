import math
import random
from typing import List, Dict, Optional
import numpy as np
from scipy.ndimage import shift, gaussian_filter
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Database ORM Imports
from sqlalchemy import create_engine, Column, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# =====================================================================
# DATABASE CONFIGURATION & CONNECTIVITY
# =====================================================================
DATABASE_URL = "sqlite:///./openair.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =====================================================================
# POSTGRESQL CORE OBJECT MAPPINGS (TABLES)
# =====================================================================
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
# APPLICATION CONFIGURATION & STATE INITIALIZATION
# =====================================================================
app = FastAPI(
    title="OpenAir Lyon Predictive Pollution Backend",
    description="Microscale atmospheric engine fueled by structured relational table lookups.",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LYON_CENTER_LAT = 45.76
LYON_CENTER_LNG = 4.87
GRID_SIZE = 50  # Keep our internal core 50x50 numpy computational matrix

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
    device_token: Optional[str] = None

mock_users_db: Dict[str, UserProfile] = {
    "elena_123": UserProfile(user_id="elena_123", name="Elena Rostova", mode="young", device_token="apns_token_elena"),
    "bernard_789": UserProfile(user_id="bernard_789", name="Bernard Dupont", mode="elder", device_token="apns_token_bernard")
}

def seed_database_placeholders():
    """Validates repository data; registers initial Lyon blocks & point emitters if tables are blank."""
    db = SessionLocal()
    if db.query(AreaOfInterestTable).first() is not None:
        db.close()
        return  

    print("[DATABASE SEED ENGINE] Injecting core placeholder assets into database tables...")
    
    # 1. Seed Stationary Point Emitters
    emitters = [
        PollutionSensorTable(id="source_factory_01", name="Vallée de la Chimie Industrial Complex", emblem_type="factory", latitude=45.751, longitude=4.8544),
        PollutionSensorTable(id="source_power_02", name="Villeurbanne District Thermal Power Plant", emblem_type="power_plant", latitude=45.7645, longitude=4.8836)
    ]
    db.add_all(emitters)

    # 2. FIXED: Map 100-meter steps cleanly to cover our GRID_SIZE area
    LAT_STEP = 0.0009   # ~100 meters North-South
    LNG_STEP = 0.0012   # ~100 meters East-West (Adjusted for contraction at Lyon's latitude)
    
    # Offset center so that the layout wraps completely around Lyon Center
    start_lat = LYON_CENTER_LAT - (GRID_SIZE / 2) * LAT_STEP
    start_lng = LYON_CENTER_LNG - (GRID_SIZE / 2) * LNG_STEP

    block_id_counter = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            block_lat = start_lat + (i * LAT_STEP)
            block_lng = start_lng + (j * LNG_STEP)
            
            # Construct a tight 100m bounding box geometry layout
            w_lat = LAT_STEP / 2
            w_lng = LNG_STEP / 2
            
            polygon_coordinates = [[
                [block_lng - w_lng, block_lat - w_lat],
                [block_lng + w_lng, block_lat - w_lat],
                [block_lng + w_lng, block_lat + w_lat],
                [block_lng - w_lng, block_lat + w_lat],
                [block_lng - w_lng, block_lat - w_lat]
            ]]
            
            name = f"Villeurbanne Block #{block_id_counter}" if j > (GRID_SIZE / 2) else f"Lyon District Block #{block_id_counter}"
            
            block_record = AreaOfInterestTable(
                id=f"block_{block_id_counter}",
                name=name,
                layer_type="street_block",
                coordinates=polygon_coordinates
            )
            db.add(block_record)
            block_id_counter += 1
            
    db.commit()
    db.close()
    print(f"[DATABASE SEED ENGINE] Complete. {block_id_counter} 100m grid entries populated.")

seed_database_placeholders()

# =====================================================================
# ALGOLIRITHMIC DISPERSION COMPUTATION ENGINE
# =====================================================================
class EnvironmentalEngine:
    @staticmethod
    def get_current_wind_vector():
        return {"speed": 12.5, "direction": 210}  

    @staticmethod
    def simulate_sensor_grids() -> Dict[str, np.ndarray]:
        grids = {}
        for pollutant in POLLUTANT_THRESHOLDS.keys():
            grid = np.zeros((GRID_SIZE, GRID_SIZE))
            
            # Map plume vectors onto indices matching the database seed footprint coordinates
            grid[10, 12] = 85.0 if pollutant in ["no2", "so2", "pm25"] else 5.0
            grid[25, 28] = 95.0 if pollutant in ["pm10", "pb", "co"] else 12.0
            grid += np.random.uniform(1.0, 5.0, (GRID_SIZE, GRID_SIZE))
            
            grids[pollutant] = grid
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
# REST ENDPOINTS FOR THE MOBILE FRONTEND
# =====================================================================
@app.get("/api/areas")
def get_all_database_areas(db: Session = Depends(get_db)):
    records = db.query(AreaOfInterestTable).all()
    return {
        "status": "success",
        "total_records": len(records),
        "data": [
            {
                "area_id": r.id,
                "name": r.name,
                "layer_type": r.layer_type,
                "geometry_coordinates": r.coordinates
            } for r in records
        ]
    }

@app.get("/api/map/initial-blocks")
def get_initial_blocks_map(user_id: Optional[str] = "default", db: Session = Depends(get_db)):
    user_profile = mock_users_db.get(user_id, UserProfile(user_id="default", name="Guest User", mode="standard"))
    model_output = EnvironmentalEngine.run_advection_diffusion_model()
    
    geojson_features = []
    active_alerts_count = 0
    
    db_blocks = db.query(AreaOfInterestTable).all()
    db_sensors = db.query(PollutionSensorTable).all()
    
    # Process Street Block Polygons matching structured array matrix elements
    for index, block in enumerate(db_blocks):
        # FIXED: Directly maps block row sequences to exact positions on our grid array
        gx = index % GRID_SIZE
        gy = index // GRID_SIZE
        
        current_metrics = {}
        forecast_timeline = []
        
        for hour in [0, 1, 2, 3]:
            hour_metrics = {}
            for p in POLLUTANT_THRESHOLDS.keys():
                val = float(model_output[p][hour][gy, gx])
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
        
        if block_color == "Red": active_alerts_count += 1
            
        geojson_features.append({
            "type": "Feature",
            "properties": {
                "layer_type": block.layer_type,
                "block_id": block.id,
                "block_name": block.name,
                "color_code": block_color,  
                "current_pollutants": current_metrics,
                "forecast_timeline": forecast_timeline
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": block.coordinates
            }
        })
        
    # Process Point Emitters from DB Records
    for source in db_sensors:
        geojson_features.append({
            "type": "Feature",
            "properties": {
                "layer_type": source.layer_type,
                "source_id": source.id,
                "source_name": source.name,
                "emblem_type": source.emblem_type,
                "core_pollutants": {"pm25": 42.1, "no2": 31.5} 
            },
            "geometry": {
                "type": "Point",
                "coordinates": [source.longitude, source.latitude]
            }
        })

    return {
        "type": "FeatureCollection",
        "user_context": {"name": user_profile.name, "mode_applied": user_profile.mode},
        "features": geojson_features
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)