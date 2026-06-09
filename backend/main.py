import math
import random
import requests
from typing import List, Dict, Optional
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, String, Float, JSON, Integer
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

class HistoricalLogTable(Base):
    __tablename__ = "historical_logs"
    id = Column(String(50), primary_key=True, index=True)
    sensor_id = Column(String(50), index=True)
    hours_ago = Column(Integer, index=True)
    pm25 = Column(Float)
    no2 = Column(Float)

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

def seed_database_placeholders():
    db = SessionLocal()
    # If database is already seeded, clear historical logs to re-seed fresh profiles
    db.query(HistoricalLogTable).delete()
    
    # FIXED: Replaced Tensor with None
    if db.query(PollutionSensorTable).first() is None:
        emitters = [
            PollutionSensorTable(id="source_factory_01", name="Vallée de la Chimie Industrial Complex", emblem_type="factory", latitude=45.76, longitude=4.878),
            PollutionSensorTable(id="source_power_02", name="Villeurbanne District Thermal Power Plant", emblem_type="power_plant", latitude=45.805, longitude=4.942)
        ]
        db.add_all(emitters)

    # Seed 24 hours of distinct operational history for BOTH facilities
    historical_logs = []
    for h in range(1, 25):
        # Factory 1: Chemical Complex incident peaks 0-4 hours ago
        f1_pm = max(12.0, 85.0 - (h * 15.0)) if h <= 6 else 12.0
        f1_no = max(15.0, 62.0 - (h * 10.0)) if h <= 6 else 15.0
        
        # Factory 2: Thermal Power Plant had an independent maintenance spike 7-11 hours ago
        if 7 <= h <= 11:
            f2_pm = max(15.0, 78.0 - ((h - 7) * 15.0))
            f2_no = max(18.0, 58.0 - ((h - 7) * 10.0))
        else:
            f2_pm = 15.0
            f2_no = 18.0
        
        historical_logs.append(HistoricalLogTable(id=f"log_f1_{h}", sensor_id="source_factory_01", hours_ago=h, pm25=f1_pm, no2=f1_no))
        historical_logs.append(HistoricalLogTable(id=f"log_f2_{h}", sensor_id="source_power_02", hours_ago=h, pm25=f2_pm, no2=f2_no))
        
    db.add_all(historical_logs)
    
    # Check and seed street block polygons if empty
    if db.query(AreaOfInterestTable).first() is None:
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
                db.add(AreaOfInterestTable(id=f"block_{block_id_counter}", name=f"District Block #{block_id_counter}", layer_type="street_block", coordinates=polygon_coordinates))
                block_id_counter += 1
            
    db.commit()
    db.close()

seed_database_placeholders()

# =====================================================================
# ENVIRONMENTAL ENGINE
# =====================================================================
class EnvironmentalEngine:
    _live_data = None

    @classmethod
    def fetch_live_data(cls):
        if cls._live_data: return cls._live_data 
        try:
            w_url = "https://api.open-meteo.com/v1/forecast?latitude=45.76&longitude=4.87&current=wind_speed_10m,wind_direction_10m"
            weather = requests.get(w_url).json()["current"]
            
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
        except Exception:
            return {"speed": 2.5, "direction": 210, "pm25": 12.0, "pm10": 20.0, "no2": 15.0, "o3": 40.0}

# =====================================================================
# REST ENDPOINTS
# =====================================================================
@app.get("/api/map/initial-blocks")
def get_initial_blocks_map(hours_ago: int = 0, user_id: Optional[str] = "default", db: Session = Depends(get_db)):
    geojson_features = []
    db_sensors = db.query(PollutionSensorTable).all()
    live_env_data = EnvironmentalEngine.fetch_live_data()
    
    # Calculate historical wind vectors dynamically based on timeline scrubbing
    if hours_ago == 0:
        calculated_wind = {
            "speed": live_env_data.get("speed", 2.5),
            "direction": live_env_data.get("direction", 210)
        }
    else:
        # Simulate a shifting wind front moving through Lyon over the past 12 hours
        base_dir = live_env_data.get("direction", 210)
        calculated_wind = {
            "speed": max(1.2, live_env_data.get("speed", 2.5) + math.sin(hours_ago) * 1.2),
            "direction": (base_dir + (hours_ago * 35)) % 360
        }
    
    for source in db_sensors:
        if hours_ago == 0:
            pm25 = 85.0 if "Vallée" in source.name else live_env_data.get("pm25", 15.0)
            no2 = 62.0 if "Vallée" in source.name else live_env_data.get("no2", 18.0)
        else:
            log = db.query(HistoricalLogTable).filter(
                HistoricalLogTable.sensor_id == source.id, 
                HistoricalLogTable.hours_ago == hours_ago
            ).first()
            pm25 = log.pm25 if log else live_env_data.get("pm25", 12.0)
            no2 = log.no2 if log else live_env_data.get("no2", 15.0)

        geojson_features.append({
            "type": "Feature",
            "properties": {
                "layer_type": source.layer_type,
                "source_name": source.name,
                "core_pollutants": {
                    "pm25": round(pm25, 1), 
                    "no2": round(no2, 1)
                } 
            },
            "geometry": {"type": "Point", "coordinates": [source.longitude, source.latitude]}
        })

    return {
        "type": "FeatureCollection",
        "live_weather": calculated_wind, 
        "features": geojson_features
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)