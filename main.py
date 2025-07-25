from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import ee
import os
from datetime import datetime, timedelta
import requests
import httpx


# Step 1: Initialize Earth Engine
app = FastAPI()

class Location(BaseModel):
    lat: float
    lng: float

try:
    ee.Initialize(project='gen-lang-client-0496984018')
except Exception:
    ee.Authenticate()
    ee.Initialize()

GOOGLE_MAPS_API_KEY = "AIzaSyB5oltuq_EXN0Gx_lo6jo8idtA0EtT0nII"
SAVE_DIR = "static/maps"
os.makedirs(SAVE_DIR, exist_ok=True)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input model
class LocationRequest(BaseModel):
    lat: float
    lon: float

# Step 6: Main route to fetch and save LST image
@app.post("/get_lst/")
def get_lst(data: LocationRequest):
    lat, lon = data.lat, data.lon
    point = ee.Geometry.Point([lon, lat])
    area = point.buffer(50000).bounds()

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=200)

    # LST collection
    lst_collection = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterBounds(area)
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    )

    if lst_collection.size().getInfo() == 0:
        return {"error": "No valid LST data available for this location/date range."}

    raw_lst = lst_collection.select("LST_Day_1km").median()
    lst_image = (
        raw_lst.multiply(0.02)
        .subtract(273.15)
        .focal_mean(radius=1, units='pixels')
        .clip(area)
    )

    # Solar radiation: NASA POWER daily surface shortwave radiation
    srad_collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(area)
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        .select("surface_solar_radiation_downwards_sum")
    )

    solar_image = srad_collection.mean().clip(area)

    # Visualization style
    vis_params = {
        "min": 20,
        "max": 40,
        "palette": [
            "001137", "002171", "02489d", "2c7bb6", "abd9e9",
            "ffffbf", "fdae61", "f46d43", "d73027", "7f0000"
        ]
    }

    # File name
    filename = f"lst_{round(lat, 6)}_{round(lon, 6)}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    path = os.path.join(OUTPUT_DIR, filename)

    # Get thumbnail
    url = lst_image.getThumbURL({
        "region": area,
        "dimensions": 512,
        "format": "png",
        **vis_params
    })

    # Get value at point (mean of 1km square around point)
    lst_value = lst_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=1000
    ).getInfo()

    srad_value = solar_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=1000
    ).getInfo()

    # Parse values
    lst_celsius = lst_value.get("LST_Day_1km")
    solar_radiation = srad_value.get("surface_solar_radiation_downwards_sum")

    # Download image
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        return {
            "message": "LST image saved.",
            "path": path,
            "url": url,
            "lst": round(lst_celsius, 2) if lst_celsius else None,
            "solar_radiation": round(solar_radiation / 1_000_000, 2) if solar_radiation else None
        }
    else:
        return {"error": "Failed to fetch image", "status": response.status_code}


@app.post("/streetview")
async def get_street_view(loc: Location):
    street_view_url = (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size=600x400"
        f"&location={loc.lat},{loc.lng}"
        f"&heading=90&pitch=0"
        f"&key={GOOGLE_MAPS_API_KEY}"
    )

    # Download the image
    async with httpx.AsyncClient() as client:
        response = await client.get(street_view_url)
        if response.status_code != 200:
            return {"error": "Could not fetch Street View image"}

    filename = f"streetview_{str(loc.lat).replace('.','_')}_{str(loc.lng).replace('.','_')}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(response.content)

    return {"message": "Street View image saved", "file_path": filepath}

@app.post("/hybrid_map")
async def get_hybrid_map(loc: Location):
    hybrid_url = (
        f"https://maps.googleapis.com/maps/api/staticmap"
        f"?center={loc.lat},{loc.lng}"
        f"&zoom=18"
        f"&size=640x640"
        f"&scale=2"
        f"&maptype=hybrid"
        f"&markers=color:red%7Clabel:%7C{loc.lat},{loc.lng}"  # 👈 Add this line
        f"&key={GOOGLE_MAPS_API_KEY}"
    )

    async with httpx.AsyncClient() as client:
        response = await client.get(hybrid_url)
        if response.status_code != 200:
            return {"error": "Could not fetch satellite image"}

    filename = f"hybridmap_{str(loc.lat).replace('.','_')}_{str(loc.lng).replace('.','_')}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(response.content)

    return {"message": "Hybrid satellite image saved", "file_path": filepath}



# Constants
CO2_PER_KWH_KG = 0.7  # kg CO₂ per kWh
CARBON_CREDIT_PER_KG = 1 / 1000  # 1 credit per 1000 kg
TARIFF_MIN = 0.218  # RM/kWh (low usage)
TARIFF_MAX = 0.571  # RM/kWh (high usage)

# Input model
class TEGPlanInput(BaseModel):
    num_tegs: int
    energy_per_module_wh: float  # Wh/day
    cost_per_module_rm: float

@app.post("/calculate_teg_plan/")
def calculate_teg_plan(data: TEGPlanInput):
    total_energy_wh = data.num_tegs * data.energy_per_module_wh
    total_energy_kwh = total_energy_wh / 1000

    co2_saved_kg = total_energy_kwh * CO2_PER_KWH_KG
    carbon_credits = co2_saved_kg * CARBON_CREDIT_PER_KG

    # Energy cost savings based on Malaysian TNB tariff range
    min_savings_rm = total_energy_kwh * TARIFF_MIN
    max_savings_rm = total_energy_kwh * TARIFF_MAX

    return {
        "Total Energy Generated (Wh/day)": total_energy_wh,
        "Total Energy (kWh/day)": round(total_energy_kwh, 3),
        "Number of TEG Modules": data.num_tegs,
        "CO2 Saved (kg/day)": round(co2_saved_kg, 3),
        "Carbon Credits Earned per Day": round(carbon_credits, 4),
        "Daily Energy Cost Savings (RM Range)": f"RM {round(min_savings_rm, 2)} - RM {round(max_savings_rm, 2)}"
    }
