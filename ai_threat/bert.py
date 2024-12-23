# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import joblib
import numpy as np
import time
import random
from typing import List
import threading

# ========== 1. LOAD MODEL AND SCALER ==========
print("Loading model and scaler...")
model = joblib.load("cybersecurity_model.pkl")
scaler = joblib.load("scaler.pkl")
print("Model and scaler loaded successfully.")

# Define input schema for single and batch predictions
class TrafficLog(BaseModel):
    Destination_Port: float
    Flow_Duration: float
    Total_Fwd_Packets: float
    Total_Length_of_Fwd_Packets: float
    Fwd_Packet_Length_Max: float
    Fwd_Packet_Length_Mean: float
    Fwd_IAT_Total: float
    Flow_Packets_per_s: float
    Packet_Length_Mean: float

app = FastAPI()

# ========== CORS MIDDLEWARE ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from React app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all HTTP headers
)

# ========== 2. PREDICTION ROUTES ==========

# Route for single real-time prediction
@app.post("/predict/")
async def predict_traffic(data: TrafficLog):
    # Prepare input data
    input_data = [[
        np.log1p(data.Flow_Duration),  # Log transform
        np.log1p(data.Total_Length_of_Fwd_Packets),
        data.Destination_Port,
        data.Total_Fwd_Packets,
        data.Fwd_Packet_Length_Max,
        data.Fwd_Packet_Length_Mean,
        data.Fwd_IAT_Total,
        data.Flow_Packets_per_s,
        data.Packet_Length_Mean
    ]]
    input_scaled = scaler.transform(input_data)
    input_scaled = np.clip(input_scaled, -3, 3)

    # Predict with probabilities
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    result = "Anomalous" if prediction_proba > 0.5 else "Normal"

    # Return result
    return {
        "Prediction": result,
        "Probability": prediction_proba
    }

# Route for batch predictions
@app.post("/predict/batch/")
async def predict_batch_traffic(data: List[TrafficLog]):
    predictions = []
    for entry in data:
        input_data = [[
            np.log1p(entry.Flow_Duration),
            np.log1p(entry.Total_Length_of_Fwd_Packets),
            entry.Destination_Port,
            entry.Total_Fwd_Packets,
            entry.Fwd_Packet_Length_Max,
            entry.Fwd_Packet_Length_Mean,
            entry.Fwd_IAT_Total,
            entry.Flow_Packets_per_s,
            entry.Packet_Length_Mean
        ]]
        input_scaled = scaler.transform(input_data)
        input_scaled = np.clip(input_scaled, -3, 3)

        prediction_proba = model.predict_proba(input_scaled)[0][1]
        result = "Anomalous" if prediction_proba > 0.5 else "Normal"

        predictions.append({"Prediction": result, "Probability": prediction_proba})

    return {"Results": predictions}

# ========== 3. SIMULATED DATA INGESTION ==========
def simulate_real_time_data():
    """Simulates real-time network traffic for testing."""
    while True:
        sample = {
            "Destination_Port": random.randint(0, 65535),
            "Flow_Duration": random.uniform(1e6, 1e7),
            "Total_Fwd_Packets": random.randint(10, 1000),
            "Total_Length_of_Fwd_Packets": random.uniform(1e3, 1e5),
            "Fwd_Packet_Length_Max": random.uniform(100, 1500),
            "Fwd_Packet_Length_Mean": random.uniform(50, 1200),
            "Fwd_IAT_Total": random.uniform(1e2, 1e5),
            "Flow_Packets_per_s": random.uniform(1, 1e3),
            "Packet_Length_Mean": random.uniform(500, 1500)
        }
        print(f"Simulated Real-Time Data: {sample}")
        time.sleep(2)  # Simulate data arriving every 2 seconds

# ========== 4. BACKGROUND TASK ==========
print("Starting real-time data simulation...")
threading.Thread(target=simulate_real_time_data, daemon=True).start()

# Run with: uvicorn app:app --reload

