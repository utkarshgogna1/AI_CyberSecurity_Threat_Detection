# Import necessary libraries
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time
import random
from typing import List
import threading

# ========== 1. LOAD BERT MODEL AND TOKENIZER ==========
print("Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("bert_anomaly_model.pth", map_location=torch.device("cpu")))
model.eval()
print("BERT model and tokenizer loaded successfully.")

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

# ========== 2. HELPER FUNCTION FOR INPUT PROCESSING ==========
def preprocess_input(data: TrafficLog):
    # Convert input into text suitable for BERT
    input_text = (
        f"Destination Port: {data.Destination_Port}, "
        f"Flow Duration: {data.Flow_Duration}, "
        f"Total Fwd Packets: {data.Total_Fwd_Packets}, "
        f"Total Length of Fwd Packets: {data.Total_Length_of_Fwd_Packets}, "
        f"Fwd Packet Length Max: {data.Fwd_Packet_Length_Max}, "
        f"Fwd Packet Length Mean: {data.Fwd_Packet_Length_Mean}, "
        f"Fwd IAT Total: {data.Fwd_IAT_Total}, "
        f"Flow Packets per s: {data.Flow_Packets_per_s}, "
        f"Packet Length Mean: {data.Packet_Length_Mean}"
    )
    return input_text

# ========== 3. PREDICTION ROUTES ==========

# Route for single real-time prediction
@app.post("/predict/")
async def predict_traffic(data: TrafficLog):
    input_text = preprocess_input(data)
    encoded_input = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Make predictions
    with torch.no_grad():
        outputs = model(**encoded_input)
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
        prediction = "Anomalous" if probabilities[1] > 0.5 else "Normal"

    return {
        "Prediction": prediction,
        "Probability": probabilities[1]
    }

# Route for batch predictions
@app.post("/predict/batch/")
async def predict_batch_traffic(data: List[TrafficLog]):
    predictions = []
    for entry in data:
        input_text = preprocess_input(entry)
        encoded_input = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = model(**encoded_input)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
            prediction = "Anomalous" if probabilities[1] > 0.5 else "Normal"

        predictions.append({"Prediction": prediction, "Probability": probabilities[1]})

    return {"Results": predictions}

# ========== 4. SIMULATED DATA INGESTION ==========
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

# ========== 5. BACKGROUND TASK ==========
print("Starting real-time data simulation...")
threading.Thread(target=simulate_real_time_data, daemon=True).start()

# Run with: uvicorn app:app --reload
