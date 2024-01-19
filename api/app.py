###############################################################
# Author: walterwhites
# Hackathon: CrackedDevs Hackathon Jan 2024
# This code is subject Devpost Hackathon and restrictions.
###############################################################

import pandas as pd
import requests
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from api.preprocessing import preprocess_text, search

app = FastAPI()
combined_pipeline = None
mlb = None

def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)

@app.on_event("startup")
def load_models():
    global model

    model = load('model_Word2Vec.joblib')

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/search")
def searchJob(q: str):
    df_cleaned = pd.read_pickle("api/cleaned_jobs_data.pkl")
    processed_query = preprocess_text(q)
    search_results = search(processed_query, model, df_cleaned)
    result_list = search_results.to_dict(orient="records")
    return {"prediction": result_list}
