from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from ml_models.model_training import train_model
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

class ModelInput(BaseModel):
    X: list
    y: list
    algorithm: str
    metric: str
    features: list

@app.post("/process/")
async def process_input(input_data: ModelInput):
    # Convert input data to DataFrame
    X = pd.DataFrame(input_data.X)
    y = pd.Series(input_data.y)

    # Train model and get evaluation results
    evaluation_results = train_model(X, y, input_data.algorithm)

    # Log predictions
    logging.info(f"Predictions: {evaluation_results['predictions']}")

    return evaluation_results
