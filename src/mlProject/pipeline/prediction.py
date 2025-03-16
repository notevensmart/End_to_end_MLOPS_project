import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import pipeline

print("start pipeline")
class PredictionPipeline:
    def __init__(self):
        print("initialise model")
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        print("Model loaded")

    
    def predict(self, data):
        prediction = self.model.predict(data)
        print(f"this is prediction:{prediction}")

        return prediction