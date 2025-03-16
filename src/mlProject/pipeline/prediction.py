import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import pipeline

print("Model loaded")
class PredictionPipeline:
    def __init__(self):
        print("Model loaded")
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        print("Model loaded")

    
    def predict(self, data):
        prediction = self.model.predict(data)
        print("this is prediction:{prediction}")

        return prediction