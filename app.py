from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Drift Classifier API")

# Load from Model Registry
model = mlflow.pyfunc.load_model("models:/drift-classifier/Latest")

class Features(BaseModel):
	features: list[float]

app@.post("/predict")
def predict(data: Features):
	df = pd.DataFrame([data.features])
	pred = model.predict(df)[0]
	prob = model.predict_proba(df)[0].tolist()
	return {"prediction": int(pred), "probability": prob}
