from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Drift Classifier API", version="1.0")

#Load model from MLflow Model Registry (Latest version)
model = mlflow.pyfunc.load_model("models:/drift-classifier/Latest")

class Features(BaseModel):
	features: list[float]

@app.get("/")
def home():
	return {"message": "Drift Classifier API is LIVE", "status": "healthy"}

@app.post("/predict")
def predict(data: Features):
	try:
		df = pd.DataFrame([data.features])
		pred = int(model.predict(df)[0])
		prob = model.predict_proba(df)[0].tolist()
		return {
			"prediction": pred,
			"probability": prob
		}
	except Exception as e:
		return {"error": str(e)}
