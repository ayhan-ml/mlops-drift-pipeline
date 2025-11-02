import mlflow

# Get the latest run ID from MLflow (no hardcoding)
runs = mlflow.search_runs()
latest_run = runs.iloc[-1]
run_id = latest_run["run_id"]

model_uri = f"runs:\{run_id}\model"
registered_name = "drift-classifier"

mv = mlflow.register_model(model_uri, registered_name)
print(f"Model registered: {registered_name} v{mv.version}")
print(f"Run ID used: {run_id})
