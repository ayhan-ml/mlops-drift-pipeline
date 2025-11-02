import mlflow

# YOUR CURRENT RUN ID
run_id = "f7c746c53a8f44d785ed536142c8aa29"
model_uri = f"runs:/{run_id}/model"
registered_name = "drift-classifier"

# Register (creates new version)
mv = mlflow.register_model(model_uri, registered_name)
print(f"Model registered: {registered_name} v{mv.version}")
