import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv("ml_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Log params, metrics, model
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)
    mlflow.sklearn.log_model(model, "model")

    # Log data as artifact
    mlflow.log_artifact("ml_data.csv")

    print(f"Accuracy: {acc:.3f}, AUC: {auc:.3f}")
    print("Model logged to MLflow")
