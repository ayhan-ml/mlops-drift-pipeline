# CI test -run #2
import pandas as pd
import datetime

def check_mean_drift(data, threshold=10.0):
	mean_val = data.mean()
	if mean_val > threshold:
		return f"Mean {mean_val:.2f}: Drift detected - alert!"
	else:
		return f"Mean  {mean_val:.2f}: No drift - good!"

df = pd.read_csv("drift_data.csv")
result = check_mean_drift(df["value"])
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = f"[{timestamp}] {result}\n"
with open("pipeline.log", "a") as f:
	f.write(log_entry)
print("Logged to pipeline.log")
