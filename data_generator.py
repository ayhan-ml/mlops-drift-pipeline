import pandas as pd
from sklearn.datasets import make_classification

# Generate synthetic classification data
X, y = make_classification(
	n_samples=1000, 
	n_features=20, 
	n_informative=15, 
	n_redundant=5,
	n_classes=2,
	random_state=42
)

# Create DataFrame with features
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

# ADD TARGET COLUMN
df['target'] = y

# Save to CSV
df.to_csv("ml_data.csv")
print("shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First row target:", df['target'].iloc[0])
