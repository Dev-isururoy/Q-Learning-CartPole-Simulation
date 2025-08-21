import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
num_samples = 500

# Generate random features similar to Iris dataset ranges
sepal_length = np.random.normal(loc=5.8, scale=0.8, size=num_samples)
sepal_width = np.random.normal(loc=3.0, scale=0.4, size=num_samples)
petal_length = np.random.normal(loc=3.7, scale=1.5, size=num_samples)
petal_width = np.random.normal(loc=1.2, scale=0.8, size=num_samples)

# Generate random species classes (0, 1, or 2)
species = np.random.choice([0, 1, 2], size=num_samples)

# Build DataFrame
df = pd.DataFrame({
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width,
    "species": species
})

# Clean any negative values (since lengths can't be negative)
for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
    df[col] = df[col].clip(lower=0.1)

# Save to CSV
df.to_csv("data.csv", index=False)

print(f"Generated dataset with {num_samples} samples saved to data.csv")
