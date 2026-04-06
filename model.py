# import pandas as pd

# # Load dataset
# df = pd.read_csv("dataset.csv")

# print(df.shape)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Features (input)
X = df.drop("Disease", axis=1)

# Target (output)
y = df["Disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Accuracy check
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved successfully!")