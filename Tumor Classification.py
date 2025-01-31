import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Creating a synthetic dataset
data = {
    "Mutation Type": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    "Frequency": [0.1, 0.8, 0.6, 0.2, 0.9, 0.3, 0.15, 0.7, 0.95, 0.05, 0.85, 0.25, 0.65, 0.9, 0.1],
    "Cancerous": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print(df.head())

# Splitting dataset into features (X) and target variable (y)
X = df[["Mutation Type", "Frequency"]]
y = df["Cancerous"]

# Splitting data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=["Non-Cancerous", "Cancerous"], yticklabels=["Non-Cancerous", "Cancerous"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot the decision boundary
plt.figure(figsize=(8,6))
plt.scatter(df["Frequency"], df["Cancerous"], c=df["Mutation Type"], cmap="coolwarm", edgecolors="k")
plt.xlabel("Mutation Frequency")
plt.ylabel("Cancerous (0 = No, 1 = Yes)")
plt.title("Mutation Frequency vs Cancerous Outcome")
plt.colorbar(label="Mutation Type (0 = Benign, 1 = Harmful)")
plt.show()
