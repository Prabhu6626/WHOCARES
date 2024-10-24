import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("crop_recommendation.csv")

# Features: Environmental factors
features = ['temperature', 'humidity', 'ph', 'moisture', 'rainfall']
X = data[features]
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training completed and saved as crop_model.pkl")
