import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('soil_data.csv')

label_encoder = LabelEncoder()
data['SoilType'] = label_encoder.fit_transform(data['SoilType'])
data['Crop'] = label_encoder.fit_transform(data['Crop'])

X = data.drop('Crop', axis=1)
y = data['Crop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

with open('soil_model.pkl', 'wb') as file:
    pickle.dump(model, file)
