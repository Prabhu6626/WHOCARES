from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('crop_model.pkl', 'rb') as f:
    crop_model = pickle.load(f)

with open('soil_model.pkl', 'rb') as f:
    soil_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/organic_fertilizers')
def organic_fertilizers():
    return render_template('organic_fertilizers_details.html')

@app.route('/organic_pesticides')
def organic_pesticides():
    return render_template('organic_pesticides_details.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        moisture = float(request.form['moisture'])
        rainfall = float(request.form['rainfall'])

        input_data = [[temperature, humidity, ph, moisture, rainfall]]
        prediction = crop_model.predict(input_data)[0]
        return jsonify({'prediction': prediction})
    
    return render_template('predict.html') 

@app.route('/terrace_farming_crops')
def terrace_farming_crops():
    return render_template('terrace_farming_crops.html')

@app.route('/terrace_farming_crops_predict', methods=['POST'])
def predict_crops():
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    moisture = float(request.form['moisture'])
    rainfall = float(request.form['rainfall'])

    prediction = crop_model.predict([[temperature, humidity, ph, moisture, rainfall]])
    
    return jsonify({'prediction': prediction[0]})

@app.route('/soil_predict')
def soil_predict():
    return render_template('soil_predict.html')

@app.route('/soil_predict', methods=['POST'])
def soil_predict_post():
    soil_type = request.form.get('soil_type')
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    ph = float(request.form.get('ph'))
    moisture = float(request.form.get('moisture'))
    rainfall = float(request.form.get('rainfall'))

    input_data = pd.DataFrame({
        'SoilType': [soil_type],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'pH': [ph],
        'Moisture': [moisture],
        'Rainfall': [rainfall]
    })

    input_data = pd.get_dummies(input_data)

    if hasattr(soil_model, 'feature_importances_'):
        input_data = input_data.reindex(columns=soil_model.feature_importances_.index, fill_value=0)

    prediction = soil_model.predict(input_data)[0]

    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
