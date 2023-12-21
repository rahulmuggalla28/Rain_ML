from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Scaler.pkl', 'rb') as fl:
    scaler = pickle.load(fl)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = int(request.form['location'])
    min_temp = float(request.form['min_temp'])
    max_temp = float(request.form['max_temp'])
    wind_dir = float(request.form['wind_dir'])
    wind_speed = float(request.form['wind_speed'])
    humidity = float(request.form['humidity'])
    pressure = float(request.form['pressure'])
    cloud = float(request.form['cloud'])
    temp = float(request.form['temp'])
    today_rain = int(request.form['today_rain'])

    input_values = np.array([[location, temp, min_temp, max_temp, wind_speed, wind_dir, humidity, pressure, cloud, today_rain]])

    input_df = scaler.transform(input_values)

    prediction = model.predict(input_df)[0]

    if prediction == 0:
        result = 'There is no rain tomorrow'
    else:
        result = 'There is rain tomorrow'
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)