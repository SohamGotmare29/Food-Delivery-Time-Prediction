from flask import Flask, render_template, request
import pickle
import numpy as np

#logging
import logging
logging.basicConfig(filename="scrapper.log" , level=logging.DEBUG)

#load model 
model = pickle.load(open('newmodel.pkl', 'rb'))

#Initializing a Flask API
application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('indexs.html', result=None)  # Pass result as None initially

@app.route('/predict', methods=['POST','GET'])
def predict_time():
        Road_traffic_density = int(request.form.get('Road_traffic_density'))
        multiple_deliveries = int(request.form.get('multiple_deliveries'))
        Delivery_person_Ratings = int(request.form.get('Delivery_person_Ratings'))
        Vehicle_condition = int(request.form.get('Vehicle_condition'))
        Weather_conditions = int(request.form.get('Weather_conditions'))
        Type_of_vehicle = int(request.form.get('Type_of_vehicle'))
        Distance = float(request.form.get('Distance'))
        Delivery_person_Age = int(request.form.get('Delivery_person_Age'))

        features = [Road_traffic_density, multiple_deliveries, Delivery_person_Ratings,
                    Vehicle_condition, Weather_conditions, Type_of_vehicle, Distance, Delivery_person_Age]

        input_data = np.array(features).reshape(1, -1)

        # prediction
        result = model.predict(input_data)

        return render_template('indexs.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
