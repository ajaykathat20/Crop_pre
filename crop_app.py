
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect
app = Flask(__name__)
pickle_in = open(r"E:\Crop yieldss Prediction\Crop_model.pkl","rb")
Crop_model=pickle.load(pickle_in)
dtr = pickle.load(open(r'E:\Crop yieldss Prediction\model.pkl','rb'))
preprocessor = pickle.load(open(r'E:\Crop yieldss Prediction\pre.pkl','rb'))


@app.route('/')
def home():
    return render_template('Home_1.html')

@app.route('/Predict')
def prediction():
    return render_template('Index.html')

@app.route('/form', methods=["POST"])
def brain():
    Nitrogen=float(request.form['Nitrogen'])
    Phosphorus=float(request.form['Phosphorus'])
    Potassium=float(request.form['Potassium'])
    Temperature=float(request.form['Temperature'])
    Humidity=float(request.form['Humidity'])
    Ph=float(request.form['ph'])
    Rainfall=float(request.form['Rainfall'])
     
    values=[Nitrogen,Phosphorus,Potassium,Temperature,Humidity,Ph,Rainfall]
    
    if Ph>0 and Ph<=14 and Temperature<100 and Humidity>0:
        arr = [values]
        acc = Crop_model.predict(arr)
        # print(acc)
        return render_template('prediction.html', prediction=str(acc))
    else:
        return "Sorry...  Error in entered values in the form Please check the values and fill it again"
@app.route('/yield')
def crop_yield():
    return render_template('crop_yield.html')


@app.route("/predict_yield",methods=['POST'])
def predict_yield():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']

        features =[[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]]
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features)

        return render_template('crop_yield.html',prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)















