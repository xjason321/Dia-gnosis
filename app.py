import urmom
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    # Main website
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pregnancies = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['bloodPressure'])
    skin_thickness = int(request.form['skinThickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetesPedigreeFunction'])
    age = int(request.form['age'])

    submittedInfoMessage = f'''
        <h1>Submitted Information:</h1>
        <p>Pregnancies: {pregnancies}</p>
        <p>Glucose: {glucose}</p>
        <p>Blood Pressure: {blood_pressure}</p>
        <p>Skin Thickness: {skin_thickness}</p>
        <p>Insulin: {insulin}</p>
        <p>BMI: {bmi}</p>
        <p>Diabetes Pedigree Function: {diabetes_pedigree_function}</p>
        <p>Age: {age}</p>
    '''
    
    userSubmittedInfo = [pregnancies, glucose, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

    prediction, percentprob = urmom.ai_predict(userSubmittedInfo)

    return render_template('predict.html', submittedInfoMessage=submittedInfoMessage, prediction=prediction, percentprob=percentprob)

if __name__ == '__main__':
    app.run()