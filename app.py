import functions
from flask import Flask, render_template, request

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
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
        <p class="u-text u-text-5">Pregnancies: {pregnancies}</p>
        <p class="u-text u-text-5">Glucose: {glucose}</p>
        <p class="u-text u-text-5">Blood Pressure: {blood_pressure}</p>
        <p class="u-text u-text-5">Skin Thickness: {skin_thickness}</p>
        <p class="u-text u-text-5">Insulin: {insulin}</p>
        <p class="u-text u-text-5">BMI: {bmi}</p>
        <p class="u-text u-text-5">Diabetes Pedigree Function: {diabetes_pedigree_function}</p>
        <p class="u-text u-text-5">Age: {age}</p>
    '''
    
    userSubmittedInfo = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

    prediction, percentprob = functions.ai_predict(userSubmittedInfo)
    percentprob = (str(round(percentprob, 2)) + "%") # format to two decimals

    if prediction == "Diabetes Positive":
        prediction_message = "Patient with entered information might have a risk of diabetes. It might be a good idea to perform tests to double-check."
    else:
        prediction_message = "Patient with entered information doesn't appear to have a risk of diabetes. However, it still might be a good idea to perform tests to double-check."

    return render_template('predict.html', submittedInfoMessage=submittedInfoMessage, prediction=prediction, percentprob=percentprob, predictionMsg=prediction_message)

if __name__ == '__main__':
    app.run()