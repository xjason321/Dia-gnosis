import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv

def predict(userInputtedData):
    ori = pd.read_csv('Neural_Network/diabetes.csv', delimiter=",")

    loaded_model = tf.keras.models.load_model('Neural_Network/diabetes_identifier.h5')

    with open('Neural_Network/diabetes.csv', 'a', newline = '') as f_object:
    
        writer_object = csv.writer(f_object)
        f_object.write('\n')
        writer_object.writerow(userInputtedData)

        f_object.close()

    df = pd.read_csv('Neural_Network/diabetes.csv', delimiter=",")

    X = df.drop('Outcome', axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    userInputtedData = np.array([X[-1]])

    print(userInputtedData)
    print(X[0])

    # df = df.drop(df.tail(1).index,inplace=True)
        
    ori.to_csv('Neural_Network/diabetes.csv', index=False)    

    # formatted = pd.DataFrame({'Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome'})

    percentprob = (loaded_model.predict([userInputtedData]) [0][0]) * 100

    if percentprob >= 50.0:
        prediction = 'Diabetes Positive'
    else:
        prediction = 'Diabetes Negative'

    return prediction, percentprob