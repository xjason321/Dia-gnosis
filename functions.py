import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv

def ai_predict(userInputtedData):
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

    # df = df.drop(df.tail(1).index,inplace=True)
        
    ori.to_csv('Neural_Network/diabetes.csv', index=False)    

    # formatted = pd.DataFrame({'Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome'})

    percentprob = (loaded_model.predict([userInputtedData])[0][0]) * 100


    if percentprob >= 50.0:
        prediction = 'Diabetes Positive'
    else:
        prediction = 'Diabetes Negative'

    return prediction, percentprob

def fillcsv(csvfile):
    f = open("file_uploads/predicted.csv", "w")
    f.truncate()
    f.close()
    predictool = tf.keras.models.load_model('Neural_Network/diabetes_identifier.h5')
    ori = pd.read_csv(csvfile, delimiter = ',')
    ori_array = np.array(ori)
    print(ori_array)
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    print(ori_array[0])
    for i, row in enumerate(rows):
        if i == 0:
            row.append('Diabetes Prediction')
            row.append('Diabetes Prediction Probability')
        else:
            prediction, percentprob = ai_predict(ori_array[i-1])
            if prediction == 'Diabetes Positive':
                prediction = 1
            elif prediction == 'Diabetes Negative':
                prediction = 0
            row.append(prediction)
            row.append(round(percentprob, 2))
        with open('file_uploads/predicted.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)        
fillcsv('Neural_Network/diabtest.csv')


