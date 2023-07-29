import tensorflow as tf
from sklearn.preprocessing import StandardScaler

loaded_model = tf.keras.models.load_model('Neural_Network/diabetes_identifier.h5')

userInputtedData = [[1,85,66,29,0,26.6,0.351,31]]
# scaler = StandardScaler()
# userInputtedData = scaler.fit_transform(userInputtedData)

percentprob = loaded_model.predict(userInputtedData)[0][0] * 100
if percentprob >= 50.0:
    prediction = 'Diabetes Positive'
else:
    prediction = 'Diabetes Negative'

print(prediction, f'{percentprob}% of having Diabetes')