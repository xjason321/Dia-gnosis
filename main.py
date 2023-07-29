import tensorflow as tf
from sklearn.preprocessing import StandardScaler


loaded_model = tf.keras.models.load_model('Neural_Network/diabetes_identifier.h5')

userInputtedData = [[5, 111, 73, 26, 0, 38, 0.548, 62]]
scaler = StandardScaler()
userInputtedData = scaler.fit(userInputtedData)

predictions = loaded_model.predict(userInputtedData)

print(predictions)