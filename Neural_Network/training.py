import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Neural_Network/diabetes.csv', delimiter=",")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()

X = scaler.fit_transform(X)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {(accuracy * 100)}%")
print(f'Loss: {loss}')

model.save('Neural_Network/diabetes_identifier.h5')