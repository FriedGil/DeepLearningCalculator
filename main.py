import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("addition.csv")
df.head(5)
data = df[["num_1", "num_2"]]
target = df[["sum"]]
x_train, x_valid, y_train, y_valid = train_test_split(data, target)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_valid = np.asarray(x_valid)
y_valid = np.asarray(y_valid)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.fit_transform(x_valid)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(30, activation="relu", input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))
mse_test = model.evaluate(x_valid, y_valid)
x_new = x_test[:3]
y_pred = model.predict(x_new)

# model.predict([[int1,int2]])
