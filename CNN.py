import tensorflow as tf
from tensorflow import keras
import numpy as np
from extra_keras_datasets import emnist 
(x_train, y_train), (x_test, y_test) = emnist.load_data(type='balanced')
print("x train shape is: " , x_train.shape)
print("x test shape is: " , x_test.shape)
print("y train shape is: " , y_train.shape)
print("y test shape is: " , y_test.shape)
var = int(x_test.shape[0]/2)
print(var)
x_val = x_test[0:var,:,:]
y_val = y_test[0:var]

x_test2 = x_test[var:,:,:]
y_test2 = y_test[var:]

#to avoid shape 2D .... error
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_val  = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test2 = x_test2.reshape(x_test2.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

#set the CNN Architecture
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3),padding="same",
        activation='relu', #stride and padding my default 1,1 and valid  
        input_shape=(28,28,1)), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
    pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, (3,3), padding="same",
        activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
    pool_size=(2,2)),
        tf.keras.layers.Conv2D(128, (3,3), padding="same",
        activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
    pool_size=(2,2)),
        tf.keras.layers.Conv2D(128, (3,3), padding="same",
        activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
    pool_size=(2,2)),
        tf.keras.layers.Conv2D(128, (3,3), padding="same",
        activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(47,activation='softmax')


]
)


opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, # Optimizer  Defaults learning rate 0.001
             loss='sparse_categorical_crossentropy', # Loss function to minimize
             metrics=['accuracy'])


#fit the model
model.fit(x_train, y_train, batch_size=32, epochs=8, verbose=1,
          validation_data=(x_val, y_val)) # Reserve  samples for validation
#Evaluate the model on the test data using `evaluate`
modelLoss, modelAccuracy = model.evaluate(x_test2, y_test2) 
print(modelLoss)
print(modelAccuracy)


# Save the model weights for future reference
model.save('model')
model.summary()
