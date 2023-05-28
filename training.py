
import keras.activations
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Rescaling, Dropout, BatchNormalization, Activation, RandomZoom, RandomRotation
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import keras
from data_preparation import prepare_data, prepare_az_dataset
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.preprocessing import LabelBinarizer

width = 28
height = 28
TEST_SIZE = .25
EPOCHS = 8

forma_imagine = (width, height, 3)
dimensiuni_imagine = (forma_imagine[0], forma_imagine[1])
BATCH_SIZE = 16

train_images, train_labels = prepare_az_dataset("A_Z_Handwritten_Data/A_Z_Handwritten_Data.csv")

train_images = np.expand_dims(train_images, axis=-1)

le = LabelBinarizer()
train_labels = le.fit_transform(train_labels)


# train_images /= 255.0

# train_images, train_labels = shuffle(train_images, train_labels)

x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size = .25, random_state = 42, shuffle = True)


def get_model():
    inputs = Input(shape=(width, height, 1))
    x = Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)

    outputs = Dense(26, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_model_new():
    inputs = Input(shape=(width, height, 1))
    x = Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dropout(.5)(x)
    outputs = Dense(26, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model_new()

model.summary()

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data = (x_test, y_test))

model.save("nn_letters_to_braille_updated")