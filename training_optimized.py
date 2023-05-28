import tensorflow as tf
import keras.activations
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Rescaling, Dropout, BatchNormalization, Activation, RandomZoom, RandomRotation
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import keras
from data_preparation import prepare_data, prepare_az_dataset, prepare_full_data, csv_to_dataset
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator


width = 28
height = 28
TEST_SIZE = .25
EPOCHS = 15
BATCH_SIZE = 64
rootdir = "new_dataset"
NB_CLASSES = 47


train_images, train_labels = csv_to_dataset("new_dataset/emnist-balanced-train.csv")
validation_images, validation_labels = csv_to_dataset("new_dataset/emnist-balanced-test.csv")

train_images = np.expand_dims(train_images, axis=-1)

print(f"val labels:{validation_labels[:5]}")


le = LabelBinarizer()
train_labels = le.fit_transform(train_labels)
validation_labels = le.fit_transform(validation_labels)



# train_images /= 255.0

# train_images, train_labels = shuffle(train_images, train_labels)

# x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size = .25, random_state = 42, shuffle = True)

callbacks =[
    tf.keras.callbacks.EarlyStopping(patience=5, monitor="loss", verbose=1),
]

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(.25),
    tf.keras.layers.RandomTranslation(height_factor= .13, width_factor= .13),
])

rescaling = tf.keras.Sequential([
    Rescaling(scale = 1. / 255)
])

def get_model_new():
    inputs = Input(shape=(width, height, 1))
    x = data_augmentation(inputs)
    x = Conv2D(filters=32, kernel_size=(3,3))(x)
    x = MaxPooling2D(pool_size=(2,2 ))(x)

    x = Conv2D(filters=64, kernel_size=(3,3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # x = Conv2D(filters=64, kernel_size=(3,3), activation="relu" )(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(units=128, activation="relu")(x)
    x = Dropout(.5)(x)
    outputs = Dense(NB_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model_new()

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = .001), loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(train_images, train_labels, epochs = EPOCHS, batch_size = BATCH_SIZE, callbacks= callbacks, validation_data=(validation_images, validation_labels))

model.save("nn_digits_and_letters")