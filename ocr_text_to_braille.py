import keras
import numpy as np
import tensorflow as tf
import string

# sterge


litere = []
litere[:0] = string.ascii_uppercase
clase_categorii = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + litere


model = keras.models.load_model("nn_text_to_braille")


def converteste_imagine(cale_imagine):
    img = tf.keras.utils.load_img(
        cale_imagine
    )

    img_tensor = tf.keras.utils.img_to_array(img)
    img_tensor = tf.expand_dims(img_tensor, 0)
    img_tensor = tf.image.resize(img_tensor, (32, 32))

    predictii = model.predict(img_tensor)
    predictie = predictii[0]

    return clase_categorii[np.argmax(predictie)]


def converteste_imagine_lista(imagine_lista):
    img_tensor = imagine_lista[:]
    img_tensor = tf.expand_dims(img_tensor, 0)
    img_tensor = tf.image.resize(img_tensor, (32, 32))

    precitii = model.predict(img_tensor)
    predictie = precitii[0]

    return clase_categorii[np.argmax(predictie)], np.max(predictie)


# print(converteste_imagine("imgtest.jpg"))
