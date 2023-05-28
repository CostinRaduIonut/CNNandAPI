import keras
import cv2 as cv
import numpy as np
import tensorflow 
from keras.models import Model
import keras.preprocessing.image as process
from matplotlib import pyplot as plt 
from numpy import expand_dims

model = keras.models.load_model("nn_digits_and_letters")
# model = keras.models.load_model("nn_letters_to_braille_updated")



for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)


model = Model(inputs = model.inputs, outputs = model.layers[1].output)

layers = model.layers

img = process.image_utils.load_img("Untitled.jpg", target_size=(28, 28))
img = np.mean(img, axis=2)
img = process.image_utils.img_to_array(img)
img = expand_dims(img, axis=0)

img /= 255.0

#calculating features_map
features = model.predict(img)

fig = plt.figure(figsize=(20,15))
for i in range(1,features.shape[3]+1):

    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1] , cmap='gray')
    
plt.show()