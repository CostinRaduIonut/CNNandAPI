from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd 


# pentru litere, se va citi dintr-un fisier csv
# pe prima coloana se afla un nr asociat cu litera corespunzatoare
# iar pe celelalte 28 x 28 coloane se afla imaginea in sine

# rutina ce genereaza un tensor de imagini si un vector de etichete(0 - 25)


# TODO: detecteaza si litere in viitorul apropiat

def prepare_az_dataset(filePath):
    datasetFile = open(filePath, "r")  # se citeste fisierul
    images = []
    labels = []
    # pentru fiecare linie din fisier, le prelucram

    for line in datasetFile:
        line = line.split(",")  # convertim linia intr-un sir de numere
        # primul element reprezinta eticheta ce apartine literei
        label = int(line[0])
        # se ignora primul element fiindca el reprezinta labelul
        image_vector = line[1:]

        # cream un vector unidimensional
        image = np.array([int(x) for x in image_vector], dtype="uint8")

        # cum fiecare linie contine 28x28 elemente, pentru a obtine o imagine el este convertit la o matrice 28x28
        image = image.reshape((28, 28))

        images.append(image)
        labels.append(label)

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int")
    images = np.reshape(images, (372451, 28, 28, 1)) / 255.0
    # images /= 255.0

    return images, labels


def prepare_full_data(filePath, nb=697932):
    datasetFile = open(filePath, "r")  # se citeste fisierul
    images = []
    labels = []
    # pentru fiecare linie din fisier, le prelucram

    for line in datasetFile:
        line = line.split(",")  # convertim linia intr-un sir de numere
        # primul element reprezinta eticheta ce apartine literei
        label = int(line[0])
        # se ignora primul element fiindca el reprezinta labelul
        image_vector = line[1:]

        # cream un vector unidimensional
        image = np.array([int(x) for x in image_vector], dtype="uint8")

        # cum fiecare linie contine 28x28 elemente, pentru a obtine o imagine el este convertit la o matrice 28x28
        image = image.reshape((28, 28))

        images.append(image)
        labels.append(label)

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int")
    images = np.reshape(images, (nb, 28, 28, 1)) 
    # images /= 255.0

    return images, labels


def csv_to_dataset(filePath):
    csvFile = pd.read_csv(filePath)
    labels = csvFile.iloc[:, 0]
    data = csvFile.iloc[:, 1:]

    data = data.values.reshape(-1, 28, 28, 1)
    data = data.astype("float32") / 255.0

    return data, labels 


def prepare_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype("float32") / 255

    test_images = train_images.reshape((60000, 28, 28, 1))
    test_images = train_images.astype("float32") / 255

    return train_images, train_labels, test_images, test_labels


# train_images, train_labels, test_images, test_labels = prepare_mnist_data()
# print(train_images.shape)


def prepare_data():
    data_az, labels_az = prepare_az_dataset("A_Z_Handwritten_Data/A_Z_Handwritten_Data.csv")
    labels_az += 10
    data_mnist, labels_mnist, _, _ = prepare_mnist_data()
    data = np.vstack([data_mnist, data_az])
    labels = np.hstack([labels_mnist, labels_az])
    return data, labels

data,labels= prepare_data()

# print(data.shape)
