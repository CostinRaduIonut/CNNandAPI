# #afisare a entropiei generale ca numar
# import pandas as pd
# import numpy as np
# from scipy.stats import entropy

# def calculate_entropy(data):
#     # Convertim setul de date într-o matrice Numpy
#     data = np.array(data)
    
#     # Calculăm histograma valorilor din setul de date
#     hist, _ = np.histogram(data, bins=256, range=[0, 255])
    
#     # Calculăm probabilitățile
#     probabilities = hist / np.sum(hist)
    
#     # Calculăm entropia
#     entropy_value = entropy(probabilities, base=2)
    
#     return entropy_value

# # Numele fișierului CSV de intrare
# filename = "new_dataset/emnist-balanced-test.csv"

# # Citim datele din fișierul CSV folosind pandas
# data_df = pd.read_csv(filename)

# # Obținem valorile din dataframe sub formă de listă
# data = data_df.values.flatten().tolist()

# # Calculăm entropia
# entropy_value = calculate_entropy(data)

# print("Entropia setului de date este:", entropy_value)

# #afisare entropie per clasa
# import pandas as pd
# import numpy as np
# from scipy.stats import entropy

# def calculate_entropy(data):
#     # Convertim setul de date într-o matrice Numpy
#     data = np.array(data)
    
#     # Calculăm histograma valorilor din setul de date
#     hist, _ = np.histogram(data, bins=256, range=[0, 255])
    
#     # Calculăm probabilitățile
#     probabilities = hist / np.sum(hist)
    
#     # Calculăm entropia
#     entropy_value = entropy(probabilities, base=2)
    
#     return entropy_value

# # Numele fișierului CSV de intrare
# filename = "new_dataset/emnist-balanced-train.csv"

# # Citim datele din fișierul CSV folosind pandas
# data_df = pd.read_csv(filename)

# # Calculăm entropia pentru fiecare clasă
# entropy_values = {}
# for column in data_df.columns:
#     data = data_df[column].values.flatten().tolist()
#     entropy_value = calculate_entropy(data)
#     entropy_values[column] = entropy_value

# Afișăm entropia pentru fiecare clasă
# for column, entropy_value in entropy_values.items():
#     print("Entropia pentru clasa", column, "este:", entropy_value)

#desenare entropie pe clase
# import pandas as pd
# import numpy as np
# from scipy.stats import entropy
# import matplotlib.pyplot as plt

# def calculate_entropy(data):
#     # Convertim setul de date într-o matrice Numpy
#     data = np.array(data)
    
#     # Calculăm histograma valorilor din setul de date
#     hist, _ = np.histogram(data, bins=256, range=[0, 255])
    
#     # Calculăm probabilitățile
#     probabilities = hist / np.sum(hist)
    
#     # Calculăm entropia
#     entropy_value = entropy(probabilities, base=2)
    
#     return entropy_value

# # Numele fișierului CSV de intrare
# filename = "new_dataset/emnist-balanced-test.csv"
# # filename = "A_Z Handwritten Data.csv"

# # Citim datele din fișierul CSV folosind pandas
# data_df = pd.read_csv(filename)

# # Calculăm entropia pentru fiecare clasă
# entropy_values = {}
# for column in data_df.columns:
#     data = data_df[column].values.flatten().tolist()
#     entropy_value = calculate_entropy(data)
#     entropy_values[column] = entropy_value

# # Desenăm graficul
# plt.bar(range(len(entropy_values)), list(entropy_values.values()), tick_label=list(entropy_values.keys()))
# plt.ylabel("Valoare entropie")
# plt.xlabel("Clase")
# plt.title("Entropia pentru fiecare clasă")
# plt.show()



# #afisare mse
# import pandas as pd
# import tensorflow as tf
# import numpy as np

# # Load the CSV file using pandas
# data = pd.read_csv('new_dataset/emnist-balanced-test.csv')

# # Extract the image data and labels from the DataFrame
# x_test = data.iloc[:, :-1].values
# y_true = data.iloc[:, -1].values

# # Load the trained model
# model = tf.keras.models.load_model('nn_digits_and_letters')

# # Convert the image data to the desired format (e.g., reshape if necessary)
# x_test = np.reshape(x_test, (-1, 28, 28, 1))

# # Perform predictions on the test data
# y_pred = model.predict(x_test)

# # Ensure both arrays have the same shape
# y_true = np.reshape(y_true, (-1, 1))

# # Calculate the mean squared error (MSE)
# mse = np.mean((y_true - y_pred)**2)

# print("Mean Squared Error (MSE):", mse)


# #desenare curba de invatare
# import matplotlib.pyplot as plt

# # Încărcați datele de antrenare și valorile metricilor
# train_loss = [1,2,3,4]  # Lista sau array-ul cu valorile pierderii de antrenare
# val_loss = [2,3,4,7]  # Lista sau array-ul cu valorile pierderii de validare
# train_accuracy = [2,3,9]  # Lista sau array-ul cu valorile acurateții de antrenare
# val_accuracy = [1,2,4]  # Lista sau array-ul cu valorile acurateții de validare

# # Trasați curba de învățare pentru pierdere
# plt.figure(figsize=(8, 6))
# plt.plot(train_loss, label='Train Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Curba de învățare - Pierdere')
# plt.xlabel('Epocă')
# plt.ylabel('Pierdere')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Trasați curba de învățare pentru acuratețe
# plt.figure(figsize=(8, 6))
# plt.plot(train_accuracy, label='Train Accuracy')
# plt.plot(val_accuracy, label='Validation Accuracy')
# plt.title('Curba de învățare - Acuratețe')
# plt.xlabel('Epocă')
# plt.ylabel('Acuratețe')
# plt.legend()
# plt.grid(True)
# plt.show()
