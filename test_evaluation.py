from sklearn.naive_bayes import LabelBinarizer
from data_preparation import prepare_full_data
import keras

print("Loading testing data...")
x_test, y_test = prepare_full_data("new_dataset/emnist-balanced-test.csv", 18800)



model = keras.models.load_model("nn_digits_and_letters")

le = LabelBinarizer()
y_test = le.fit_transform(y_test)

score = model.evaluate(x_test, y_test, verbose = 0)

# print(score)