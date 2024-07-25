import gzip
import pickle

import numpy as np


def load_data():
    """Although there are many sites where the mnist_data can be downloaded from, this specific file was taken from
    Michael Nielsen https://github.com/mnielsen/neural-networks-and-deep-learning.git"""
    with gzip.open("./mnist.pkl.gz", "rb") as f:
        """each variable will take on a tuple where the first entry will contain a list of the images and the second entry will contain
        a list of the labels"""
        training_data, validation_data, test_data = pickle.load(
            f, encoding="latin1"
        )  # encoding set to latin1 since utf did not work
        return training_data, validation_data, test_data


def correct_output(x):
    # returns a 10 x 1 vector which is to represent the correct output for the image
    # essentially this is the label for the image associated
    v = np.zeros((10, 1))
    v[x] = 1.0  # 1.0 because 1 being 100 percent positive this is the right answer
    return v


def data_wrapper():
    # here we are going to reshape the data to be in the correct dimensions for us to manipulate
    training_data, validation_data, test_data = load_data()
    # the 28x28 images will be transformed into 784x1 numpy array (kinda like a matrix/vector)
    training_input = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_result = [correct_output(x) for x in training_data[1]]
    # here we zip them up again to their original format as being a tuple with the first entry being the data, and the second the labels
    training_data = zip(training_input, training_result)
    validation_input = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_input, validation_data[1])
    test_input = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_input, test_data[1])
    return (training_data, validation_data, test_data)
