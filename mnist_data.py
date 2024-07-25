import gzip
import pickle

import numpy as np

"""import seaborn as sb
from matplotlib import pyplot as plt
from scipy.io import loadmat"""

# opening the data obtaining the actual values and the labels
"""mnist = loadmat("./mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]"""


"""image_size_px = int(np.sqrt(mnist_data.shape[1]))
print("\u2705 [Info] The images size is (", image_size_px, "x", image_size_px, ")")"""


"""def mnist_random_example():
    idx = np.random.randint(70000)
    exp = mnist_data[idx].reshape(image_size_px, image_size_px)
    print("\u2705 [Info] The number in the image below is:", mnist_label[idx])
    plt.imshow(exp, cmap=sb.color_palette("mako", as_cmap=True))
    plt.show()"""


"""def data():
    # split the data into different sets
    data_dev = mnist_data[0:10000]
    dev_labels = mnist_label[0:10000]
    print(data_dev.shape)
    print(dev_labels)"""


def load_data():
    with gzip.open("./mnist.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
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
    training_input = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_result = [correct_output(x) for x in training_data[1]]
    training_data = zip(training_input, training_result)
    validation_input = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_input, validation_data[1])
    test_input = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_input, test_data[1])
    return (training_data, validation_data, test_data)


data_wrapper()
# print(mnist_data[0].reshape(784, 1).dot(2))


# mnist_random_example()
