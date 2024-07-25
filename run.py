import mnist_data
import net

training_data, validation_data, test_data = mnist_data.data_wrapper()
network = net.Network([784, 30, 10])
network.SGD(training_data, 30, 10, 0.1, test_data=test_data)
