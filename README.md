
# neuralNetwork

A neural network designed to recognize images of hand written digits from the [MNIST database.](https://en.wikipedia.org/wiki/MNIST_database)  
> This project was largely influenced by the book "**Neural Networks and Deep Learning** by **Michael Nielson**"  
> The code in this repo was adapted from what is in the book with modifications (such as refactoring, activation functions, other features)  

This neural network can have its structure modified dynamically, specifying the amount of layers  
and the number of neurons per layer.

The data is split three sets (which two are only used for right now): validation, test, and  
training data. The network uses the training data to train the network and uses the test  
data to display how effectively the network has adjusted the weights and biases.  

Optionally, the user can also draw their own digits to be tested against the network.

# Specifications/Dependencies

Python version: **3.11.9**  
(should work with 3.xx.x >=)  

numpy: **2.0.1**  
(have not tested with other versions but should be fine)  

# How to use

The three main files you only need are **net.py**, **mnist_data.py**, and the actual data **mnist_data.pkl.gz**.  

After downloading and having the necessary dependencies installed, simply run
> python3 run.py  

Output will be displayed on the terminal such as the current, past epochs, and the number of  
correct images classified.  

# WIP

This project is still a work-in-progress. There is current development for a feature to allow  
the user to draw a number for it to be tested against the network.
