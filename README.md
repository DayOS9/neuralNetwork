
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

--For the drawing module--  

pillow (PIL): **10.4.0**

# How to use

The three main files you only need are **net.py**, **mnist_data.py**, and the actual data **mnist_data.pkl.gz**.  

After downloading and having the necessary dependencies installed, simply run
> python3 run.py  

Output will be displayed on the terminal such as the current, past epochs, and the number of  
correct images classified.  

The run.py file can be modified to adjust layers, nodes, learning rate, and if the drawing module will be used.

# WIP

This project is still a work-in-progress. Although the drawing module was just recently implemented,  
there is still plans to modify and refine the feature. I plan on making the drawing module more dynamic  
(additional buttons) and run concurrently with the main program for smoothness. Possible additional features  
include real-time analysis and testing when the user is drawing.
