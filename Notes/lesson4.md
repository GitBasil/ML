## Artificial Neural Networks (ANNs)

![alt text](image-3.png)

Each neuron receive a data then do the calculation and produce a single value then send it to the next layer.

A single neuron perform the below formula which is the regression + activation function to produce the single value that it will pass to the next layer.

- f is the activation
- w is the wights (coefficients)
- x is the features
- b is the bias

![alt text](image-5.png)

![alt text](image-4.png)

**Perceptron** is the simplest form of neural network, which is one hidden layer with one neuron in it.

### how it work:

- first get random weights
- define the learning rate
- call the predict function
- check the loss function
- update the weights
- call the predict again
- stopping the loop depends on the need (epochs, until it reach specific loss or if the loss is not enhancing in n number of tries)
