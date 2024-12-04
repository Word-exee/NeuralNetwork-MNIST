# NeuralNetwork-MNIST

A customizable neural network implemented from scratch using Python and NumPy, designed to train on the MNIST dataset.

## Features

### Custom Neural Network Class (`NeuralNetwork`)

- **Define the number of layers and neurons per layer**: Easily configure the architecture of your neural network by specifying the number of layers and the number of neurons in each layer.
- **Choose activation functions**: Supports a variety of activation functions including:
  - Sigmoid
  - Tanh
  - ReLU
  - Leaky ReLU
  - Softmax (used in the output layer)
- **Select weight initialization methods**: Initialize your network's weights using different strategies:
  - Zero Initialization
  - Random Initialization
  - Normal Initialization (Normal distribution with mean 0 and standard deviation 1)
- **Set training parameters**:
  - Learning rate
  - Number of epochs
  - Batch size
- **Training and Evaluation Functions**:
  - `fit`: Train the model on input data.
  - `predict`: Predict class labels for input data.
  - `predict_proba`: Predict class-wise probabilities for input data.
  - `score`: Evaluate the accuracy of the trained model.

### Activation Functions

- **Sigmoid**: Smoothly maps input values to an output range between 0 and 1.
- **Tanh**: Maps input values to an output range between -1 and 1.
- **ReLU (Rectified Linear Unit)**: Outputs the input directly if it is positive; otherwise, it outputs zero.
- **Leaky ReLU**: Allows a small, non-zero gradient when the unit is not active.
- **Softmax**: Converts logits to probabilities, typically used in the output layer for classification tasks.

### Weight Initialization Methods

- **Zero Initialization**: Initializes all weights to zero. (Note: Not recommended for deep networks as it can lead to symmetry problems.)
- **Random Initialization**: Initializes weights randomly, typically using a uniform distribution.
- **Normal Initialization**: Initializes weights using a normal distribution with mean 0 and standard deviation 1.

### Training on MNIST Dataset

- **Data Preprocessing and Normalization**: Preprocesses the MNIST data by normalizing pixel values and preparing it for training.
- **80:10:10 Train-Validation-Test Split**: Divides the dataset into training, validation, and testing sets with an 80:10:10 ratio.
- **Configurable Training Parameters**: Allows customization of training parameters such as learning rate, number of epochs, and batch size.
- **Visualization of Training and Validation Loss**: Plots the loss curves for both training and validation sets over epochs to monitor the training process.

### Model Persistence

- **Save Trained Models as `.pkl` Files**: Enables saving of trained models for later use and demonstration, allowing you to load and use them without retraining.
Link for MNIST Dataset to be used : https://www.kaggle.com/datasets/hojjatk/mnist-dataset
