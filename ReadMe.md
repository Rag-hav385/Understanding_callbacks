# Model Training and Analysis

## Introduction

This code is designed to train and analyze multiple neural network models on a given dataset. The goal is to classify data into two categories, similar to the provided image.

![Data Classification](https://i.imgur.com/33ptOFy.png)

## Getting Started

Follow these steps to get started:

1. Download the dataset from [here](https://drive.google.com/file/d/15dCNcmKskcFVjs7R0ElQkR61Ex53uJpM/view?usp=sharing).

2. Place the downloaded dataset in the project directory.

## Models

The code includes four different models (Model-1, Model-2, Model-3, and Model-4) with various configurations for experimentation. Each model has different activation functions, optimizers, and weight initializations.

### Model-1

- Activation Function: Tanh for hidden layers, Sigmoid for the output layer.
- Optimizer: Stochastic Gradient Descent (SGD) with momentum.
- Weight Initialization: RandomUniform(0,1).

### Model-2

- Activation Function: ReLU for hidden layers, Sigmoid for the output layer.
- Optimizer: Stochastic Gradient Descent (SGD) with momentum.
- Weight Initialization: RandomUniform(0,1).

### Model-3

- Activation Function: ReLU for hidden layers, Sigmoid for the output layer.
- Optimizer: Stochastic Gradient Descent (SGD) with momentum.
- Weight Initialization: He Uniform.

### Model-4

- Customizable model with various parameter values for improved accuracy/F1 score.

## Libraries Used

- Pandas
- NumPy
- TensorFlow
- Keras
- Scikit-Learn

## Dataset Loading

The code loads the dataset from the provided CSV file, splitting it into input features (X) and target values (Y).

## Model Training

Each model is trained using the training data and evaluated on the test data. The training process includes various callbacks and hyperparameters tuning.

## Callbacks

- TerminateNaN: Terminates training if NaN or Inf loss is encountered.
- ModelCheckpoint: Saves the best model weights during training.
- ReduceLROnPlateau: Reduces learning rate during training.
- LearningRateScheduler: Custom learning rate scheduling.
- EarlyStopping: Stops training early if the specified condition is met.
- TensorBoard: Provides visualizations and monitoring during training.

## F1 Score and AUC Score

The code calculates and displays F1 Score and AUC Score to evaluate the model's performance.

## Author

- Raghav Agarwal

Feel free to modify and experiment with the code to achieve better results and gain a deeper understanding of neural network training.
