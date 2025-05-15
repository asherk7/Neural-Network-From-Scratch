# Convolutional Neural Network (CNN) from Scratch  
Creating a Convolutional Neural Network (CNN) from scratch using Python and NumPy, model is a simplified version of the ResNet architecture from the paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385).  

## Overview  
This repository contains an implementation of a convolutional neural network and the popular components, built entirely from scratch using Python and NumPy, modeled after a simple ResNet. The goal of this project is to gain a deep understanding of the inner workings of neural networks, including:  
- Forward propagation  
- Backward propagation  
- Activation functions  
- Loss functions  
- Optimization techniques
- Residual connections

The network will be trained and evaluated on the **CIFAR-10 dataset**, a widely used benchmark for image classification tasks.  

---

## Features  
- Manual implementation of key neural network components:
  - Multiple layers 
  - Non-linear activation functions 
  - Loss functions
  - Gradient descent and backpropagation
  - Residual connections
- Training and evaluation pipeline for the CIFAR-10 dataset

---

## Objectives  
1. Understand the mathematical foundations of neural networks.  
2. Implement a neural network without relying on high-level libraries like TensorFlow or PyTorch.  
3. Build intuition on how different components work together.  

---

## Prerequisites  
- **Python 3.8+**
- **NumPy 2.2.1**
- Basic knowledge of linear algebra, calculus (derivatives & multi-variable), and programming.  

---

## Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/asherk7/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

---

## Usage
1. Make sure you have the required libraries above.
2. Run the cifar script to train the model:  
   ```bash
   python cifar.py
   ```
3. The model will be trained on the CIFAR-10 dataset, and you can monitor the training process through the printed metrics.
4. After training, the model's performance will be evaluated on the test set, and results will be displayed.

---

## References
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Build a Neural Network from scratch in C++ to deeply understand how it works, not just how to use it](https://medium.com/@sirawitchokphantavee/build-a-neural-network-from-scratch-in-c-to-deeply-understand-how-it-works-not-just-how-to-use-008426212f57)
- [Neural Networks from Scratch in Python](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)