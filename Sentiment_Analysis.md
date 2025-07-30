#  MNIST Digit Classification using Variable CNN (PyTorch)

This repository contains an implementation of a customizable Convolutional Neural Network (CNN) in PyTorch for classifying handwritten digits from the MNIST dataset. The architecture is designed to be flexible, allowing the number of convolutional layers to be adjusted to explore the trade-off between bias and variance.

---

## Project Overview

- **Frameworks Used**: PyTorch, Torchvision
- **Dataset**: MNIST (28×28 grayscale images of digits 0–9)
- **Objective**: Build a CNN model with variable depth and analyze its performance on digit classification.
- **Design**: Object-Oriented Programming (OOP) with configurable layer depth.
- **Key Features**:
  - Dynamic CNN architecture
  - Dataset normalization
  - Data splitting (training/validation/test)
  - Accurate classification of digits

---

## Data Preprocessing

The MNIST dataset contains 8-bit grayscale images with pixel values ranging from 0 (black) to 255 (white). To optimize training, images are:

- Converted to PyTorch tensors with values in range [0.0, 1.0]
- Normalized using:
  - **Mean**: `0.1307`
  - **Standard Deviation**: `0.3081`

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

