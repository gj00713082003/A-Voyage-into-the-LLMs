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
```

## Dataset Summary
- Training: 50,000 images
- Validation: 10,000 images
- Test: 10,000 images

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

##  Model Architecture (`VariableCNN`)

The number of convolutional layers is adjustable using the `num_conv_layers` parameter.

### Features:
- Convolution (3x3, padding=1)
- ReLU activation
- MaxPooling (2x2)

### Classifier:
- Flatten
- Linear (hidden layer of 128 units)
- Output (10-class softmax)


## Training and Validation

- **Epochs**: 3
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Metrics Tracked**: 
  - Training Loss
  - Validation Loss
  - Training Accuracy
  - Validation Accuracy


## 📈 Loss & Accuracy Graphs

### Training vs Validation Accuracy and Loss


<img width="1189" height="490" alt="download" src="https://github.com/user-attachments/assets/674ed3f4-999b-4e67-8047-ffdfb0d57d7a" />

- Slight overfitting detected after 2nd epoch.
- Model generalizes well overall with high accuracy and low validation loss.



## 📊 Number of Conv Layers vs Test Accuracy

### Test Accuracy vs Conv Layers

<img width="709" height="470" alt="download" src="https://github.com/user-attachments/assets/27c35e78-4400-49d9-8708-cd2f5c7f0b95" />


### Test Accuracy Results:
| Conv Layers | Test Accuracy |
|-------------|----------------|
| 1           | 97.54%         |
| 2           | 98.97%         |
| 3           | **99.13%**     |
| 4           | 98.95%         |



##  Conclusions

- **3 Conv Layers Achieved Best Accuracy**: Test accuracy peaked with 3 convolutional layers.
- **Overfitting Detected Beyond Optimal Depth**: Adding a 4th layer slightly reduced performance.
- **Training vs Validation Curve Insights**:
  - Training loss/accuracy improves consistently.
  - Validation performance fluctuates, indicating overfitting risk after a point.
- **Bias vs Variance Tradeoff**:
  - Fewer layers → Underfitting (high bias)
  - Too many layers → Overfitting (high variance)
## Colab Link
https://colab.research.google.com/drive/1GJ8whWnjzzEEqSIp6CnoFlwaj7_LLAo3?usp=sharing
