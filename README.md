# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset is a well-known dataset in the field of machine learning, consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to build a deep learning model capable of classifying images from the CIFAR-10 dataset into one of ten classes. The model is built using TensorFlow and Keras, and it achieves a test accuracy of approximately 71.94%.

## Dataset

The CIFAR-10 dataset contains 60,000 32x32 color images across 10 different classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is split into 50,000 training images and 10,000 test images.

## Model Architecture

The Convolutional Neural Network (CNN) used in this project has the following architecture:

- **Conv2D Layer**: 32 filters, kernel size of 3x3, ReLU activation
- **MaxPooling2D Layer**: Pool size of 2x2
- **Conv2D Layer**: 64 filters, kernel size of 3x3, ReLU activation
- **MaxPooling2D Layer**: Pool size of 2x2
- **Flatten Layer**: Converts the 2D matrices into a 1D vector
- **Dense Layer**: 64 units, ReLU activation
- **Dropout Layer**: 50% dropout rate to reduce overfitting
- **Dense Layer**: 10 units, Softmax activation (for the 10 classes)

The model is trained using the Adam optimizer and categorical cross-entropy loss.

## Results

The model was trained for 10 epochs, and the results are as follows:

- **Training Accuracy**: Approximately 72.47%
- **Test Accuracy**: Approximately 71.94%

![Model Accuracy](path_to_accuracy_plot.png"C:\Users\dell\Pictures\Screenshots\Screenshot 2024-08-30 011945.png")
*Accuracy over training epochs.*

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cifar10-image-classification.git
   cd cifar10-image-classification
   ```

2. Create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install tensorflow keras numpy matplotlib seaborn
   ```
## Usage

1. Run the training script to train the model:
   ```bash
   python image_classification.py
   ```

2. The script will train the model and display accuracy and loss plots.

3. The trained model will be saved as `cifar10_cnn_model.h5`.

## Future Improvements

- **Increase Model Accuracy**: Experiment with different architectures, hyperparameters, and data augmentation techniques to improve the model’s accuracy.
- **Add More Layers**: Explore deeper CNN architectures to potentially enhance performance.
- **Optimize Training Time**: Implement techniques like transfer learning or distributed training to speed up the model training process.
- **Deploy the Model**: Consider deploying the trained model as a web service using tools like Flask or FastAPI.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The CIFAR-10 dataset is provided by the Canadian Institute For Advanced Research.
- Thanks to the developers of TensorFlow and Keras for their powerful and easy-to-use deep learning frameworks.
- Special thanks to all contributors and the open-source community for their support and resources.







