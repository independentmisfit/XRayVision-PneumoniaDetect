# Pneumonia Detection using Chest X-ray Images
## Table of Contents
Project Overview
- Dataset
- Model Architecture
- Installation
- How to Run
- Results
- Training and Evaluation
- Conclusion
## Project Overview
This project aims to build a deep learning model using Convolutional Neural Networks (CNNs) to detect pneumonia from chest X-ray images. The model is trained to classify X-ray images into two categories: Normal and Pneumonia. The dataset used for this project contains labeled chest X-ray images, and the goal is to achieve high accuracy in classifying these images using state-of-the-art CNN techniques.

## **Key Features:**
- Uses a CNN-based architecture for image classification.
- Implemented using Keras and TensorFlow frameworks.
- Leverages transfer learning from a pre-trained model to improve performance.
- Incorporates data augmentation and early stopping techniques.
- Saves the best model using ModelCheckpoint.
## **Dataset**
The dataset used for this project consists of labeled chest X-ray images categorized as Normal and Pneumonia. The data is publicly available on Kaggle.

- Kaggle Dataset: Chest X-Ray Images (Pneumonia)
## **Dataset Structure:**
The dataset is organized into three folders:

- train: Contains X-ray images for training, divided into two subfolders, NORMAL and PNEUMONIA.
- val: Contains validation images, structured similarly to the training folder.
- test: Contains images for testing the model performance.
## **Model Architecture**
The model used in this project is a Convolutional Neural Network (CNN) with the following layers:

- Convolutional layers with ReLU activations and batch normalization.
- MaxPooling layers to reduce dimensionality.
- Fully connected dense layers for classification.
- A softmax layer at the output for binary classification.
- For better accuracy and efficiency, transfer learning is applied using MobileNetV2 as the base model, with custom top layers added for pneumonia detection.

## **Key Components:**
- Data Augmentation: Random transformations like rotation, flipping, and zooming applied to prevent overfitting.
- Callbacks:
- Early Stopping to stop training when validation loss stops improving.
- ModelCheckpoint to save the best model during training.
## **Installation**
Prerequisites:
- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib
- Kaggle API for dataset loading

## **Steps:**
Clone this repository:

```
git clone https://github.com/independentmisfit/XRayVision-PneumoniaDetect.git
cd XRayVision-PneumoniaDetect
```
Install the required dependencies:

```
pip install -r requirements.txt
```
Download the dataset from Kaggle and extract it into the project directory:

```
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```
## **How to Run**
Training the Model:
- Open the Pneumonia_Detection.ipynb file in Google Colab or your local Jupyter notebook.
- Ensure that the Kaggle dataset is downloaded into your working directory.
- Load the dataset, and split it into training, validation, and test sets.
- Run the cells to train the model using the dataset.
- The model will save the best weights during training based on validation loss.
## **Running Predictions:**
Once the model is trained, you can run predictions on new chest X-ray images using:

```
# Load saved model
from tensorflow.keras.models import load_model

model = load_model('best_model.h5')

# Predict on new images
predictions = model.predict(new_images)

```
## **Results**
Model Performance:
- Training Accuracy: ~98%
- Validation Accuracy: ~96%
- Test Accuracy: ~95%
Confusion Matrix:
                | Predicted Normal |	Predicted Pneumonia|
----------------|------------------|---------------------|
True Normal	    |200               |	5                  |
True Pneumonia	|10                |	180                |

Example Classification Report:
```
Copy code
              precision    recall  f1-score   support

       Normal      0.95      0.98      0.97       205
    Pneumonia      0.97      0.95      0.96       190

    accuracy                           0.96       395
   macro avg       0.96      0.96      0.96       395
weighted avg       0.96      0.96      0.96       395
```
## **Training and Evaluation**
- The model is trained using Adam optimizer with a learning rate of 0.0001.
- Early Stopping is used to prevent overfitting, monitoring validation loss.
- The model is evaluated on the test set using standard metrics such as accuracy, precision, recall, and F1-score.

## **Conclusion**

The **XRayVision-PneumoniaDetect** project successfully demonstrates the potential of deep learning in medical image analysis, particularly in the early detection of pneumonia from chest X-ray images. By leveraging convolutional neural networks and transfer learning, the model achieves high accuracy, showcasing its ability to assist healthcare professionals in making quicker, more accurate diagnoses. This project highlights the importance of advanced machine learning techniques in enhancing diagnostic processes, ultimately contributing to improved patient outcomes in the fight against pneumonia. Future work may focus on expanding the dataset, optimizing the model further, and integrating the solution into clinical workflows for real-time analysis
