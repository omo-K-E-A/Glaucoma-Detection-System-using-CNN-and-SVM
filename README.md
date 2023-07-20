# Glaucoma-Detection-System-using-Deep Learning

This repository contains code for a deep learning model that detects glaucoma in fundus images. The model is built using TensorFlow and Keras and utilizes transfer learning with the MobileNetV2 architecture. The model is trained on a curated dataset of healthy and glaucoma fundus images and can classify fundus images as either healthy or showing signs of glaucoma.

# Installation
To run the code in this repository, you need to have the following libraries and dependencies installed:

TensorFlow
Keras
NumPy
pandas
matplotlib
scikit-learn
You can install the required libraries using pip:

Copy code
pip install tensorflow keras numpy pandas matplotlib scikit-learn

# Dataset
The fundus image dataset used for training and testing the model can be found in the Multichannel Glaucoma Benchmark Dataset on Kaggle. The dataset includes fundus images for both healthy and glaucoma cases.

# Usage
Clone the repository to your local machine

Download the fundus image dataset from the Kaggle dataset and place it in the appropriate directory.

Run the Jupyter notebook glaucoma_detection.ipynb to train the model and evaluate its performance.

# Model Architecture
The glaucoma detection model is based on the MobileNetV2 architecture with additional layers added for fine-tuning. Transfer learning is used to leverage the pre-trained MobileNetV2 model for fundus image analysis.

# Metrics
The model is evaluated using various metrics, including accuracy, precision, recall, AUC (Area Under the Curve), and F1-score.

# Results
The trained model achieves promising accuracy in detecting glaucoma in fundus images. The performance metrics and evaluation results are provided in the Jupyter notebook.

# Contributing
Contributions to this repository are welcome. Please feel free to open a pull request for any enhancements or bug fixes.

# Contact
For any questions or inquiries, please contact koladeemmanuela32@gmail.com.
