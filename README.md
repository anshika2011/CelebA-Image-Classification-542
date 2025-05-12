Binary Facial Attribute Classification: CelebA Dataset

Project Overview

This project aims to classify the binary attribute "Smiling" from the CelebA dataset using a variety of models, ranging from classical machine learning approaches to advanced deep learning methods such as Bayesian Neural Networks (BNNs) and Vision Transformers (ViT).

Dataset

The CelebA dataset contains 202,599 celebrity face images with binary attribute labels, including "Smiling." Each image is annotated with 5 facial landmarks.

Challenges

High Dimensionality: Images have a size of 218×178×3, leading to high-dimensional input vectors.

Overfitting Risk: Classical models may suffer from the curse of dimensionality.

Computationally Intensive: Training deep models like ViT can be resource-heavy.

Project Structure

The project structure is organized as follows:
'''
project_root
├── data/                 # Contains raw and preprocessed data
├── models/            # Jupyter Notebooks for model training and evaluation
│   ├── logistic_lda_qda.ipynb    # Logistic Regression
│   ├── SVM.ipynb         # Support Vector Machines (Linear and RBF)
│   ├── BayesianNeuralNet.ipynb    # Bayesian Neural Networks
│   └── VisionTransformer.ipynb         # Vision Transformer scripts/
│   ├── RandomForestXGboost.py  # Random Forest and Xgboost Classification
├── README.md             # Project description and instructions
└── requirements.txt      # Python dependencies
'''

Installation

To run the project, first clone the repository and install the required packages:

git clone https://github.com/username/CelebA-Image-Classification-542.git
cd CelebA-Image-Classification-542
pip install -r requirements.txt

Usage

Running Models:
Each model can be run individually from the respective Jupyter Notebook or Python script.

## Key Findings
- ViT outperforms classical models and BNNs due to its ability to capture spatial structure in image data.
- BNNs are computationally expensive and fail to leverage spatial information effectively.
- Classical models such as Logistic Regression and SVM perform well with dimensionality reduction (PCA), but underperform compared to ViT.

## Contributors
- Alarsh Tiwari
- Anshika Pradhan
- Bohan Li
- Momin Shah
- Sarthak Morj

