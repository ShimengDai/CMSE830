cd "C:\Users\daish\OneDrive\Desktop\Project"
conda activate X # make new environment for the project
python 0_setup.py
python 1_prepare_data.py
python 2_train_model.py
python 3_test_model.py
python 4_predict.py

python 0_setup.py
python 1_prepare_data.py
python 2_nonDNN_train_model.py
python 3_nonDNN_test_model.py
python 4_nonDNN_predict.py


# The 0_setup.py 

script is designed to prepare your project environment by setting up necessary folders, copying configuration files, and ensuring that required NLTK resources are available. 

# 1_prepare_data.py

The 1_prepare_data.py script is responsible for preparing the dataset for the tweet classification task.

# 2_train_model.py

The 2_train_model.py script is responsible for training a machine learning model on the preprocessed and feature-extracted tweet data. 

# 3_test_model.py

The 3_test_model.py script is designed to evaluate the performance of the trained machine learning model on the test dataset. 

# 4_predict.py

The 4_predict.py script is designed to generate predictions on new, unseen data using the trained machine learning model.  

# conf.yaml


# utils

The conf.yaml file contains various settings and configurations for your tweet classification project.

## dataset.py

The dataset.py script contains utility functions and a custom dataset class specifically designed to handle text data for your tweet classification project. 

TextDataset: Wrap your preprocessed data into a TextDataset object before passing it to a PyTorch DataLoader.
clean_data: Clean your dataset before starting any feature extraction or model training.
balance_data: Balance your dataset if you're dealing with class imbalance.
print_data_dims: Use this function to check the dimensions of your datasets after splitting and preprocessing

## feature_extraction.py

The feature_extraction.py script contains classes and methods for extracting features from text data, which are essential for converting raw text into numerical representations that can be fed into machine learning models

## models.py 

The models.py script contains the definitions of neural network models used for your tweet classification project. 

## preprocessing.py
The preprocessing.py script contains various text preprocessing functions that are essential for cleaning and preparing text data before feature extraction and model training.

## utils.py

The utils.py script contains utility classes that assist with model evaluation, logging, and saving during the training and testing phases of your project. These utilities are critical for tracking the progress of your model's training and ensuring that you retain the best-performing version for deployment or further analysis.

# Tweet classification using deep neural networks
## Getting started
To start with, install the required packages. It is recommended that you do so by using Anaconda and creating an environment for this project (e.g., `tweet_analyis` or a name of your choice). Activate the environments and run `conda install pip`. Then, `cd` to where you have your local copy of this repository and install the packages in `requirements.txt` by running `pip install -r requirements.txt`.

Detailed steps for setting up for the experiment and training the model will be given.
