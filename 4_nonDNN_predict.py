import os
import yaml
import re, string
import pickle
import torch
import random
import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_text, remove_stopwords, get_wordnet_pos, lemmatizer, preprocessing_compose
from tqdm import tqdm

def prepare_new_data(args):
    """
    Prepare the data for which you want to use the trained model to generate predictions. This dataset is called "new data."
    Returns: 1) A copy of the new data with cleaned text; 2) features extracted from the cleaned text, used later as inputs to the classification model to generate predictions
    """
    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader = yaml.FullLoader)
    new_data_path = os.path.join(config['Corpus']['data_dir'], config['Corpus']['new_data_filename'])
    text_col = config['New_data_setup']['text_col']
    include_first_n = config['New_data_setup']['include_first_n']
    feat_extraction_method = config['Feature_extraction']['method']
    
    # Get new data
    new_data = pd.read_csv(new_data_path)
    # print(f'No. of tweets: {len(new_data.index)}') ##
    
    # Check if you want to include just the first n rows in the new data:
    if include_first_n:
        new_data = new_data.head(include_first_n)
    
    # Drop rows with NaNs in any columns
    new_data = new_data.dropna(how = 'all')
    print(f'No. of valid tweets: {len(new_data.index)}')
    
    # Data cleaning
    print('Preprocessing tweets in new data...')
    tqdm.pandas()
    new_data['clean_text'] = new_data[text_col].progress_apply(lambda x: preprocessing_compose(x))
    print('Done')
    
    # print(f'No. of tweets after cleaning: {len(new_data.index)}') ##
    
    # Load the fitted feature extractor
    print(f'Extracting features using {feat_extraction_method}...')
    feat_extractor_path = os.path.join(args.exp_dir, 'trained_models', feat_extraction_method + '_fitted.pkl')
    with open(feat_extractor_path, 'rb') as fe:
        feat_extractor = pickle.load(fe)
    new_data_feats = feat_extractor.transform(new_data['clean_text'])
    print('Dimension of extracted features:', new_data_feats.shape)
    
    return new_data, new_data_feats






def get_predictions(new_data, new_data_feats, args):
    """
    Get predicted labels for new data.
    """
    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    n_classes = config['Data_setup']['n_classes']
    pred_col = config['New_data_setup']['pred_col']
    classifier = config['NonDNN_setup']['classifier']
    seed = config['Data_setup']['random_seed']
    random.seed(seed)
    
    # Get the trained model and predict
    print(f'Classifier: {classifier}')
    model_path = os.path.join(args.exp_dir, 'trained_models', classifier + '_best.pickle')
    model = pickle.load(open(model_path, 'rb'))
    y_pred, _ = model.predict(new_data_feats)
    
    # Add the predictions to the dataframe of new data
    new_data[pred_col] = y_pred
    new_data = new_data.drop(columns=['clean_text'])
    
    # Do a prediction summary
    print('Prediction summary:')
    print(new_data[pred_col].value_counts())
    
    # Save the dataframe with predictions
    pred_path = os.path.join(args.exp_dir, 'predictions')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
        print(f"  Folder {pred_path} doesn't exist. Create it.")
    
    output_fn = config['Corpus']['new_data_filename'].split('.')[0] + '_with_preds.csv'
    output_path = os.path.join(pred_path, output_fn)
    new_data.to_csv(output_path, index=False)
    print('Done')










if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf.yaml')
    parser.add_argument('--exp_dir', default = 'exp')

    args = parser.parse_args()
    new_data, new_data_feats = prepare_new_data(args)
    get_predictions(new_data, new_data_feats, args)