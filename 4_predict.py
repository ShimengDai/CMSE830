import os
import yaml
import nltk
import re, string
import gensim
import pickle
import torch
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from utils.preprocessing import preprocess_text, remove_stopwords, get_wordnet_pos, lemmatizer, preprocessing_compose
from utils.feature_extraction import MeanEmbeddingVectorizer
from torch.utils.data import Dataset, DataLoader
from utils.models import FullyConnected, FullyConnected2
from tqdm import tqdm






def prepare_new_data(args):
    """
    Prepare the data for which you want to use the trained model to generate predictions. This dataset is called "new data."
    Returns: 1) A copy of the new data with cleaned text; 2) features extracted from the cleaned text, used later as inputs to the classification model to generate predictions
    """
    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    new_data_path = os.path.join(config['Corpus']['data_dir'], config['Corpus']['new_data_filename'])
    text_col = config['New_data_setup']['text_col']
    include_first_n = config['New_data_setup']['include_first_n']
    
    # Get new data
    new_data = pd.read_csv(new_data_path)
    
    # Check if you want to include just the first n rows in the new data:
    if include_first_n:
        new_data = new_data.head(include_first_n)
    
    # Drop rows with NaNs in any columns
    new_data = new_data.dropna(how='all')
    print(f'No. of valid tweets: {len(new_data.index)}')
    
    # Data cleaning
    print('Preprocessing tweets in new data...')
    tqdm.pandas()
    new_data['clean_text'] = new_data[text_col].progress_apply(lambda x: preprocessing_compose(x))
    print('Done')
    
    # Load the trained MeanWordEmbedding model
    with open('exp/trained_models/w2v_model.pkl', 'rb') as m:# TFIDF_fitted.pkl  w2v_model.pkl
        modelw = pickle.load(m)
        new_data_tok = [nltk.word_tokenize(i) for i in new_data['clean_text']]
        new_data_vectors_w2v = modelw.transform(new_data_tok)
    print('Dimension of extracted features:', new_data_vectors_w2v.shape)
    
    return new_data, new_data_vectors_w2v









def get_predictions(new_data, new_data_feats, args):
    """
    Get predicted labels for new data.
    """
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    n_classes = config['Data_setup']['n_classes']
    dropout = config['Model_setup']['dropout']
    pred_col = config['New_data_setup']['pred_col']
    
    # Load the trained model
    trained_model_path = os.path.join(args.exp_dir, 'trained_models', 'best_model.pt')

    input_size = train_dataset.get_input_size()
    model = FullyConnected2(input_size=input_size, n_classes=n_classes, dropout=dropout).to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    
    # Use trained model to get predictions
    model.eval()
    with torch.no_grad():
        preds = []
        for nd_idx, nd_txt_feats in enumerate(new_data_feats):
            nd_txt_feats = torch.tensor(nd_txt_feats).to(device)
            nd_txt_feats = nd_txt_feats[None, :]
            nd_txt_feats = nd_txt_feats.to(torch.float32)
            outputs = model(nd_txt_feats)

            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.tolist())
    
    # Add the predictions to the dataframe of new data
    new_data[pred_col] = preds
    
    # Prediction summary
    print('Prediction summary:')
    print(new_data[pred_col].value_counts())
    
    # Save the dataframe with predictions as CSV
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
    parser.add_argument('--conf_dir', default='conf/conf.yaml')
    parser.add_argument('--exp_dir', default='exp')




    args = parser.parse_args()



    dataset_path = os.path.join(args.exp_dir, 'datasets')    
    tr = open(os.path.join(dataset_path, 'train_data.pkl'), 'rb') 
    va = open(os.path.join(dataset_path, 'valid_data.pkl'), 'rb')   
    train_dataset, valid_dataset = pickle.load(tr), pickle.load(va)


    new_data, new_data_feats = prepare_new_data(args)
    get_predictions(new_data, new_data_feats, args)


