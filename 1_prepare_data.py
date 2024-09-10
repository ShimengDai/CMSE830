import os, sys
import yaml
import re, string
import random
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocess_text, remove_stopwords, get_wordnet_pos, lemmatizer, preprocessing_compose
from utils.feature_extraction import *
from utils.dataset import clean_data, balance_data, print_data_dims, TextDataset
from scipy.linalg import get_blas_funcs



def prepare_data(args):
    """
    Prepares the datasets for the experiment.
    """
    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    data_path = os.path.join(config['Corpus']['data_dir'], config['Corpus']['data_filename'])
    n_classes = config['Data_setup']['n_classes']
    text_col = config['Data_setup']['text_col']
    label_col = config['Data_setup']['label_col']
    labels_to_ignore = config['Data_setup']['labels_to_ignore']
    random_seed = config['Data_setup']['random_seed']
    val_ratio = config['Data_setup']['val_ratio']
    test_ratio = config['Data_setup']['test_ratio']
    feat_extraction_method = config['Feature_extraction']['method']
    feat_extraction_configs = config['Feature_extraction'][feat_extraction_method + '_configs']

    assert (val_ratio + test_ratio) < 1.0, 'Sum of val_ratio and test_ratio should be lower than 1.0.'

    # Get data and clean data
    data = pd.read_csv(data_path)
    data = clean_data(data, label_col, labels_to_ignore, n_classes)
    print(f'Found {len(data.index)} valid tweets.')

    # Balance data if needed
    if config['Data_setup']['balance_data']:
        print('Balancing the data so all classes have equal no. of observations (value: count)')
        data = balance_data(data, label_col, random_seed)

    # Conditional text preprocessing based on feature extraction method
    if feat_extraction_method in ['BERTEmbedding', 'GPT2Embedding']:
        print('Using raw text for BERT or GPT-2...')
        text_data = data[text_col].fillna('').astype(str)
    else:  # TFIDF or Word2Vec
        print('Preprocessing tweets...', end=' ')
        data = data.dropna(subset=[text_col])
        data[text_col] = data[text_col].fillna('').astype(str)
        data['clean_text'] = data[text_col].apply(lambda x: preprocessing_compose(x))
        print('Done')
        text_data = data['clean_text']

    # Save a copy of the clean data as CSV
    data_cleaned_fn = os.path.basename(data_path).split('.')[0] + '_cleaned.csv'
    data.to_csv('data/' + data_cleaned_fn, index=False)

    # Data split
    print(f'Splitting data into training, validation, and test sets and extracting features (method: {feat_extraction_method})...', end=' ')
    
    random.seed(random_seed)
    X_train, X_test, y_train, y_test = train_test_split(text_data, data[label_col],
                                                        test_size=val_ratio + test_ratio, 
                                                        random_state=random_seed, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (val_ratio + test_ratio),
                                                    random_state=random_seed,
                                                    shuffle=False)

    # Select feature extraction method and apply
    feature_extractor = getattr(sys.modules[__name__], feat_extraction_method)(feat_extraction_configs)

    X_train_vectors, X_val_vectors, X_test_vectors = feature_extractor(data=data,
                                                                       X_train=X_train,
                                                                       X_val=X_val,
                                                                       X_test=X_test)
    print('Done')

    # Save the fitted feature extractor
    feat_extractor_path = os.path.join(args.exp_dir, 'trained_models', feat_extraction_method + '_fitted.pkl')
    with open(feat_extractor_path, 'wb') as fe:
        pickle.dump(feature_extractor, fe)

    # Print out dataset dimensions
    print_data_dims(X_train_vectors, X_val_vectors, X_test_vectors, y_train, y_val, y_test)

    # Export the datasets
    datasets_path = 'exp/datasets'
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    print(f'Exporting the datasets to {datasets_path}...', end=' ')
    train_dataset = TextDataset(X_train_vectors, y_train)
    valid_dataset = TextDataset(X_val_vectors, y_val)
    test_dataset = TextDataset(X_test_vectors, y_test)

    with open(os.path.join(datasets_path, 'train_data.pkl'), 'wb') as tr:
        pickle.dump(train_dataset, tr)
    with open(os.path.join(datasets_path, 'valid_data.pkl'), 'wb') as va:
        pickle.dump(valid_dataset, va)
    with open(os.path.join(datasets_path, 'test_data.pkl'), 'wb') as te:
        pickle.dump(test_dataset, te)

    # Save y_test
    with open(os.path.join(args.exp_dir, 'y_test.pkl'), 'wb') as yt:
        pickle.dump(y_test, yt)

    print('Done')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default='conf/conf.yaml')
    parser.add_argument('--exp_dir', default='exp')

    args = parser.parse_args()
    prepare_data(args)
