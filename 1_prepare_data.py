import os, sys
import yaml
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from utils.preprocessing import preprocess_text, remove_stopwords, get_wordnet_pos, lemmatizer, preprocessing_compose
from utils.feature_extraction import *
from utils.dataset import clean_data, balance_data, print_data_dims, TextDataset

def prepare_data(args):
    """
    Prepares the datasets for cross-validation with a separate hold-out test set.
    """
    # Load configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    data_path = os.path.join(config['Corpus']['data_dir'], config['Corpus']['data_filename'])
    n_classes = config['Data_setup']['n_classes']
    text_col = config['Data_setup']['text_col']
    label_col = config['Data_setup']['label_col']
    labels_to_ignore = config['Data_setup']['labels_to_ignore']
    random_seed = config['cross_validation_setup']['random_seed']
    k_folds = config['cross_validation_setup']['k_folds']  # k-folds for cross-validation
    test_ratio = config['Data_setup']['test_ratio']
    feat_extraction_method = config['Feature_extraction']['method']
    feat_extraction_configs = config['Feature_extraction'][feat_extraction_method + '_configs']

    # Get and clean data
    data = pd.read_csv(data_path)
    data = clean_data(data, label_col, labels_to_ignore, n_classes)
    print(f'Found {len(data.index)} valid tweets.')

    # Balance data if required
    if config['Data_setup']['balance_data']:
        print('Balancing the data so all classes have equal no. of observations (value: count)')
        data = balance_data(data, label_col, random_seed)

    # Conditional preprocessing based on the feature extraction method
    if feat_extraction_method in ['BERTEmbedding', 'GPT2Embedding']:
        print('Using raw text for BERT or GPT-2...')
        text_data = data[text_col].fillna('').astype(str)  # Raw text for transformers
    else:  # TFIDF or Word2Vec
        print('Preprocessing tweets...', end=' ')
        data = data.dropna(subset=[text_col])
        data[text_col] = data[text_col].fillna('').astype(str)
        data['clean_text'] = data[text_col].apply(lambda x: preprocessing_compose(x))  # Preprocessed text for traditional embeddings
        print('Done')
        text_data = data['clean_text']

    # Split data into training-validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(text_data, data[label_col].values, test_size=test_ratio, random_state=random_seed)
    print(f'Train/Validation size: {len(X_train_val)}, Test size: {len(X_test)}')

    # Initialize k-fold cross-validation for the training-validation set
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    fold_idx = 0
    for train_index, val_index in kf.split(X_train_val):
        print(f'Preparing fold {fold_idx + 1}/{k_folds}...')
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        # Feature extraction based on the specified method
        feature_extractor = getattr(sys.modules[__name__], feat_extraction_method)(feat_extraction_configs)

        X_train_vectors, X_val_vectors = feature_extractor(data=data,
                                                           X_train=X_train,
                                                           X_val=X_val,
                                                           X_test=None)  # No test set during cross-validation
        print(f'Fold {fold_idx + 1} - Feature extraction complete.')

        # Save the fitted feature extractor for this fold
        feat_extractor_path = os.path.join(args.exp_dir, 'trained_models', f'{feat_extraction_method}_fold{fold_idx + 1}_fitted.pkl')
        with open(feat_extractor_path, 'wb') as fe:
            pickle.dump(feature_extractor, fe)

        # Export the datasets for this fold
        datasets_path = os.path.join(args.exp_dir, 'datasets', f'fold_{fold_idx + 1}')
        if not os.path.exists(datasets_path):
            os.makedirs(datasets_path)

        print(f'Exporting fold {fold_idx + 1} datasets to {datasets_path}...')
        train_dataset = TextDataset(X_train_vectors, y_train)
        valid_dataset = TextDataset(X_val_vectors, y_val)

        with open(os.path.join(datasets_path, 'train_data.pkl'), 'wb') as tr:
            pickle.dump(train_dataset, tr)
        with open(os.path.join(datasets_path, 'valid_data.pkl'), 'wb') as va:
            pickle.dump(valid_dataset, va)

        fold_idx += 1

    # Feature extraction for the test set
    print("Feature extraction for the test set...")
    X_test_vectors = feature_extractor(data=data, X_train=None, X_val=None, X_test=X_test)[2]

    # Save the test set
    test_dataset = TextDataset(X_test_vectors, y_test)
    test_data_path = os.path.join(args.exp_dir, 'datasets', 'test_data.pkl')
    with open(test_data_path, 'wb') as te:
        pickle.dump(test_dataset, te)

    # Save y_test
    with open(os.path.join(args.exp_dir, 'y_test.pkl'), 'wb') as yt:
        pickle.dump(y_test, yt)

    print('Cross-validation data preparation completed.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default='conf/conf.yaml')
    parser.add_argument('--exp_dir', default='exp')

    args = parser.parse_args()
    prepare_data(args)
