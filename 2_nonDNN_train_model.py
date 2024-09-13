import yaml
import os
import sys
import random
import pickle
from utils.nonDNN_models import RandomForest, SVM, LR
from utils.nonDNN_utils import generate_hyperparameter_grid, save_best_model_info, ModelTracker
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score


    
def train_model(train_dataset, valid_dataset, args):

    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader = yaml.FullLoader)
    n_classes = config['Data_setup']['n_classes']
    classifier = config['NonDNN_setup']['classifier']
    classifier_configs = config['NonDNN_setup'][classifier + '_configs']
    
    # Get training and validation data and labels
    X_train = train_dataset.get_X().numpy()
    y_train = train_dataset.get_y().numpy()
    X_val = valid_dataset.get_X().numpy()
    y_val = valid_dataset.get_y().numpy()
    
    # Training setup
    # input_size = config['Training_setup']['input_size']
    
    seed = config['Data_setup']['random_seed']
    random.seed(seed)
    
    # Train model and test it on the validation set
    print(f'Classifier: {classifier}')
    
    grid = generate_hyperparameter_grid(classifier_configs)
    print(f'{len(grid)} hyperparameter setting(s) tested')
    
    tracker = ModelTracker(classifier = classifier)
    best_model_fn = os.path.join(args.exp_dir, 'trained_models', classifier + '_best.pickle')
    
    for i in grid.keys():
        params = grid[i].copy()
        params['seed'] = seed
        model = getattr(sys.modules[__name__], classifier)(params)
        y_pred_train, y_pred_val, _, _ = model(X_train = X_train, y_train = y_train,
                                               X_val = X_val, y_val = y_val)
        
        tracker.update(model = model, y_train = y_train, y_pred_train = y_pred_train,
                      y_val = y_val, y_pred_val = y_pred_val, params = grid[i].copy())
        
        if tracker.save_model:
            pickle.dump(tracker.best_model, open(best_model_fn, 'wb'))
            
    print()
    print(f'Best model (highest validation accuracy) found with the following hyperparameter settings:')
    print(f'  Hyperparameters: {tracker.best_params}')
    print(f'  (train_acc: {tracker.best_train_acc}, val_acc: {tracker.best_val_acc})')
    save_best_model_info(os.path.join(args.exp_dir, 'results/best_model_info.txt'), tracker)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf.yaml')
    parser.add_argument('--exp_dir', default = 'exp')
    args = parser.parse_args()
    
    # dataset_path = os.path.join(args.exp_dir, 'datasets')
    tr_dataset_path = os.path.join(args.exp_dir, 'datasets', 'train_data.pkl')
    va_dataset_path = os.path.join(args.exp_dir, 'datasets', 'valid_data.pkl')
    with open(tr_dataset_path, 'rb') as tr, open(va_dataset_path, 'rb') as va:
        train_dataset, valid_dataset = pickle.load(tr), pickle.load(va)
    
    train_model(train_dataset, valid_dataset, args)