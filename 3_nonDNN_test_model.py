import yaml
import os
import random
import pickle
import numpy as np
import pandas as pd
from utils.utils import ModelLogger
from utils.nonDNN_models import RandomForest
from sklearn.metrics import classification_report, confusion_matrix

def test_model(test_dataset, args):

    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    n_classes = config['Data_setup']['n_classes']
    classifier = config['NonDNN_setup']['classifier']
    seed = config['Data_setup']['random_seed']
    random.seed(seed)
    
    # Get test data and labels
    X_test = test_dataset.get_X().numpy()
    y_test = test_dataset.get_y().numpy()
    
    # Load the trained model and predict the testset
    print(f'Classifier: {classifier}')
    model_path = os.path.join(args.exp_dir, 'trained_models', classifier + '_best.pickle')
    model = pickle.load(open(model_path, 'rb'))
    y_pred, _ = model.predict(X_test)
    
    # Print and export results
    class_report = classification_report(y_test, y_pred, digits=4)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    print(class_report)
    print('Confusion matrix:')
    print(cm)
          
    results_path = os.path.join(args.exp_dir, 'results', 'test_results.txt')
    with open(results_path, 'w') as out:
        print(class_report, file=out)
        print('Confusion matrix:', file=out)
        print(cm, file=out)
    
    print(f'Results saved to {results_path}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default='conf/conf.yaml')
    parser.add_argument('--exp_dir', default='exp')
    args = parser.parse_args()
    
    te_dataset_path = os.path.join(args.exp_dir, 'datasets', 'test_data.pkl')
    with open(te_dataset_path, 'rb') as te:
        test_dataset = pickle.load(te)
    
    test_model(test_dataset, args)
