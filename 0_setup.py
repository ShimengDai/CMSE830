import os
import nltk
import numpy as np
from shutil import copyfile

def setup(args):
    """
    Set up folders for the experiment and download required nltk resources.
    """
    config_path = args.conf_dir
    exp_path = args.exp_dir
    resources = args.resources
    require_zip = args.require_zip

    # Create necessary directories
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        print(f"  Folder {exp_path} doesn't exist. Created it.")
    
    results_path = os.path.join(exp_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"  Folder {results_path} doesn't exist. Created it.")

    trained_models_path = os.path.join(exp_path, 'trained_models')
    if not os.path.exists(trained_models_path):
        os.makedirs(trained_models_path)
        print(f"  Folder {trained_models_path} doesn't exist. Created it.")
    
    # Copy configuration and model files
    conf_dst = os.path.join(exp_path, os.path.basename(config_path))        
    copyfile(config_path, conf_dst)
    print(f'  Copied {config_path} to {conf_dst}.')
    
    model_dst = os.path.join(exp_path, 'models.py')
    copyfile('utils/models.py', model_dst)
    print(f'  Copied utils/models.py to {model_dst}.\n')
    
    # Download nltk resources if they're not installed yet
    print('  Searching for required nltk resources...')
    for r in resources:
        try:
            nltk.data.find(r)
            print(f'    {r}... OK')
        except LookupError:
            print(f'    {r} not found. Downloading...')
            nltk.download(os.path.basename(r))

    print('  All required nltk resources are available.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default='conf/conf.yaml')
    parser.add_argument('--exp_dir', default='exp')
    parser.add_argument('--resources', default=[
        'corpora/stopwords', 'corpora/wordnet', 'corpora/omw-1.4',
        'tokenizers/punkt', 'taggers/averaged_perceptron_tagger'])
    parser.add_argument('--require_zip', default=[
        'corpora/wordnet', 'corpora/omw-1.4'])

    args = parser.parse_args()
    print('Setting up for the experiment...')
    setup(args)
