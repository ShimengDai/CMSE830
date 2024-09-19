import yaml
import os
import torch
import pickle
from utils.models import FullyConnected2
from utils.utils import ModelScorer

import yaml
import os
import torch
import pickle
from utils.models import FullyConnected, FullyConnected2, LSTMModel  # Import all models
from utils.utils import ModelScorer

def test_model(test_dataset, args):
    """
    Test the final model on the hold-out test set using the best models from cross-validation folds.
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    n_classes = config['Data_setup']['n_classes']
    input_size = test_dataset.get_input_size()
    dropout = config['Model_setup']['dropout']
    model_name = config['Model_setup']['model_name']  # Get the model name from config

    # Dynamically select the model based on the config file
    model_class = globals()[model_name]  # Get the class reference from the model name

    # DataLoader for test set
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Initialize lists to store predictions and targets
    fold_preds = []
    fold_targets = []
    
    num_folds = config['cross_validation_setup']['k_folds']

    # Loop over all folds and evaluate the best model for each fold
    for fold_idx in range(1, num_folds + 1):
        print(f"Evaluating best model from fold {fold_idx}/{num_folds}...")
        
        # Load the best model for this fold
        trained_model_path = os.path.join(args.exp_dir, 'trained_models', f'best_model_fold_{fold_idx}.pt')
        model = model_class(input_size=input_size, n_classes=n_classes, dropout=dropout).to(device)  # Dynamically instantiate model
        model.load_state_dict(torch.load(trained_model_path, map_location=device))

        # Set model to evaluation mode
        model.eval()
        
        fold_preds_fold = []
        fold_targets_fold = []

        # Evaluate the model on the test set
        with torch.no_grad():
            for test_txt_feats, test_labels in test_loader:
                test_txt_feats = test_txt_feats.to(device)
                test_labels = test_labels.to(device)

                outputs = model(test_txt_feats)
                _, predicted = torch.max(outputs.data, 1)

                fold_preds_fold.extend(predicted.tolist())
                fold_targets_fold.extend(test_labels.tolist())

        # Store predictions and targets for this fold
        fold_preds.append(fold_preds_fold)
        fold_targets.append(fold_targets_fold)

    # Compute the average performance metrics across all folds
    avg_preds = [sum(x) / num_folds for x in zip(*fold_preds)]
    avg_targets = fold_targets[0]  # Targets are the same for all folds

    # Initialize scorer and compute metrics
    scorer = ModelScorer(avg_targets, avg_preds)
    scorer.compute_metrics('accuracy', 'precision', 'recall', 'F1')

    # Save results
    results_path = os.path.join(args.exp_dir, 'results')
    scorer.save_results(results_path)

    # Print the final test accuracy
    test_acc = scorer.accuracy()
    print(f'Test accuracy: {100 * test_acc:.3f}%')
    print(f'Check {results_path} for more detailed results.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default='conf/conf.yaml')
    parser.add_argument('--exp_dir', default='exp')
    args = parser.parse_args()

    # Load the hold-out test dataset
    dataset_path = os.path.join(args.exp_dir, 'datasets')
    with open(os.path.join(dataset_path, 'test_data.pkl'), 'rb') as te:
        test_dataset = pickle.load(te)

    # Perform testing using the best models from cross-validation
    test_model(test_dataset, args)
