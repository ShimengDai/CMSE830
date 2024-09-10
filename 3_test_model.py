import yaml
import os
import torch
import pickle
from utils.models import FullyConnected2
from utils.utils import ModelScorer

def test_model(test_dataset, args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    n_classes = config['Data_setup']['n_classes']
    input_size = test_dataset.get_input_size()
    dropout = config['Model_setup']['dropout']
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    trained_model_path = os.path.join(args.exp_dir, 'trained_models', 'best_model.pt')

    model = FullyConnected2(input_size=input_size, n_classes=n_classes, dropout=dropout).to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    
    model.eval()
    with torch.no_grad():
        preds = []
        targets = []
        for test_txt_feats, test_labels in test_loader:
            test_txt_feats = test_txt_feats.to(device)
            test_labels = test_labels.to(device)
            
            outputs = model(test_txt_feats)

            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.tolist())
            targets.extend(test_labels.tolist())
                
    scorer = ModelScorer(targets, preds)
    scorer.compute_metrics('accuracy', 'precision', 'recall', 'F1')
    
    results_path = os.path.join(args.exp_dir, 'results')
    scorer.save_results(results_path)
    
    test_acc = scorer.accuracy()
    print(f'Test accuracy: {100 * test_acc:.3f}%')
    print(f'Check {results_path} for more detailed results.')
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default='conf/conf.yaml')
    parser.add_argument('--exp_dir', default='exp')
    args = parser.parse_args()
    
    dataset_path = os.path.join(args.exp_dir, 'datasets')    
    with open(os.path.join(dataset_path, 'test_data.pkl'), 'rb') as te:
        test_dataset = pickle.load(te)
    
    test_model(test_dataset, args)
