import yaml
import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils import ModelLogger
from utils.models import FullyConnected, FullyConnected2, LSTMModel  # Import all models you might use

def train_model(args):
    """
    Trains the model using cross-validation.
    """
    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Train model using {device}')

    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    n_classes = config['Data_setup']['n_classes']

    # Training setup
    epochs = config['Training_setup']['epochs']
    batch_size = config['Training_setup']['batch_size']
    learning_rate = config['Training_setup']['learning_rate']

    # Model setup
    dropout = config['Model_setup']['dropout']
    model_name = config['Model_setup']['model_name']  # Get the model name from config

    # Dynamically select the model based on the config file
    model_class = globals()[model_name]  # Get the class reference from the model name

    # Initialize logger
    logger = ModelLogger(args.exp_dir)

    # Load cross-validation folds
    num_folds = config['cross_validation_setup']['k_folds']
    
    # Iterate over each fold for cross-validation
    for fold_idx in range(1, num_folds + 1):
        print(f'Training fold {fold_idx}/{num_folds}...')
        
        # Load fold-specific training and validation datasets
        dataset_path = os.path.join(args.exp_dir, 'datasets', f'fold_{fold_idx}')
        with open(os.path.join(dataset_path, 'train_data.pkl'), 'rb') as tr:
            train_dataset = pickle.load(tr)
        with open(os.path.join(dataset_path, 'valid_data.pkl'), 'rb') as va:
            valid_dataset = pickle.load(va)
        
        # Get input size from dataset
        input_size = train_dataset.get_input_size()

        # Initialize the model for this fold
        model = model_class(input_size=input_size, n_classes=n_classes, dropout=dropout).to(device)

        # DataLoader for training and validation
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)

        # Loss function, optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(len(train_loader)) * 25, gamma=0.7)

        # Track best validation loss for this fold
        best_val_loss = float('inf')

        # Training loop for the current fold
        for epoch in range(epochs):
            model.train()
            train_loss = []
            train_correct, train_total = 0, 0
            
            for batch_idx, (txt_feats, labels) in enumerate(train_loader):
                labels = labels.to(device)
                txt_feats = txt_feats.to(device)

                optimizer.zero_grad()
                outputs = model(txt_feats)

                # Calculate the loss
                loss = criterion(outputs, labels.long())  # Convert labels to LongTensor
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss.append(loss.detach().cpu().numpy())
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            avg_train_loss = sum(train_loss) / len(train_loss)
            train_acc = 100 * train_correct / train_total

            # Validation loop
            model.eval()
            val_loss = []
            val_correct, val_total = 0, 0

            with torch.no_grad():
                for val_batch_idx, (val_txt_feats, val_labels) in enumerate(valid_loader):
                    val_labels = val_labels.to(device).long()  # Ensure labels are LongTensor
                    val_txt_feats = val_txt_feats.to(device)

                    outputs = model(val_txt_feats)
                    loss = criterion(outputs, val_labels)

                    val_loss.append(loss.detach().cpu().numpy())

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += val_labels.size(0)
                    val_correct += (predicted == val_labels).sum().item()

            avg_val_loss = sum(val_loss) / len(val_loss)
            val_acc = 100 * val_correct / val_total

            lr = scheduler.get_last_lr()[0]

            # Log performance for this epoch
            logger.update(epoch, avg_train_loss, avg_val_loss,
                          train_acc=train_acc, val_acc=val_acc,
                          learning_rate=lr)

            # Save the model if the best validation loss is encountered
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(args.exp_dir, 'trained_models', f'best_model_fold_{fold_idx}.pt'))

            print(f"Fold {fold_idx} - Epoch {epoch + 1}: train_acc: {train_acc:.2f}%, val_acc: {val_acc:.2f}%, train_loss: {avg_train_loss:.3f}, val_loss: {avg_val_loss:.3f}, LR: {lr:.2E}")

        # Save the final model for this fold
        torch.save(model.state_dict(), os.path.join(args.exp_dir, 'trained_models', f'final_model_fold_{fold_idx}.pt'))
        print(f'Fold {fold_idx} completed.\n')

    logger.save_history()
    logger.print_best_model_info()
    print('Cross-validation training completed.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default='conf/conf.yaml')
    parser.add_argument('--exp_dir', default='exp')

    args = parser.parse_args()
    train_model(args)
