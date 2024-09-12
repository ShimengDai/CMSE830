import yaml
import os
import random
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils import ModelLogger
from utils.models import FullyConnected, FullyConnected2, LSTMModel  # Import all models you might use

def train_model(train_dataset, valid_dataset, args):
    # Use GPU if available 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Train model using {device}')

    # Get configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    n_classes = config['Data_setup']['n_classes']
    
    # Training setup
    input_size = train_dataset.get_input_size()  # Automatically get input size based on dataset
    epochs = config['Training_setup']['epochs']
    batch_size = config['Training_setup']['batch_size']
    learning_rate = config['Training_setup']['learning_rate']
    
    # Model setup
    dropout = config['Model_setup']['dropout']
    model_name = config['Model_setup']['model_name']  # Get the model name from config
    
    # Dynamically select the model based on the config file
    model_class = globals()[model_name]  # Get the class reference from the model name
    model = model_class(input_size=input_size, n_classes=n_classes, dropout=dropout).to(device)
    
    # Get the datasets and model    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
    
    # Loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(len(train_loader)) * 25, gamma=0.7)
    
    # Initialize ModelLogger
    logger = ModelLogger(args.exp_dir)
    
    # Start training
    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        train_loss = []
        model.train()
        for batch_idx, (txt_feats, labels) in enumerate(train_loader):
            labels = labels.to(device)
            txt_feats = txt_feats.to(device)
            #print(txt_feats.size())
            outputs = model(txt_feats)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss.append(loss.detach().cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = sum(train_loss) / len(train_loss)
        train_acc = 100 * train_correct / train_total
        
        # Validate on the validation set
        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            val_loss = []
            for val_batch_idx, (val_txt_feats, val_labels) in enumerate(valid_loader):
                val_labels = val_labels.to(device)
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
        
        # Update training history
        logger.update(epoch, avg_train_loss, avg_val_loss, 
                      train_acc=train_acc, val_acc=val_acc, 
                      learning_rate=lr)
        
        # Save the model if the best validation loss is encountered
        if logger.save_best_model:
            logger.save_model(model)
    
        print(f"Epoch: {epoch + 1} \t train_acc: {train_acc:.2f}% \t val_acc: {val_acc:.2f}% \t train_loss: {avg_train_loss:.3f} \t val_loss: {avg_val_loss:.3f} \t LR: {lr:.2E}")
    
    logger.save_history()
    logger.print_best_model_info()
    print('Training completed')

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
    
    train_model(train_dataset, valid_dataset, args)

