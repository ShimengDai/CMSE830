import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ModelScorer(object):
    def __init__(self, targets, preds):
        self.targets = targets
        self.preds = preds
        self.metrics = {}
    
    def accuracy(self):
        print(self.targets)
        print(self.preds)
        return accuracy_score(self.targets, self.preds)
    
    def precision(self):
        return precision_score(self.targets, self.preds)
    
    def recall(self):
        return recall_score(self.targets, self.preds)
    
    def F1(self):
        return f1_score(self.targets, self.preds)
    
    def confusion_matrix(self):
        return confusion_matrix(self.targets, self.preds)
    
    def compute_metrics(self, *args):
        for m in args:
            func = getattr(self, m)
            result = func()
            self.metrics[m] = result
    
    def save_results(self, outpath):
        
        # Export the confusion matrix
        cm = self.confusion_matrix()
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        plt.savefig(os.path.join(outpath, 'test_confusion_matrix.png'))
        plt.close()
        
        # Export the computed metrics
        if self.metrics:
            with open(os.path.join(outpath, 'test_metrics.txt'), 'w') as out:
                for m in self.metrics:
                    print(f'{m}: {self.metrics[m]}', file = out)
        
class ModelLogger(object):
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.model_outpath = os.path.join(exp_dir, 'trained_models')
        self.history_outpath = os.path.join(exp_dir, 'results')
        
        if not os.path.exists(self.model_outpath):
            os.makedirs(self.model_outpath)
        if not os.path.exists(self.history_outpath):
            os.makedirs(self.history_outpath)
            
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': []}
        self.best_epoch = None
        self.best_val_loss = None
        self.save_best_model = False

    def update(self, epoch, train_loss, val_loss, **kwargs):
        """
        Updates the loggger with information from each epoch. Requires at least the epoch, training loss, and validation loss.
        """
        if not epoch: # For the very first epoch
            self.best_epoch = epoch
            self.best_val_loss = val_loss
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for k in kwargs:
                self.history[k] = [kwargs[k]]
            self.save_best_model = True
        else:
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for k in kwargs:
                self.history[k].append(kwargs[k])
            if val_loss < self.best_val_loss:
                self.best_epoch = epoch + 1
                self.best_val_loss = val_loss
                self.save_best_model = True
            else:
                self.save_best_model = False

    def save_model(self, model, model_fn = 'best_model.pt'):
        """
        Saves the model
        """
        torch.save(model.state_dict(), os.path.join(self.model_outpath, model_fn))
        self.save_best_model = False
        
    def save_history(self):
        """
        Exports training history
        """
        with open(os.path.join(self.history_outpath, 'history.txt'), 'w') as h:
            col_names = list(self.history.keys())
            header = '\t'.join(col_names)
            print(header, file = h)
            
            iterables = []
            for c in col_names:
                iterables.append(self.history[c])
            
            for line in zip(*iterables):
                output_line = [str(x) for x in list(line)]
                print('\t'.join(output_line), file = h)
    
    def print_best_model_info(self):
        print('Best model encountered in epoch {} with val_loss = {:.4f}'.format(self.best_epoch, self.best_val_loss))