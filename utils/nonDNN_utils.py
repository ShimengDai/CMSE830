import itertools

def generate_hyperparameter_grid(configs):
    all_param_combs = [configs[k] if isinstance(configs[k], list) else [configs[k]] for k in configs.keys()]
    all_param_combs = list(itertools.product(*all_param_combs))
    param_dict_list = [dict(zip(configs.keys(), params)) for params in all_param_combs]
    keys = [str(k) for k in range(1, len(param_dict_list)+1)]
    return dict(zip(keys, param_dict_list))

def save_best_model_info(output_path, tracker):
    with open(output_path, 'w') as out:
        print(f'Classifier: {tracker.classifier}', file = out)
        print(f'Train_acc: {tracker.best_train_acc}', file = out)
        print(f'Valid_acc: {tracker.best_val_acc}', file = out)
        print(f'Hyperparameters: {tracker.best_params}', file = out)

class ModelTracker(object):
    def __init__(self, classifier = None):
        self.classifier = classifier
        self.best_model = None
        self.best_train_acc = -1.0
        self.best_val_acc = -1.0
        self.best_params = None
        self.save_model = False
        
    def update(self, **kwargs):
        model = kwargs['model']
        y_train, y_pred_train = kwargs['y_train'], kwargs['y_pred_train']
        y_val, y_pred_val = kwargs['y_val'], kwargs['y_pred_val']
        params = kwargs['params']
        
        train_acc = sum(y_pred_train == y_train) / len(y_train)
        val_acc = sum(y_pred_val == y_val) / len(y_val)
        
        print('Train acc: {:.4f}, Valid acc: {:.4f}. Hyperparams: {}'.format(train_acc, val_acc, params))
        
        if val_acc > self.best_val_acc:
            self.best_model = model
            self.best_val_acc = val_acc
            self.best_train_acc = train_acc
            self.best_params = params
            self.save_model = True
        else:
            self.save_model = False