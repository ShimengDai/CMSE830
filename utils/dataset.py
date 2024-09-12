import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = torch.tensor(vectors)
        self.labels = torch.tensor(list(labels))
        self.input_size = self.vectors.size(-1) 

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.vectors[idx], self.labels[idx])
    
    def get_X(self):
        return self.vectors
    
    def get_y(self):
        return self.labels
    
    def get_input_size(self):
        return self.input_size



def clean_data(data, label_col, labels_to_ignore, n_classes):
    """
    Make sure that the data have valid values (filter out NaNs, etc.).
    """
    # Drop rows with NaNs in all columns
    data = data.dropna(how = 'all')
    
    # Remove rows with invalid label values
    data = data[~data[label_col].isin(labels_to_ignore)]
    
    # Make sure that the labels are integers
    data[label_col] = data[label_col].astype(int)
    
    assert len(data[label_col].unique()) == n_classes, "No. of classes found in the data doesn't match n_classes in conf.yaml."
    
    return data

def balance_data(data, label_col, random_seed = 2023):
    """
    Create a balanced dataset in which all classes have an equal number of tweets.
    """
    data = data.groupby(label_col)
    data = data.apply(lambda x: x.sample(data.size().min(), random_state = random_seed)).reset_index(drop = True)
    for val, count in data[label_col].value_counts().items():
        print(f'  {val}: {count}')
    print(f'  {data[label_col].value_counts().sum()} tweets in total after balancing')
    return data

def print_data_dims(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Display dimensions of the training, validation, and test data
    """
    print('Dimensions of the datasets:')
    print('  X_train: {:<10}  y_train: {:<8}'.format(str(X_train.shape), str(y_train.shape)))
    print('  X_val  : {:<10}  y_val  : {:<8}'.format(str(X_val.shape), str(y_val.shape)))
    print('  X_test : {:<10}  y_test : {:<8}'.format(str(X_test.shape), str(y_test.shape)))