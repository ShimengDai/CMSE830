import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # For SVM
from sklearn.linear_model import LogisticRegression  # For Logistic Regression

class RandomForest(object):
    def __init__(self, configs):
        self.configs = configs
    
    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        y_train = kwargs['y_train']
        X_val = kwargs['X_val']
        y_val = kwargs['y_val']
        self.forest = RandomForestClassifier(n_estimators = self.configs['n_estimators'],
                                             max_depth = self.configs['max_depth'],
                                             min_samples_split = self.configs['min_samples_split'],
                                             min_samples_leaf = self.configs['min_samples_leaf'],
                                             random_state = self.configs['seed'])
        self.forest.fit(X_train, y_train)
        y_pred_train = self.forest.predict(X_train)
        y_pred_val = self.forest.predict(X_val)
        
        y_prob_train = self.forest.predict_proba(X_train)[:,1]
        y_prob_val = self.forest.predict_proba(X_val)[:,1]
        
        return y_pred_train, y_pred_val, y_prob_train, y_prob_val
    
    def predict(self, X_test):
        y_pred = self.forest.predict(X_test)
        y_prob = self.forest.predict_proba(X_test)[:,1]
        
        return y_pred, y_prob
    










# SVM Model
class SVM(object):
    def __init__(self, configs):
        self.configs = configs
    
    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        y_train = kwargs['y_train']
        X_val = kwargs['X_val']
        y_val = kwargs['y_val']
        
        # Initialize SVM with kernel and regularization parameter C
        self.svm = SVC(kernel=self.configs['kernel'],
                       C=self.configs['C'],
                       probability=True,  # Enable probability estimates for SVM
                       random_state=self.configs['seed'])
        
        # Train the SVM model
        self.svm.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.svm.predict(X_train)
        y_pred_val = self.svm.predict(X_val)
        
        # Get probability scores
        y_prob_train = self.svm.predict_proba(X_train)[:, 1]
        y_prob_val = self.svm.predict_proba(X_val)[:, 1]
        
        return y_pred_train, y_pred_val, y_prob_train, y_prob_val
    
    def predict(self, X_test):
        y_pred = self.svm.predict(X_test)
        y_prob = self.svm.predict_proba(X_test)[:, 1]
        
        return y_pred, y_prob











# Logistic Regression Model
class LogisticRegressionModel(object):
    def __init__(self, configs):
        self.configs = configs
    
    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        y_train = kwargs['y_train']
        X_val = kwargs['X_val']
        y_val = kwargs['y_val']
        
        # Initialize Logistic Regression with regularization parameter C
        self.logistic = LogisticRegression(C=self.configs['C'],
                                           max_iter=self.configs['max_iter'],
                                           random_state=self.configs['seed'])
        
        # Train the Logistic Regression model
        self.logistic.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.logistic.predict(X_train)
        y_pred_val = self.logistic.predict(X_val)
        
        # Get probability scores
        y_prob_train = self.logistic.predict_proba(X_train)[:, 1]
        y_prob_val = self.logistic.predict_proba(X_val)[:, 1]
        
        return y_pred_train, y_pred_val, y_prob_train, y_prob_val
    
    def predict(self, X_test):
        y_pred = self.logistic.predict(X_test)
        y_prob = self.logistic.predict_proba(X_test)[:, 1]
        
        return y_pred, y_prob
