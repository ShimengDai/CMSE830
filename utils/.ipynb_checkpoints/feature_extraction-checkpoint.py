import nltk
import gensim
import pickle
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class MeanWordEmbedding(object):
    """
    Converting sentence to vectors/numbers from word vectors result by Word2Vec and applying Word2Vec to the training and test data
    """
    def __init__(self, configs):
        self.configs = configs
        
    def __call__(self, **kwargs):
        data = kwargs['data']
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']
        
        # Convert preprocessed sentence to tokenized sentence
        data['clean_text_tok'] = [nltk.word_tokenize(i) for i in data['clean_text']]

        # min_count = 1 means word should be present at least once across all documents
        # If min_count = 2 means if the word is present less than 2 times across all the documents then we shouldn't consider it
        model = Word2Vec(data['clean_text_tok'], min_count = self.configs['min_count'])

        w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))  # Combination of word and its vector
        modelw = MeanEmbeddingVectorizer(w2v)

        X_train_tok = [nltk.word_tokenize(i) for i in X_train]  # for word2vec
        X_val_tok = [nltk.word_tokenize(i) for i in X_val]      # for word2vec
        X_test_tok = [nltk.word_tokenize(i) for i in X_test]    # for word2vec

        # Word2vec: transform
        X_train_vectors_w2v = modelw.transform(X_train_tok)
        X_val_vectors_w2v = modelw.transform(X_val_tok)
        X_test_vectors_w2v = modelw.transform(X_test_tok)

        # Save Word2Vec model
        with open('exp/trained_models/w2v_model.pkl', 'wb') as outp:
            pickle.dump(modelw, outp)

        return X_train_vectors_w2v, X_val_vectors_w2v, X_test_vectors_w2v

class TFIDF(object):
    """
    Extract TF-IDF features.
    """
    def __init__(self, configs):
        self.configs = configs
    
    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']
        
        # Convert x_train to vector since model can only run on numbers and not words. Do fit_transform
        self.tfidf_vectorizer = TfidfVectorizer(min_df = self.configs['min_df'],
                                           use_idf = self.configs['use_idf'])
        X_train_vectors_tfidf = self.tfidf_vectorizer.fit_transform(X_train) # tfidf runs on non-tokenized sentences unlike word2vec
        
        # Only transform x_val and x_test (not fit and transform)
        X_val_vectors_tfidf = self.tfidf_vectorizer.transform(X_val)
        X_test_vectors_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        # Convert to numpy arrays
        X_train_vectors_tfidf = X_train_vectors_tfidf.toarray().astype('float32')
        X_val_vectors_tfidf = X_val_vectors_tfidf.toarray().astype('float32')
        X_test_vectors_tfidf = X_test_vectors_tfidf.toarray().astype('float32')
        
        return X_train_vectors_tfidf, X_val_vectors_tfidf, X_test_vectors_tfidf
    
    def transform(self, X_new):
        X_new_vectors_tfidf = self.tfidf_vectorizer.transform(X_new)
        return X_new_vectors_tfidf.toarray().astype('float32')
        