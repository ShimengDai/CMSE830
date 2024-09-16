import nltk
import gensim
import pickle
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# for GPT2 and BERT
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
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
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, **kwargs):
        data = kwargs['data']
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        data['clean_text_tok'] = [nltk.word_tokenize(i) for i in data['clean_text']]

        model = Word2Vec(data['clean_text_tok'], min_count=self.configs['min_count'])
        w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))
        modelw = MeanEmbeddingVectorizer(w2v)

        X_train_tok = [nltk.word_tokenize(i) for i in X_train]
        X_val_tok = [nltk.word_tokenize(i) for i in X_val]

        if X_test is not None:
            X_test_tok = [nltk.word_tokenize(i) for i in X_test]
            X_test_vectors_w2v = modelw.transform(X_test_tok)
        else:
            X_test_vectors_w2v = None

        X_train_vectors_w2v = modelw.transform(X_train_tok)
        X_val_vectors_w2v = modelw.transform(X_val_tok)

        return X_train_vectors_w2v, X_val_vectors_w2v, X_test_vectors_w2v


class word2vec_3D(object):
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, **kwargs):
        data = kwargs['data']
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        data['clean_text_tok'] = [nltk.word_tokenize(i) for i in data['clean_text']]
        model = Word2Vec(data['clean_text_tok'], min_count=self.configs['min_count'])
        w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))

        X_train_tok = [nltk.word_tokenize(i) for i in X_train]
        X_val_tok = [nltk.word_tokenize(i) for i in X_val]

        if X_test is not None:
            X_test_tok = [nltk.word_tokenize(i) for i in X_test]
            X_test_vectors_w2v = self.transform(X_test_tok, w2v)
        else:
            X_test_vectors_w2v = None

        X_train_vectors_w2v = self.transform(X_train_tok, w2v)
        X_val_vectors_w2v = self.transform(X_val_tok, w2v)

        return X_train_vectors_w2v, X_val_vectors_w2v, X_test_vectors_w2v

    def transform(self, tokenized_sentences, w2v):
        max_length = 280
        vector_size = self.configs['vector_size']

        vectorized_sentences = []

        for sentence in tokenized_sentences:
            sentence_vectors = []

            for word in sentence[:max_length]:
                if word in w2v:
                    sentence_vectors.append(w2v[word])
                else:
                    sentence_vectors.append(np.zeros(vector_size))

            while len(sentence_vectors) < max_length:
                sentence_vectors.append(np.zeros(vector_size))

            vectorized_sentences.append(sentence_vectors)

        return np.array(vectorized_sentences)


class TFIDF(object):
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        self.tfidf_vectorizer = TfidfVectorizer(min_df=self.configs['min_df'], use_idf=self.configs['use_idf'])
        X_train_vectors_tfidf = self.tfidf_vectorizer.fit_transform(X_train)

        X_val_vectors_tfidf = self.tfidf_vectorizer.transform(X_val)

        if X_test is not None:
            X_test_vectors_tfidf = self.tfidf_vectorizer.transform(X_test)
            X_test_vectors_tfidf = X_test_vectors_tfidf.toarray().astype('float32')
        else:
            X_test_vectors_tfidf = None

        X_train_vectors_tfidf = X_train_vectors_tfidf.toarray().astype('float32')
        X_val_vectors_tfidf = X_val_vectors_tfidf.toarray().astype('float32')

        return X_train_vectors_tfidf, X_val_vectors_tfidf, X_test_vectors_tfidf


class P_BERTEmbedding(object):
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.dim = 768

    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        X_train_vectors_bert = self.transform(X_train)
        X_val_vectors_bert = self.transform(X_val)

        if X_test is not None:
            X_test_vectors_bert = self.transform(X_test)
        else:
            X_test_vectors_bert = None

        return X_train_vectors_bert, X_val_vectors_bert, X_test_vectors_bert

    def transform(self, X):
        embeddings = []
        for sentence in X:
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten())
        return np.array(embeddings)


class P_GPT2Embedding(object):
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2Model.from_pretrained('gpt2').to(self.device)
        self.dim = 768

    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        X_train_vectors_gpt2 = self.transform(X_train)
        X_val_vectors_gpt2 = self.transform(X_val)

        if X_test is not None:
            X_test_vectors_gpt2 = self.transform(X_test)
        else:
            X_test_vectors_gpt2 = None

        return X_train_vectors_gpt2, X_val_vectors_gpt2, X_test_vectors_gpt2

    def transform(self, X):
        embeddings = []
        for sentence in X:
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten())
        return np.array(embeddings)


class BERTEmbedding(object):
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.dim = 768

    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        X_train_vectors_bert = self.transform(X_train)
        X_val_vectors_bert = self.transform(X_val)

        if X_test is not None:
            X_test_vectors_bert = self.transform(X_test)
        else:
            X_test_vectors_bert = None

        return X_train_vectors_bert, X_val_vectors_bert, X_test_vectors_bert

    def transform(self, X_data):
        embeddings = []
        max_length = 280

        for text in X_data:
            inputs = self.tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            last_hidden_state = torch.squeeze(outputs.last_hidden_state)
            embeddings.append(last_hidden_state.cpu().numpy())

        return np.array(embeddings)


class GPT2Embedding(object):
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained('gpt2').to(self.device)
        self.dim = 768

    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        X_train_vectors_gpt2 = self.transform(X_train)
        X_val_vectors_gpt2 = self.transform(X_val)

        if X_test is not None:
            X_test_vectors_gpt2 = self.transform(X_test)
        else:
            X_test_vectors_gpt2 = None

        return X_train_vectors_gpt2, X_val_vectors_gpt2, X_test_vectors_gpt2

    def transform(self, X_data):
        embeddings = []
        max_length = 280

        for text in X_data:
            inputs = self.tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            last_hidden_state = torch.squeeze(outputs.last_hidden_state)
            embeddings.append(last_hidden_state.cpu().numpy())

        return np.array(embeddings)
