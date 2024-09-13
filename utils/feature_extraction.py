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
    




class word2vec_3D(object):
    """
    Converting sentence to a sequence of word vectors using Word2Vec (without averaging)
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
        
        # Train Word2Vec model
        model = Word2Vec(data['clean_text_tok'], min_count=self.configs['min_count'])

        w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))  # Combination of word and its vector
        
        # Tokenize the input data for training, validation, and test sets
        X_train_tok = [nltk.word_tokenize(i) for i in X_train]
        X_val_tok = [nltk.word_tokenize(i) for i in X_val]
        X_test_tok = [nltk.word_tokenize(i) for i in X_test]

        # Transform each tokenized sentence to its corresponding word vectors
        X_train_vectors_w2v = self.transform(X_train_tok, w2v)
        X_val_vectors_w2v = self.transform(X_val_tok, w2v)
        X_test_vectors_w2v = self.transform(X_test_tok, w2v)
        
        # Save Word2Vec model
        with open('exp/trained_models/w2v_model_non_avg.pkl', 'wb') as outp:
            pickle.dump(model, outp)

        return X_train_vectors_w2v, X_val_vectors_w2v, X_test_vectors_w2v
    
    def transform(self, tokenized_sentences, w2v):
        """
        Convert tokenized sentences into a sequence of word vectors (without averaging).
        Each sentence is converted to a sequence of word vectors, and padding is applied to make all sequences the same length.
        """
        max_length = max([len(sent) for sent in tokenized_sentences])  # Get the max sentence length

        vectorized_sentences = []
        
        for sentence in tokenized_sentences:
            sentence_vectors = []
            
            for word in sentence:
                if word in w2v:
                    sentence_vectors.append(w2v[word])  # Get the vector for each word
                else:
                    sentence_vectors.append(np.zeros(self.configs['vector_size']))  # Use a zero vector for unknown words

            # Padding: Add zero vectors if the sentence is shorter than the max length
            while len(sentence_vectors) < max_length:
                sentence_vectors.append(np.zeros(self.configs['vector_size']))

            vectorized_sentences.append(sentence_vectors)
        
        return np.array(vectorized_sentences)  # Return as a 3D numpy array (num_sentences, max_length, vector_size)









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
        



class P_BERTEmbedding(object):
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.dim = 768  # BERT base model's output dimension

    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        X_train_vectors_bert = self.transform(X_train)
        X_val_vectors_bert = self.transform(X_val)
        X_test_vectors_bert = self.transform(X_test)

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
        self.dim = 768  # GPT-2 base model's output dimension

    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        X_train_vectors_gpt2 = self.transform(X_train)
        X_val_vectors_gpt2 = self.transform(X_val)
        X_test_vectors_gpt2 = self.transform(X_test)

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
        self.dim = 768  # BERT base model's output dimension

    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        X_train_vectors_bert = self.transform(X_train)
        X_val_vectors_bert = self.transform(X_val)
        X_test_vectors_bert = self.transform(X_test)

        return X_train_vectors_bert, X_val_vectors_bert, X_test_vectors_bert

    def transform(self, X_data):
        embeddings = []
        max_length = 280  # Set max_length to 280 tokens

        for text in X_data:
            # Tokenize the input and apply padding to 280 tokens
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,  # Hardcoded max length to 280 tokens
                truncation=True,
                padding='max_length'  # Ensure all sequences are padded to 280 tokens
            )

            # Move inputs to the appropriate device (GPU if available)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Extract embeddings without computing gradients
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Take the last hidden state which is a sequence of token embeddings
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, embedding_dim)
            #print(last_hidden_state.size())
            # Remove the extra dimension if necessary
            last_hidden_state = torch.squeeze(last_hidden_state)  # Removes the extra 1 dimension if it exists
            #print(last_hidden_state.size())
            # Append the 3D tensor for this tweet to the embeddings list
            embeddings.append(last_hidden_state.cpu().numpy())

        # Convert the list of embeddings to a numpy array with shape (num_tweets, 280, embedding_dim)
        return np.array(embeddings)



class GPT2Embedding(object):
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Set pad_token to eos_token or define a custom one
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token as padding token
        # If you want to add a custom PAD token instead, you can do this:
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = GPT2Model.from_pretrained('gpt2').to(self.device)
        self.dim = 768  # GPT-2 base model's output dimension

    def __call__(self, **kwargs):
        X_train = kwargs['X_train']
        X_val = kwargs['X_val']
        X_test = kwargs['X_test']

        X_train_vectors_gpt2 = self.transform(X_train)
        X_val_vectors_gpt2 = self.transform(X_val)
        X_test_vectors_gpt2 = self.transform(X_test)

        return X_train_vectors_gpt2, X_val_vectors_gpt2, X_test_vectors_gpt2

    def transform(self, X_data):
        embeddings = []
        max_length = 280  # Set max_length to 280 tokens

        for text in X_data:
            # Tokenize the input and apply padding to 280 tokens
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,  # Hardcoded max length to 280 tokens
                truncation=True,
                padding='max_length'  # Ensure all sequences are padded to 280 tokens
            )

            # Move inputs to the appropriate device (GPU if available)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Extract embeddings without computing gradients
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Take the last hidden state which is a sequence of token embeddings
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, embedding_dim)

            # Use torch.squeeze() to remove any extra dimensions
            last_hidden_state = torch.squeeze(last_hidden_state)

            # Append the 3D tensor for this tweet to the embeddings list
            embeddings.append(last_hidden_state.cpu().numpy())

        # Convert the list of embeddings to a numpy array with shape (num_tweets, 280, embedding_dim)
        return np.array(embeddings)

