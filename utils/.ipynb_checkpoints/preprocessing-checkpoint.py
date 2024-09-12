import nltk
import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer

def preprocess_text(text):
    text = text.lower() # lowercase text
    text = text.strip()  # get rid of leading/trailing whitespace 
    text = re.compile('<.*?>').sub('', text) #Remove HTML tags/markups
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  #Replace punctuation with space. Careful since punctuation can sometime be useful
    text = re.sub('\s+', ' ', text)  #Remove extra space and tabs
    text = re.sub(r'\[[0-9]*\]',' ',text) #[0-9] matches any digit (0 to 10000...)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) #matches any digit from 0 to 100000..., \D matches non-digits
    text = re.sub(r'\s+',' ',text) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 
    return text

def remove_stopwords(text):
    text = [i for i in text.split() if i not in stopwords.words('english')]
    return ' '.join(text)

# snow = SnowballStemmer('english')
def stemming(text, word_tokenize, stemmer):
    text = [stemmer.stem(i) for i in word_tokenize(text) ]
    return ' '.join(text)

def get_wordnet_pos(tag):
    """
    This is a helper function to map NTLK position tags.
    Full list is available here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(text, wl):
    word_pos_tags = nltk.pos_tag(word_tokenize(text)) # Get position tags
    text = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return ' '.join(text)

def preprocessing_compose(text):
    """
    Apply all text preprocessing steps to the input text.
    """
    wl = WordNetLemmatizer()
    return lemmatizer(remove_stopwords(preprocess_text(text)), wl)