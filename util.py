import os
import re
import nltk
import pickle
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder


stemmer = PorterStemmer()


def word_tokenize(text) :
    "tokenize given text"
    return [stemmer.stem(token) for token in nltk.word_tokenize(text)]


def clear_text(text) :
    "cleaning given text"
    text = text.lower()
    text = re.sub("[^0-9a-z]", " ", text)
    text = re.sub(" +", " ", text)
    return text


def load_data(feature_type="onehot"
            , path="data/20news-18828"
            , data_path="data.pkl"
            ) :
    "make data or load"

    if os.path.exists(data_path) :
        with open(data_path, "rb") as handle :
            return pickle.load(handle)

    document = []
    category = []
    folders = os.listdir(path)
    for folder in tqdm(folders) :
        folder_path = os.path.join(path, folder)
        files = os.listdir(folder_path)
        for file in files :
            file = open(os.path.join(folder_path, file), "rb")
            text = clear_text(file.read().decode("utf-8", "ignore"))
            document.append(text)
            category.append(folder)
            file.close()

    print(document[0])
    print(category[0])
    corpus = []
    for text in tqdm(document) :
        tokens = word_tokenize(text)
        if feature_type=="onehot" : 
            tokens = list(set(tokens))
        corpus.append(' '.join(tokens))
    print(corpus[0])

    if feature_type=="onehot" :
        Vectorizer = CountVectorizer
    else :
        Vectorizer = TfidfVectorizer
    vectorizer = Vectorizer(stop_words="english"
                            , analyzer=str.split
                            , min_df=2)

    token_features = vectorizer.fit_transform(corpus)
    category = LabelEncoder().fit_transform(category)
    
    data = {"feature":token_features
            , "category":category}
    
    with open(data_path, "wb") as handle :
        pickle.dump(data, handle)

    return data