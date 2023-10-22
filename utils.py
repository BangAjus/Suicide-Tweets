import nltk
import re
from pickle import load
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def cleaning_one(stri):

    stri = stri.split(" ")
    for i in range(len(stri)):
        for j in ['@', '#']:
            if j in stri[i]:
                stri[i] = stri[i].replace(stri[i], "")
    stri = " ".join(stri)
    stri = re.sub(r"[\W+0-9+]", " ", stri)
    stri = re.findall(r"[A-Za-z]+", stri)
    stri = " ".join(stri)
    stri = stri.lower()

    return stri

def nltk_processing(stri):

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    stri = stri.split(" ")
    stri = [ps.stem(i) for i in stri if i not in stop_words]
    stri = " ".join(stri)

    return stri

def tf_processing(corpus):

    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = load(file)
    sequences = tokenizer.texts_to_sequences(corpus)
    maxlen = 115
    sequences = pad_sequences(sequences, maxlen=maxlen)

    return sequences