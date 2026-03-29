# utils.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download once (ignore if already done)
nltk.download('stopwords')

# Initialize
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    text = text.strip()
    
    return text


def remove_stopwords(text):
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered)


def stem_text(text):
    words = text.split()
    stemmed = [stemmer.stem(word) for word in words]
    return " ".join(stemmed)


def remove_repeated_chars(text):
    # gooooood → good
    return re.sub(r'(.)\1+', r'\1\1', text)


def preprocess_text(text):
    text = clean_text(text)
    text = remove_repeated_chars(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text
