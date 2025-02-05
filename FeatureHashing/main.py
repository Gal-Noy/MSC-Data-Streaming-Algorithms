import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    X, y = load_dataset('news_category')
    
    # TF-IDF Bag of Words
    vec = TfidfVectorizer(stop_words='english')
    X_bow = vec.fit_transform(X)
    feature_names = vec.get_feature_names_out()
    N, D = X_bow.shape
    print(f'Number of samples: {N}, number of features: {D}')
    
    