from typing import Literal
import pandas as pd
from sklearn.conftest import fetch_20newsgroups


def load_news_category_dataset():
    chunk_size = 1000
    chunks = []
    for chunk in pd.read_json('News_Category_Dataset_v3.json', lines=True, chunksize=chunk_size):
        chunks.append(chunk)
    dataset_df = pd.concat(chunks, ignore_index=True)
    dataset_df['content'] = dataset_df[[c for c in dataset_df.columns if c != 'category']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return dataset_df['content'], dataset_df['category']

def load_dataset(dataset_name: Literal['news_category', '20newsgroups']):
    if dataset_name == 'news_category':
       return load_news_category_dataset()
    elif dataset_name == '20newsgroups':
        return fetch_20newsgroups(subset='all', return_X_y=True)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")