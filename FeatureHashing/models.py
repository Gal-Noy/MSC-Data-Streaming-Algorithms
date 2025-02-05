from typing import Literal

from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from concurrent.futures import ThreadPoolExecutor


def train_and_evaluate_model(X, y, model: Literal['sgd', 'knn', 'svm'], test_size=0.3, random_state=42, kwargs=None) -> int:
    if kwargs is None:
        kwargs = {}
        
    if model == 'sgd':
        clf = SGDClassifier(loss='log_loss', **kwargs)
    elif model == 'knn':
        clf = KNeighborsClassifier(**kwargs)
    elif model == 'svm':
        clf = SVC(**kwargs)
    else:
        raise NotImplementedError(f"Model {model} not implemented")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# def train_and_evaluate_model_in_parallel(Xys, model: Literal['sgd', 'knn', 'svm'], kwargs=None) -> list:
#     if kwargs is None:
#         kwargs = {}
    
#     with ThreadPoolExecutor() as executor:
#         results = list(executor.map(lambda Xy: train_and_evaluate_model(Xy[0], Xy[1], model, kwargs=kwargs), Xys))
#     return results