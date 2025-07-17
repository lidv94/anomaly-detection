from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from package.utils import timer

@timer
def get_model(method:str, params:dict):
    if method == 'kmeans':
        return KMeans(**params)
    if method == 'gmm':
        return GaussianMixture(**params)
    if method == 'dbscan':
        return DBSCAN(**params)
    raise ValueError(f"{method} is not implemented!")
    
import pandas as pd
from package.utils import timer

@timer
def predict_model(model, method, X):
    proxy = X.copy()
    if method in ['kmeans', 'hdbscan', 'dbscan']:
        return predict_all(model=model, X=proxy)
    if method=='gmm':
        return predict_gmm(model=model, X=proxy)

@timer
def predict_all(model, X):
    labels = model.labels_
    return labels

@timer
def predict_gmm(model, X):
    labels = model.predict(X)
    return labels

@timer
def add_labels(X, labels):
    proxy = X.copy()
    proxy = proxy.reset_index()
    proxy = proxy[[proxy.columns[0]]]
    proxy['labels'] = labels
    return proxy