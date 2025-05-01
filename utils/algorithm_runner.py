from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.cluster import DBSCAN

from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.kde import KDE
from pyod.models.gmm import GMM

from pyod.models.feature_bagging import FeatureBagging
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP
from pyod.models.inne import INNE

from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.lmdd import LMDD

import numpy as np

def get_base_models(X, model_cls, n=5):
    return [model_cls() for _ in range(n)]

def run_algorithm(name, X):
    name = name.upper()
    model = None

    if name == "CBLOF":
        model = CBLOF()
    elif name == "HBOS":
        model = HBOS()
    elif name == "KNN":
        model = KNN()
    elif name == "LOF":
        model = LOF()
    elif name == "DBSCAN":
        try:
            model = DBSCAN()
            labels = model.fit_predict(X)
            scores = np.where(labels == -1, 1.0, 0.0)  
            print(f"[DBSCAN] #outliers: {(scores == 1.0).sum()} / {len(scores)}")
            return scores
        except Exception as e:
            print(f"[DBSCAN ERROR] -> {e}")
            return np.zeros(len(X))

    elif name == "ABOD":
        model = ABOD(n_neighbors=20)
    elif name == "COPOD":
        model = COPOD()
    elif name == "ECOD":
        model = ECOD()
    elif name == "KDE":
        model = KDE()
    elif name == "GMM":
        model = GMM()

    elif name == "IFOREST":
        model = IForest()
    elif name in ["FEATUREBAGGING", "FB"]:
        model = FeatureBagging(base_estimator=LOF())
    elif name == "LSCP":
        try:
            model = LSCP(get_base_models(X, LOF))
        except Exception as e:
            print(f"[LSCP ERROR] -> {e}")
            return np.zeros(len(X))
    elif name == "INNE":
        model = INNE()

    elif name == "PCA":
        model = PCA()
    elif name == "OCSVM":
        model = OCSVM()
    elif name == "MCD":
        model = MCD()
    elif name == "LMDD":
        model = LMDD()

    else:
        raise ValueError(f"Unknown algorithm: {name}")

    try:
        model.fit(X)
        scores = model.decision_scores_
        print(f"[{name}] min={scores.min():.6f}, max={scores.max():.6f}, std={scores.std():.6f}")
    except Exception as e:
        print(f"[{name}] Fit error: {e}")
        scores = np.zeros(len(X))

    return scores