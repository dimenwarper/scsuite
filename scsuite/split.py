from sklearn.cluster import DBSCAN
from . import representations
from . import utils


class CellTypeSplitter(object):

    def fit(self, X):
        raise NotImplementedError('CellTypeSplitter.fit not implemented')

    def predict(self, X):
        raise NotImplementedError('CellTypeSplitter.predict/split not implemented')

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def split(self, X):
        return self.fit_predict(X)


class DensitySplitter(CellTypeSplitter):

    def __init__(self, clustering_method='dbscan', 
                 representation=representations.IdentityCellRepresentation()):
        self._representation = representation
        if clustering_method == 'dbscan':
            self._clustering_method = DBSCAN(eps=3, algorithm='kd_tree')

    def fit(self, X):
        self._transformed = self._representation.fit_transform(X)

    def predict(self, X):
        labels = self._clustering_method.fit_predict(self._transformed)

        import matplotlib.pyplot as plt
        plt.scatter(self._transformed[:, 0], self._transformed[:, 1], alpha=0.6, color=utils.get_color(labels))
        plt.show()
        return labels
