from sklearn.cluster import DBSCAN


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

    def __init__(self, clustering_method='dbscan'):
        if clustering_method == 'dbscan':
            self._clustering_method = DBSCAN(eps=3, algorithm='kd_tree')

    def fit(self, X):
        pass

    def predict(self, X):
        labels = self._clustering_method.fit_predict(X)
        return labels
