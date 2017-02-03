from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import PCA


class CellRepresentation(object):
    
    def fit(self, X):
        raise NotImplementedError('CellRepresentation.fit not implemented')
    
    def transform(self, X):
        raise NotImplementedError('CellRepresentation.transform not implemented')

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class IdentityCellRepresentation(CellRepresentation):

    def fit(self, X):
        pass

    def transform(self, X):
        return X


class SpectralNLECellRepresentation(CellRepresentation):


    embedding_aliases = {'tsne': TSNE, 'diffusion-map': SpectralEmbedding}
    
    def __init__(self, n_components=2, n_pca_dims=10, nles=['tsne', 'diffusion-map']):
        self.name = nles
        self.n_components = n_components

        self._nles = [SpectralNLECellRepresentation.embedding_aliases[nle](n_components=self.n_components) for nle in nles]
        self._pca = PCA(n_components=n_pca_dims)

    def fit(self, X):
        self._pca.fit(X)

    def transform(self, X):
        pca_loadings = self._pca.transform(X)
        return [nle.fit_transform(pca_loadings) for nle in self._nles]
