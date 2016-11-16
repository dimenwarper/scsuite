from sklearn.linear_model import LassoCV
import numpy as np

class Bulk2CellDeconvolution(object):
    
    def deconvolve(self, X):
        raise NotImplementedError('Bulk2CellDeconvolution.deconvolve not implemented')
    

class LassoDeconvolution(Bulk2CellDeconvolution):

    def __init__(self, atlas):
        self.atlas = atlas
        self._lasso = LassoCV(n_alphas=10, fit_intercept=False)

    def deconvolve(self, X):
        basis = self.atlas.coordinates(with_annotations=False)
        basis = basis.values.T
        coeffs = np.zeros([X.shape[0], basis.shape[1]])
        for i in xrange(X.shape[0]):
            self._lasso.fit(basis, X[i, :].T)
            coeffs[i, :] = self._lasso.coef_
        return coeffs
