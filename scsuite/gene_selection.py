from scipy.optimize import curve_fit
import numpy as np


class GeneSelectionMethod(object):

    def select_genes(self, X):
        raise NotImplementedError('GeneSelectionMethod.select_genes not implemented')


class DiffExpGeneSelection(GeneSelectionMethod):

    models = {'michaelis_menten': lambda mean, k: mean/(mean + k),
              'sigmoid': lambda mean, k: 1./(1 + np.exp(-mean*k))}

    def __init__(self, model='michaelis-menten', threshold=0.01):
        self.threshold = threshold
        if model not in DiffExpGeneSelection.models:
            raise ValueError('Model %s not recognized for DiffExpGeneSelection' % model)
            self.model = DiffExpGeneSelection[model]

    def select_genes(self, X):
        means = X.mean(axis=0)
        dropout_fractions = (X == 0).mean(axis=0)
        popt, pcov = curve_fit(self.model, means, dropout_fractions)
        perr = np.sqrt(np.diag(pcov))
        de_genes = means[means > self.model(means, popt) + perr].index
        import matplotlib.pyplot as plt
        plt.scatter(np.log(means + 1), dropout_fractions)
        r = np.arange(0, X.max(), X.max()/100)
        plt.plot(np.log(r + 1), self.model(r, popt), color='k')
        plt.scatter(np.log(means.ix[de_genes] + 1), dropout_fractions, color='red')
        return de_genes
