import models
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GMM as GaussianMixture

from . import logging


class ModelRecommendationStrategy(object):

    def select(self, X):
        raise NotImplementedError('ModelRecommendationStrategy.select not implemented')


class PrincipalTopologyModelRecommendation(ModelRecommendationStrategy):

    def __init__(self):
        self._cluster_method = GaussianMixture
        self._cluster_kwargs = dict(covariance_type='diag')

    def select(self, X):
        prev_bic = np.inf
        n_samples = X.shape[0]
        n_points = 1
        for n_points in xrange(1, min(n_samples/2, 100)):
            meth_inst = self._cluster_method(n_components=n_points, **self._cluster_kwargs)
            meth_inst.fit(X)
            curr_bic = meth_inst.bic(X)
            if prev_bic > curr_bic:
                prev_bic = curr_bic
            else:
                break
        logging.debug('Found %s principal points' % n_points)
        if n_points == 1:
            logging.debug('Recommending models.GaussianCellPopModel')
            return models.GaussianCellPopModel, {}
        else:
            centroids = meth_inst.means_
            distance_matrix = squareform(pdist(centroids))
            G = nx.Graph(distance_matrix)
            T = nx.minimum_spanning_tree(G)
            if nx.diameter(T) >= int(0.8 * len(T)):
                logging.debug('Recommending models.SCIMITARCellPopModel')
                return models.SCIMITARCellPopModel, {}
            else:
                logging.debug('Recommending models.BranchingSCIMITARCellPopModel')
                return models.BranchingSCIMITARCellPopModel, {}
