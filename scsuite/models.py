import numpy as np
import pandas as pd
import json
import scipy.stats
from scimitar.morphing_mixture import MorphingGaussianMixture


class CellPopulationModel(object):
    
    @property
    def coordinates(self):
        raise NotImplementedError('CellPopulationModel.coordinates not implemented')

    def fit(self, X):
        raise NotImplementedError('CellPopulationModel.fit not implemented')
    
    # returns: where to place, in absolute coordinates, each sample
    def place(self, X):
        raise NotImplementedError('CellPopulationModel.place not implemented')
    
    # returns: a sample_size x number_of_model_coordinates matrix of values
    # scoring each sample on each coordinate. Effectively, this is the 'error model'
    # of each coordinate
    def score(self, X):
        raise NotImplementedError('CellPopulationModel.score not implemented')

    # returns: True if the model was fitted, False else
    def _check_model_fitted(self):
        raise NotImplementedError('CellPopulationModel._check_model_fitted not implemented')

    def save(self, fl):
        raise NotImplementedError('CellPopulationModel.save not implemented')

    @staticmethod
    def load(self, fl):
        raise NotImplementedError('CellPopulationModel.load not implemented')
    
    # Use this to dump the classes param dict into 
    # a properly, scSUITE-compatible json
    def _json_dump(self, param_dict, fl):
        cls = self.__class__
        model_name = '%s.%s' % (cls.__module__, cls.__name__)
        json.dump(dict(model=model_name, params=param_dict), fl,
                  indent=4, separators=(',', ': '))



VALID_COV_ESTIMATORS = ['sample']


class GaussianCellPopModel(CellPopulationModel):

    def __init__(self, cov_estimator='sample'):
        self.mean = None
        self.cov = None
        if cov_estimator not in VALID_COV_ESTIMATORS:
            raise ValueError('Covariance estimator %s not recognized' % cov_estimator)
        self.cov_estimator = cov_estimator

    def estimate_covariance(self, X):
        if self.cov_estimator == 'sample':
            return np.cov(X.T)

    def _check_model_fitted(self):
        assert self.mean is not None

    @property
    def coordinates(self):
        self._check_model_fitted()
        return self._coordinates_df
        
    def _generate_coordinates_df(self):
        self._coordinates_df = pd.DataFrame(np.reshape(self.mean,
                                                       (1, self.n_dims)),
                                            columns=self.variables, 
                                            index=pd.Series(['mean']))

    def fit(self, X):
        self.n_dims = X.shape[1]
        self.variables = X.columns
        self.mean = X.mean(axis=0)
        self.cov = self.estimate_covariance(X)
        self._generate_coordinates_df()

    def place(self, X):
        self._check_model_fitted()
        return np.tile(self.mean, (X.shape[0], 1))

    def score(self, X):
        self._check_model_fitted()
        scores = scipy.stats.multivariate_normal.pdf(X, mean=self.mean,
                                                     cov=self.cov, allow_singular=True)
        max_score = scipy.stats.multivariate_normal.pdf(self.mean, mean=self.mean,
                                                        cov=self.cov, allow_singular=True)
        scores /= max_score

        scores_df = pd.DataFrame(scores, columns=['score'])
        scores_df['sample'] = pd.Series(X.index.values)
        scores_df['coordinate_name'] = pd.Series(['mean']*X.shape[0])
        return scores_df

    def save(self, fl):
        self._check_model_fitted()
        param_dict = dict(cov_estimator=self.cov_estimator,
                          mean=self.mean.tolist(),
                          cov=self.cov.tolist(),
                          variables=self.variables.tolist())
        self._json_dump(param_dict, fl)

    @staticmethod
    def load(fl):
        json_dict = json.load(fl)
        param_dict = json_dict['params']
        model = GaussianCellPopModel()
        model.__dict__.update(param_dict)
        model.cov = np.array(model.cov)
        model.mean = np.array(model.mean)
        model.n_dims = model.cov.shape[0]
        model._generate_coordinates_df()
        return model

class CurveSCIMITARCellPopModel(GaussianCellPopModel):
    pass
class BranchingSCIMITARCellPopModel(GaussianCellPopModel):
    pass
'''
class SCIMITARCellPopModel(CellPopulationModel):
    
    def __init__(self, scimitar_class, 
                 reference_points, 
                 reference_point_names,
                 **scimitar_kwargs):
        self._scimitar_model = scimitar_class(**scimitar_kwargs)
        self.means = None
        self.covs = None
        self.reference_points = reference_points
        self.reference_point_names = reference_point_names

    def set_reference_structure(self):
        raise NotImplementedError('SCIMITARCellPopModel.set_reference_structure not implemented')

    def _map_samples_to_reference(self, X):
        raise NotImplementedError('SCIMITARCellPopModel.map_samples_to_reference')

    def _score_samples_to_reference(self, X):
        raise NotImplementedError('SCIMITARCellPopModel.map_samples_to_reference')

    def _check_model_fitted(self):
        assert self.means is not None

    @property
    def coordinates(self):
        self._check_model_fitted()
        return self._coordinates_df

    def _generate_coordinates_df(self):
        self._coordinates_df = pd.DataFrame(self.means,
                                            columns=self.variables,
                                            index=pd.Series(self.reference_points))

    def fit(self, X):
        self.n_dims = X.shape[1]
        self.variables = X.columns
        self._scimitar_model.fit(X)
        refined_model, self.fitted_pseudotimes = self._scimitar_model.refine(cov_reg=0.01)
        self._scimitar_model = refined_model
        self.means = self._scimitar_model.mean(self.reference_points)
        self.covs = self._scimitar_model.covariance(self.reference_points)
        self._generate_coordinates_df()

    def place(self, X):
        self._check_model_fitted()
        pts = self._map_samples_to_reference(X)
        indices = [self.reference_points.index(pt) for pt in pts]
        return self.means[indices, :]

    def score(self, X):
        self._check_model_fitted()
        pt_probs = self._score_samples_to_reference(X)
        scores_df = pd.DataFrame()
        for i, t in enumerate(self.reference_points):
            tp_df = pd.DataFrame(pt_probs[:, i], columns=['score'])
            tp_df['sample'] = pd.Series(X.index.values)
            
            name = self.reference_point_names[i]
            tp_df['coordinate_name'] = pd.Series([name]*X.shape[0])
            scores_df = scores_df.append(tp_df)
        return scores_df
    
    def save(self, fl):
        self._check_model_fitted()
        param_dict = dict(means=self.means.tolist(),
                            covs=self.covs.tolist(),
                            variables=self.variables.tolist(),
                            reference_points=self.reference_points.tolist(),
                            reference_point_names=self.reference_point_names,
                            reference_structure=self.reference_structure)
        self._json_dump(param_dict, fl)

    @staticmethod
    def load(fl, cls):
        json_dict = json.load(fl)
        param_dict = json_dict['params']
        model = cls()
        model.__dict__.update(param_dict)
        model.means = np.array(model.means)
        model.covs = np.array(model.covs)
        model.n_dims = model.covs.shape[1]
        model._generate_coordinates_df()
        return model




class CurveSCIMITARCellPopModel(SCIMITARCellPopModel):

    def __init__(self):
        mgm = MorphingGaussianMixture
        self.timepoints = np.arange(0, 1, 0.01)
        self.timepoint_names = ['pseudotime_%s' % t for t in timepoints]
        kwargs = {}
        SCIMITARCellPopModel.__init__(self, mgm, 
                                      self.timepoints,
                                      self.timepoint_names,
                                      **kwargs)

        def set_reference_structure(self):
            self.reference_structure = []
            with self.timepoints as tp:
                for i in xrange(1, len(tp)):
                    edge = (tp[i-1], tp[i])
                    self.reference_structure.append(edge)

        @staticmethod
        def load(fl):
            SCIMITARCellPopModel.load(fl, self.__class__)

class BranchingSCIMITARCellPopModel(SCIMITARCellPopModel):
    
    def __init__(self):
        bmgm = BranchingMorphingMixture
        num_nodes = 50

        nodes = np.arange(num_nodes)
        
        node_names = ['node_%s' % n for n in nodes]
        kwargs = dict(num_nodes=num_nodes)
        SCIMITARCellPopModel.__init__(self, bmgm,
                                      nodes,
                                      node_names,
                                      **kwargs)

        def set_reference_structure(self):
            self.reference_structure = self.scimitar_model.tree.edges()

        @staticmethod
        def load(fl):
            SCIMITARCellPopModel.load(fl, self.__class__)
'''
