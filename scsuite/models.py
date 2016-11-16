import numpy as np
import pandas as pd
import json
import scipy.stats


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
        cls = self.__class__
        model_name = '%s.%s' % (cls.__module__, cls.__name__)
        json.dump(dict(model=model_name, params=param_dict), fl,
                  indent=4, separators=(',', ': '))

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

#TODO implement these using scimitar package
class BranchingSCIMITARCellPopModel(GaussianCellPopModel):
    pass

class SCIMITARCellPopModel(GaussianCellPopModel):
    pass
