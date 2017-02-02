from . import utils
from . import logging
import pandas as pd
import json
import os


class CellAtlas(object):

    def __init__(self, models={}):
        #self._check_models(models)
        self.models = models
        """
        self._coordinates_df = pd.DataFrame()
        self._coordinates_df['model'] = pd.Series()
        i = 0
        for name, model in models.items():
            self._coordinates_df = self._coordinates_df.append(model.coordinates, verify_integrity=True)
            added_size = model.coordinates.shape[0]
            self._coordinates_df['model'].iloc[i:i + added_size] = name
            i += added_size
        """
        
    def _check_models(self, models):
        df1 = models.values()[0].coordinates(with_annotations=False)
        n_dims = df1.shape[1]
        variables = df1.columns
        for name, model in models.items():
            df2 = models.coordinates(with_annotations=False)
            c1 = n_dims == df2.shape[1]
            c2 = variables == df2.columns
            if not (c1 and c2):
                raise ValueError('Models to build atlas have different variables or number of dimensions')

    # do score matrix for all models x samples, then use that as space to 
    def score(self, X):
        scores_df = pd.DataFrame(columns=['sample', 'score', 'coordinate_name', 'model'])
        for name, model in self.models.items():
            model_scores_df = model.score(X)
            model_scores_df['model'] = pd.Series([name]*model_scores_df.shape[0])
            scores_df = scores_df.append(model_scores_df)
        scores_df = scores_df.reset_index(drop=True)
        return scores_df

    def coordinates(self, with_annotations=True):
        if with_annotations:
            return self._coordinates_df
        else:
            return self._coordinates_df.iloc[:, self.n_dims]

    @staticmethod
    def load(dir):
        models = {}
        for fname in os.listdir(dir):
            if '.json' in fname:
                logging.debug('Loading %s/%s' % (dir, fname))
                model_name = fname.replace('.json', '')
                json_file = '%s/%s' % (dir, fname)
                json_dict = json.load(open(json_file))
                model_class = utils.str_to_class(json_dict['model'])
                #TODO better way to load file, not read json file twice
                models[model_name] = model_class.load(open(json_file))
        return CellAtlas(models=models)
