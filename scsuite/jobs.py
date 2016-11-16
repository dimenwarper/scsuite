import pandas as pd
from . import models
from . import model_recommendation
from . import instyaml
from . import logging


class Job(object):

    def run(self):
        raise NotImplementedError('Job.run not implemented')

    def description(self):
        raise NotImplementedError('Job.description not implemented')

    def save(self, fl):
        raise NotImplementedError('Job.save not implemented')
    
    @staticmethod
    def load(fl):
        raise NotImplementedError('Job.load not implemented')


class ModelFitJob(Job):
    
    def __init__(self, job_name='',
                 data_filename='',
                 model={'class': models.CellPopulationModel, 'params': {}},
                 out_file=''):
        self.job_name = job_name
        self.data_filename = data_filename
        self.model = model
        self.out_file = out_file
        self._description = 'Fitting model %s with job name %s ==> %s' % (self.model['class'].__name__, self.job_name, self.out_file)

    def run(self):
        logging.info(self._description)
        _data = pd.read_csv(self.data_filename, sep='\t', index_col=0)
        _model = self.model['class'](**self.model['params'])
        _model.fit(_data)
        _model.save(open(self.out_file, 'w'))
    
    @property
    def description(self):
        return self._description

    def save(self, fl):
        yaml_dict = dict(job_name=self.job_name,
                         data_filename=self.data_filename,
                         model=self.model, 
                         out_file=self.out_file)

        instyaml.dump(yaml_dict, fl)

    @staticmethod
    def load(fl):
        yaml_dict = instyaml.parse_yaml(fl)
        return ModelFitJob(**yaml_dict)


class ModelRecommendationJob(Job):

    def __init__(self, job_name='',
                 data_filename='',
                 strategy={'class': model_recommendation.ModelRecommendationStrategy, 'params': {}},
                 out_file=''):
        self.job_name = job_name
        self.data_filename = data_filename
        self.strategy = strategy
        self.out_file = out_file
        self._description = 'Model selection with strategy %s with job name %s ==> %s' % (self.strategy['class'].__name__, self.job_name, self.out_file)

    def run(self):
        logging.info(self._description)
        _data = pd.read_csv(self.data_filename, sep='\t', index_col=0)
        _strategy = self.strategy['class'](**self.strategy['params'])
        model_class, model_params = _strategy.select(_data)
        model_fit_job = ModelFitJob(data_filename=self.data_filename,
                                    job_name=self.job_name.replace('model_recommendation', 'fit'),
                                    model={'class': model_class, 'params': model_params},
                                    out_file='./atlas/%s_model.json' % self.job_name.replace('model_recommendation', 'model'))
        model_fit_job.save(open(self.out_file, 'w'))

    @property
    def description(self):
        return self._description

    def save(self, fl):
        yaml_dict = dict(job_name=self.job_name,
                         data_filename=self.data_filename,
                         strategy=self.strategy,
                         out_file=self.out_file)

        instyaml.dump(yaml_dict, fl)

    @staticmethod
    def load(fl):
        yaml_dict = instyaml.parse_yaml(fl)
        return ModelRecommendationJob(**yaml_dict)
