import argparse
import os
import numpy as np
import pandas as pd
import json

from . import config
from . import jobs
from . import utils
from . import instyaml
from . import logging
from . import atlas
from . import representations

class Command(object):

    def setup_clparser(self, parser):
        raise NotImplementedError('Command.setup_clparser not implemented')

    def execute(self, clargs):
        raise NotImplementedError('Command.execute not implemented')

class RepresentCommand(Command):

    def setup_clparser(self, parser):
        parser.add_argument('--data', type=str, default=None)
        parser.add_argument('--representations', nargs='+', type=str, default=['tsne', 'diffusion-map'])
        parser.add_argument('--npcs', type=int, default=10)
        parser.add_argument('--ndims', type=int, default=2)
        return parser

    def execute(self, clargs):
        general_config = config.load()
        if 'representations' in general_config:
            reprs_config = general_config['representations']
            reprs_kwargs = utils.to_kwargs(reprs_config['method']['params'])
            reprs = reprs_config['method']['class'](**reprs_kwargs)
        else:
            reprs = representations.SpectralNLECellRepresentation(n_components=clargs.ndims,
                                                                  n_pca_dims=clargs.npcs,
                                                                  nles=clargs.representations)

        if clargs.data is None:
            data_df = pd.read_csv(general_config['data'], sep='\t', index_col=0)
        else:
            data_df = pd.read_csv(clargs.data, sep='\t', index_col=0)

        logging.info('Computing representation  %s.%s' % (reprs.__module__,
                                                          reprs.__class__.__name__))
        transformed = reprs.fit_transform(data_df)
        if type(reprs.name) == list:
            names = reprs.name
            txs = transformed
        else:
            names = [reprs.name]
            txs = [transformed]
       
        dir = 'representations'
        if not os.path.exists(dir):
            os.mkdir(dir)

        for name, tx in zip(names, txs):
            with open('%s/%s.tsv' % (dir, name), 'w') as outfile:
                outfile.write('Sample\t%s\n' % '\t'.join(['V%s' % j for j in range(tx.shape[1])]))
                for i in range(data_df.shape[0]):
                    outfile.write('%s\t%s\n' % (data_df.index[i], '\t'.join([str(f) for f in tx[i, :]])))
                

class SplitCommand(Command):

    def setup_clparser(self, parser):
        pass

    def execute(self, clargs):
        general_config = config.load()
        split_config = general_config['split']
        model_recommendation_config = general_config['model_recommendation']

        if 'data' in split_config:
            data_df = pd.read_csv(split_config['data'], sep='\t', index_col=0)
        else:
            data_df = pd.read_csv(general_config['data'], sep='\t', index_col=0)
        split_kwargs = utils.to_kwargs(split_config['method']['params'])
        splitter = split_config['method']['class'](**split_kwargs)

        logging.info('Splitting using %s.%s' % (split_config['method']['class'].__module__,
                                                split_config['method']['class'].__name__))
        assignments = splitter.fit_predict(data_df)

        logging.info('Found %s cell clusters' % len(np.unique(assignments)))
        if -1 in assignments:
            logging.warning('%s cells did not fall in any cluster (marked as the gray cluster)' % ((assignments == -1).sum()))
       
        job_dir = './model_recommendation'
        fit_dir = './model_fit'
        data_dir = general_config['data_dir'] + '/data_chunks'

        for dir in [job_dir, data_dir, fit_dir]:
            dir = dir.replace('//', '/')
            if not os.path.exists(dir):
                os.mkdir(dir)

        for cluster in np.unique(assignments):
            cl_mask = assignments == cluster
            color = utils.get_color(cluster)
            
            cluster_data_fname = '%s/%s.tsv' % (data_dir, color)
            with open(cluster_data_fname, 'w') as cluster_data_file:
                cluster_data_file.write(data_df[cl_mask].to_csv(sep='\t'))

            model_recommendation_job = jobs.ModelRecommendationJob(job_name=color + '_model_recommendation',
                                                         data_filename=cluster_data_fname,
                                                         strategy={'class': model_recommendation_config['strategy']['class'],
                                                                   'params': model_recommendation_config['strategy']['params']},
                                                         out_file='./model_fit/%s_fit.yaml' % color)
            model_recommendation_job.save(open('%s/%s_model_recommendation.yaml' % (job_dir, color), 'w'))


class JobCommand(Command):

    def __init__(self, default_job_dir='', 
                 default_results_dir='', job_class=jobs.Job):
        self.default_job_dir = default_job_dir
        self.default_results_dir = default_results_dir
        self.job_class = job_class

    def setup_clparser(self, parser):
        parser.add_argument('--job-file', type=argparse.FileType('r'), default=None)
        parser.add_argument('--job-dir', type=str, default='%s/' % self.default_job_dir)
        parser.add_argument('--results-dir', type=str, default='%s/' % self.default_results_dir)
        return parser

    def execute(self, clargs):
        #TODO Adapt to various cluster/parallelization environments
        if len(clargs.results_dir) > 0:
            if not os.path.exists(clargs.results_dir):
                os.mkdir(clargs.results_dir)
        if clargs.job_file is not None:
            job_instance = self.job_class.load(clargs.job_file)
            job_instance.run()
        else:
            for fname in os.listdir(clargs.job_dir):
                if '.yaml' in fname:
                    fl = open('%s/%s' % (clargs.job_dir, fname))
                    job_instance = self.job_class.load(fl)
                    job_instance.run()


class ModelRecommendationCommand(JobCommand):

    def __init__(self):
        JobCommand.__init__(self, default_job_dir='./model_recommendation/', job_class=jobs.ModelRecommendationJob)


class FitCommand(JobCommand):

    def __init__(self):
        JobCommand.__init__(self, default_job_dir='./model_fit/', 
                            default_results_dir='./atlas/', job_class=jobs.ModelFitJob)


class ScoreCommand(Command):

    def setup_clparser(self, parser):
        parser.add_argument('samples_tsv', type=argparse.FileType('r'))
        parser.add_argument('--atlas-dir', type=str, default='./atlas/')
        parser.add_argument('--output-type', type=str, default='tsv')
        return parser

    def execute(self, clargs):
        logging.info('Loading atlas')
        cell_atlas = atlas.CellAtlas.load(clargs.atlas_dir)
        samples_df = pd.read_csv(clargs.samples_tsv, sep='\t', index_col=0)
        scores = cell_atlas.score(samples_df)
        if clargs.output_type == 'json':
            logging.result(json.dumps(scores.to_dict(orient='index'), 
                                      indent=4, separators=(',', ': ')))
        if clargs.output_type == 'tsv':
            logging.result(scores.to_csv(sep='\t'))

class SCRANCommand(Command):

    def setup_clparser(self, parser):
        parser.add_argument('fname')
        parser.add_argument('--sizes', default='"c(20, 40, 60, 80, 100)"')
        parser.add_argument('--cluster-min-size', default='200')
        parser.add_argument('--transpose', action='store_false', default=True)
        parser.add_argument('--do-cluster', action='store_true', default=False)
        parser.add_argument('--positive', action='store_true', default=False)

        return parser

    def execute(self, clargs):
        kws = ['fname', 'transpose', 'do_cluster', 'sizes', 'cluster_min_size', 'positive']
        cmdargs = {kw: clargs.__dict__[kw] for kw in kws}
        os.system('scran.R %(fname)s %(transpose)s %(do_cluster)s %(sizes)s %(cluster_min_size)s %(positive)s' % cmdargs)

class M3DropCommand(Command):

    def setup_clparser(self, parser):
        parser.add_argument('fname')
        parser.add_argument('--transpose', action='store_false', default=True)

        return parser

    def execute(self, clargs):
        kws = ['fname', 'transpose']
        cmdargs = {kw: clargs.__dict__[kw] for kw in kws}
        os.system('m3drop.R %(fname)s %(transpose)s' % cmdargs)



class StartCommand(Command):

    def setup_clparser(self, parser):
        parser.add_argument('project_name', type=str)
        return parser

    def execute(self, clargs):
        if os.path.exists(clargs.project_name):
            raise OSError('Path %s already exists' % clargs.project_name)
        else:
            os.mkdir(clargs.project_name)
            os.mkdir('%s/data' % clargs.project_name)
            
            config_dict = dict(data='./data/data.tsv', 
                               data_dir='./data/',
                               representations=dict(method={'class':'scsuite.representations.SpectralNLECellRepresentation',
                                                            'params':{}
                                                            }),
                               split=dict(data='./representations/tsne.tsv',
                                          method={'class': 'scsuite.split.DensitySplitter', 
                                                  'params': {}
                                                  }),
                               model_recommendation=dict(strategy={'class': 'scsuite.model_recommendation.PrincipalTopologyModelRecommendation',
                                                              'params': {}
                                                              })
                               )
            instyaml.dump(config_dict, open('%s/config.yaml' % clargs.project_name, 'w'))
