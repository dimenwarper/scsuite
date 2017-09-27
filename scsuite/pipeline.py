import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import FastICA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import SpectralEmbedding

import hdbscan
from magic.MAGIC_core import magic

from scimitar.branching import BranchedEmbeddedGaussians as BEG


DIM_RED_METHODS = dict(tsne=TSNE, 
                       isomap=Isomap, 
                       ica=FastICA)


class SCWorkspace(dict):
    pass

def plot_representation(rep, colors=None, categorical=False):
    if colors is None:
        colors = 'blue'
        cmap = None
    else:
        if categorical:
            cmap = plt.get_cmap('rainbow')
        else:
            cmap = plt.get_cmap('plasma')
    plt.scatter(rep[:, 0],
                rep[:, 1],
                s=100, linewidth=0., alpha=0.5,
                c=colors, cmap=cmap)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

def filter_percent_expressed(df, frac):
    selected_genes = df.columns[((df != 0).mean(axis=0) > frac).values]
    return df[selected_genes]

def pca_and_plot(data_df, scws):
    scws['dim_red'] = {}
    scws['dim_red']['pca'] = PCA(n_components=100).fit_transform(scale(data_df))

    plt.plot(scws['dim_red']['pca'].var(axis=0))
    plt.xlabel('PC')
    plt.ylabel('Variance')

def dim_reduction(data_df, scws,
                  n_pcs=10,
                  n_components=2,
                  methods=['tsne', 'isomap', 'ica']):

    for name in methods:
        method = DIM_RED_METHODS[name](n_components=n_components)
        pca_tx = scws['dim_red']['pca'][:, :n_pcs]
        scws['dim_red'][name] = method.fit_transform(pca_tx)

        plt.figure()
        plt.title(name)
        plt.scatter(scws['dim_red'][name][:, 0],
                    scws['dim_red'][name][:, 1],
                    s=100, linewidth=0., alpha=0.5)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')


def density_cluster(data_df, scws,
                    min_samples=1, min_cluster_size=40):
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples, 
                                min_cluster_size=min_cluster_size, 
                                prediction_data=True)

    clusterer.fit(data_df)

    soft_assignments = hdbscan.all_points_membership_vectors(clusterer)
    cluster_labels = np.array([np.argmax(x) for x in soft_assignments])
    scws['cluster'] = {}
    scws['cluster']['labels'] = cluster_labels
    scws['cluster']['proba'] = soft_assignments

def entropy2(data):
    bin_fracs = np.histogram(data.values, bins=50, density=True)[0]
    return entropy(bin_fracs)


def adaptive_rbf_matrix(data_array, n_neighbors=30, scale=1.0):
    n_samples = data_array.shape[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(data_array)
    A = pairwise_distances(data_array, metric='l2')
    
    n_distances = np.reshape(nn.kneighbors(data_array)[1][:, -1], (n_samples, 1))
    S = np.dot(n_distances, n_distances.T) / A.mean()
    A = np.exp(-scale * (A + 1)/(S + 1))
    return A


def trajectory_discovery(data_df, scws):
    scws['cluster']['magic'] = {}
    scws['cluster']['diff_map'] = {}
    scws['cluster']['entropies'] = {}
    for label in np.unique(scws['cluster']['labels']):
        sample_mask = scws['cluster']['labels'] == label
        scws['cluster']['magic'][label] = pd.DataFrame(magic(data_df[sample_mask]),
                                                index=data_df[sample_mask].index,
                                                columns=data_df.columns)
        scws['cluster']['magic'][label].fillna(0, inplace=True)
        scws['cluster']['magic'][label] = filter_percent_expressed(scws['cluster']['magic'][label], 0.1)
        
        A = adaptive_rbf_matrix(scws['cluster']['magic'][label])
        embedding = SpectralEmbedding(n_components=2, affinity='precomputed')
        scws['cluster']['diff_map'][label] = embedding.fit_transform(A)
        scws['cluster']['diff_map'][label] = pd.DataFrame(scws['cluster']['diff_map'][label],
                                                index=data_df[sample_mask].index,
                                                columns=['diff_map_1', 'diff_map_2'])
        
        scws['cluster']['entropies'][label] = scws['cluster']['magic'][label].apply(entropy2, axis=1).values
        plt.figure()
        plt.title('Cluster %s' % label)
        plot_representation(scws['cluster']['diff_map'][label].values, 
                            scws['cluster']['entropies'][label]) 

def plot_scimitar_model(pt, tx):
    plt.scatter(tx[:, 0], tx[:, 1], alpha=0.1, s=200)

    plt.scatter(pt.node_positions[:, 0], pt.node_positions[:, 1], alpha=0.5, c='k', s=100)
    X, Y = [pt.node_positions[i, 0] for i in range(pt.node_positions.shape[0])], [pt.node_positions[i, 1] 
                                                   for i in range(pt.node_positions.shape[0])]
    for i, j in pt.graph.edges():
        plt.plot([X[i], X[j]], [Y[i], Y[j]],'k-', zorder=1, linewidth=1)
    

def scimitar_model(data_df, scws,
                   cluster_label,
                   n_nodes=100, 
                   n_components=2, 
                   npcs=10, 
                   sigma=0.1, 
                   gamma=0.5, 
                   n_neighbors=30, 
                   just_visualize=False):
    
    model = BEG(n_nodes=n_nodes, 
                npcs=npcs, 
                max_iter=10, 
                sigma=sigma,
                gamma=gamma,
                n_neighbors=n_neighbors,
                embedding_dims=n_components,
                just_tree=just_visualize)
   
    data = scws['cluster']['magic'][cluster_label].values
    model.fit(data)
    
    assignments = np.zeros([data.shape[0]]) - 1
    neg_log_probs = cdist(model._embedding_tx, model.node_positions) 
    for i in range(data.shape[0]):    
        assignments[i] = np.argmin(neg_log_probs[i, :])
    
    num_leafs = len([n for n in model.graph.nodes_iter() if model.graph.degree(n) == 1])
    print('Number of branches: %s' % (num_leafs - 1))
    plt.figure()
    plot_scimitar_model(model, model._embedding_tx)

    scws['cluster']['model'] = model 
    scws['cluster']['neg_log_probs'] = neg_log_probs
    scws['cluster']['assignments'] = assignments
