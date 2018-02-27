import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

class FeatureLoadingsPlot:

    def __init__(self, dataframe, pca, n_components, n_offset=0):
        self.dataframe = dataframe
        self.pca = pca
        self.n_components = n_components
        self.n_offset = n_offset

        self.pca_components = pca.components_[n_offset:n_components+n_offset]
        self.explained_variance_ratios = list(pca.explained_variance_ratio_[n_offset:n_components+n_offset])
        self.dimension_names = ['Dimension {}'.format(i+n_offset) for i in range(1,len(self.pca_components)+1)]

        self.pca_components_df = self._create_pca_dataframe_with_blank_column_for_legend()
        self.explained_variance_ratios_df = self._create_explained_variance_df()

    def _display_explained_variance_ratios(self):
        for i, ev in enumerate(self.explained_variance_ratios[:self.n_components]):
            label = "Explained Variance\n        {:.4}".format(ev)
            plt.text(i-0.40,
                     plt.ylim()[1],
                     label)

    def _create_pca_dataframe_with_blank_column_for_legend(self):
        self.dimension_names += ['']
        self.explained_variance_ratios += [0]

        pca_components_df = pd.DataFrame(self.pca_components, columns=self.dataframe.columns)
        pca_components_df = pca_components_df.append(0*pca_components_df.loc[0])

        pca_components_df.index = self.dimension_names
        return pca_components_df

    def _create_explained_variance_df(self):
        explained_variance_ratios_df = pd.DataFrame({'Explained Variance' : self.explained_variance_ratios})
        explained_variance_ratios_df.index = self.dimension_names
        return explained_variance_ratios_df

    def display_segments(self):
        n = len(self.dimension_names)
        self.pca_components_df.plot(figsize = (20,6), kind = 'bar')
        self._display_explained_variance_ratios()
        plt.legend(loc=1)
        plt.suptitle("Feature Loadings")
        plt.xticks(np.arange(n), self.dimension_names, rotation=0)


class Clusters:
    
    def __init__(self, dataframe, range_n_clusters):
        self.dataframe = dataframe
        self.range_n_clusters = range_n_clusters
        self._train_cluster_models()
        
    def _train_cluster_models(self):
        self.cluster_models = {}
        self.cluster_labels = {}
        self.cluster_centers = {}
        self.silhouette_scores = {}
        
        
        for n in tqdm(self.range_n_clusters):
            self.cluster_models[n] = KMeans(n_clusters=n)
            self.cluster_models[n].fit(self.dataframe)
            self.cluster_labels[n] = self.cluster_models[n].predict(self.dataframe)
            self.cluster_centers[n] = pd.DataFrame(self.cluster_models[n].cluster_centers_, columns=self.dataframe.columns)
            
            self.silhouette_scores[n] = silhouette_score(self.dataframe, self.cluster_labels[n])
            print("For n_clusters = {} the silhouette score is {}.".format(n, self.silhouette_scores[n]))
            
    def _cluster_plot(self, n, ax, dim_1, dim_2, samples=None):
        colors = cm.spectral(self.cluster_labels[n].astype(float)/n)
        x_dim = 'Dimension {}'.format(dim_1)
        y_dim = 'Dimension {}'.format(dim_2)
        ax.scatter(self.dataframe[x_dim], self.dataframe[y_dim], 
                   marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        centers = self.cluster_centers[n]
        centers.plot(kind='scatter', x=x_dim, y=y_dim, ax=ax, 
                     marker='o', c="white", alpha=1, s=400, edgecolor='k')

        for i, center in enumerate(centers.index):
            centers.loc[[center]].plot(kind='scatter', x=x_dim, y=y_dim, ax=ax, 
                                 marker='$%d$' % i, alpha=1, s=50, edgecolor='k')
            
        if samples is not None:
            samples.plot(kind='scatter', x=x_dim, y=y_dim, ax=ax,
                         s=250, linewidth=2, color='black', marker='x')

        ax.set_title("Clustered data for {} clusters.".format(n))
        ax.set_xlabel(x_dim)
        ax.set_ylabel(y_dim)
            
    def _silhouette_plot(self, n, ax):
        sample_silhouette_values = silhouette_samples(self.dataframe, self.cluster_labels[n])
        
        y_lower = 10
        for i in range(n):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[self.cluster_labels[n] == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

            ax.set_title("The silhouette plot for the {} clusters.".format(n))
            ax.set_xlabel("The silhouette coefficient values")
            ax.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax.axvline(x=self.silhouette_scores[n], color="red", linestyle="--")

            ax.set_yticks([])  # Clear the yaxis labels / ticks
            ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            
    def cluster_plots(self, dim_1=1, dim_2=2, samples=None):
        
        fig, ax = plt.subplots(1, len(self.range_n_clusters))
        fig.set_size_inches(24, 6)

        for n, ax in zip(self.range_n_clusters, ax):
            self._cluster_plot(n, ax, dim_1, dim_2, samples)
    
    def silhouette_plots(self):
        
        fig, ax = plt.subplots(1, len(self.range_n_clusters))
        fig.set_size_inches(24, 6)

        for n, ax in zip(self.range_n_clusters, ax):
            self._silhouette_plot(n, ax)