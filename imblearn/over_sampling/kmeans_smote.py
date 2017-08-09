"""Class to perform over-sampling using K-Means SMOTE."""

# Authors: Felix Last
# License: MIT

from __future__ import division

import numpy as np
import warnings
import math
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances

from ..over_sampling.base import BaseOverSampler
from ..over_sampling import SMOTE
from ..over_sampling import RandomOverSampler
from ..exceptions import raise_isinstance_error
from ..utils import check_neighbors_object
from ..utils.deprecation import deprecate_parameter



class KMeansSMOTE(BaseOverSampler):
    """Class to perform over-sampling using K-Means SMOTE.

    This method uses K-Means to cluster the dataset. Clusters are filtered
    based on the minority ratio.
    Parameters equivalent to SMOTE: KMeansSMOTE(imbalance_ratio_threshold=float('Inf'), kmeans_args={'n_clusters':1})
    Parameters equivalent to RandomOverSampling: KMeansSMOTE(imbalance_ratio_threshold=float('Inf'), kmeans_args={'n_clusters':1}, smote_args={'k_neighbors':0})

    Parameters
    ----------
    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    imbalance_ratio_threshold : float, optional (default=1.0)
        Specify a threshold for a cluster's imbalance ratio  ``((majority_count + 1) / (minority_count + 1))``
        under which the cluster will be considered a minority cluster.

    density_power : float, optional (default=None)
        Used to compute the minority "density" of each cluster. By default, the number of features will be used.

    minority_weight : float, optional (default=0.66)
        If kmeans_args['n_clusters'] is not specified, this parameter will be used
        to automatically determine a reasonable value for the number of clusters.
        The number of clusters is then computed as the weighted arithmetic mean
        of the number of minority instances and the number of majority instances.
        This parameter ``w1`` specifies the weight of the minority instances; the weight of
        majority instances is set to ``1-w1``.
        ``n_clusters = (w1 * minority_count) + ((1-w1) * majority_count)

    kmeans_args : dict, optional (default={})
        Parameters to be passed to sklearn.cluster.KMeans. If n_clusters
        is not explicitly set, it will be automatically set; KMeans' default will
        not apply.

    smote_args : dict, optional (default={})
        Parameters to be passed to imblearn.over_sampling.SMOTE. Parameter ratio
        will be overwritten according to ratio passed to this class. Random state
        will be passed from this class if none is specified.
    """

    def __init__(self,
                ratio='auto',
                random_state=None,
                imbalance_ratio_threshold=1.0,
                minority_weight=0.66,
                density_power=None,
                kmeans_args={},
                smote_args={}):
        super(KMeansSMOTE, self).__init__(ratio=ratio, random_state=random_state)
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        self.kmeans_args = kmeans_args
        self.smote_args = smote_args
        if random_state is not None:
            if ('random_state' not in self.smote_args):
                self.smote_args['random_state'] = random_state
            if ('random_state' not in self.kmeans_args):
                self.kmeans_args['random_state'] = random_state
        self.minority_weight = minority_weight

        self.density_power = density_power

    def _cluster(self, X):
        """Run k-means to cluster the dataset

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        Returns
        -------
        cluster_assignment : ndarray, shape (n_samples)
            The corresponding cluster label of `X_resampled`.
        """

        kmeans = MiniBatchKMeans(**self.kmeans_args)

        if(X.shape[0] < kmeans.n_clusters):
            warnings.warn('Adapting kmeans_args.n_clusters to {0} because X only has {1} samples.'.format(X.shape[0],X.shape[0]))
            self.kmeans_args['n_clusters'] = X.shape[0]
            kmeans = MiniBatchKMeans(**self.kmeans_args)

        if 'init_size' not in self.kmeans_args:
            self.kmeans_args['init_size'] = min(2 * self.kmeans_args['n_clusters'], X.shape[0])
            kmeans = MiniBatchKMeans(**self.kmeans_args)

        kmeans.fit_transform(X)
        cluster_assignment = kmeans.labels_
        # kmeans.labels_ does not use continuous labels,
        # i.e. some labels in 0..n_clusters may not exist. Tidy up this mess.
        return cluster_assignment

    def _filter_clusters(self, X, y, cluster_assignment, minority_class_label):
        """Run k-means to cluster the dataset

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.
        cluster_assignment : ndarray, shape (n_samples)
            The corresponding cluster label of `X_resampled`.
        minority_class_label : int
            Label of the minority class to filter by.

        Returns
        -------
        sampling_weights : ndarray, shape (np.max(np.unique(cluster_assignment)),)
            Vector of sampling weights for each cluster
        """
        # compute the shape of the density factors
        # since the cluster labels are not continuous, make it large enough
        # to fit all values up to the largest cluster label
        largest_cluster_label = np.max(np.unique(cluster_assignment))
        density_factors = np.zeros((largest_cluster_label + 1,), dtype=np.float64)
        minority_mask = (y == minority_class_label)
        density_sum = 0
        for i in np.unique(cluster_assignment):
            cluster = X[cluster_assignment == i]
            mask = minority_mask[cluster_assignment == i]
            minority_count = cluster[mask].shape[0]
            majority_count = cluster[-mask].shape[0]
            imbalance_ratio = (majority_count + 1) / (minority_count + 1)
            if (imbalance_ratio < self.imbalance_ratio_threshold) and (minority_count > 1):
                average_minority_distance = np.mean(euclidean_distances(cluster))
                if average_minority_distance is 0: average_minority_distance = 1e-1 # to avoid division by 0
                density_factor = minority_count / (average_minority_distance ** self.density_power)
                density_factors[i] = density_factor

        # prevent division by zero; set zero weights in majority clusters
        density_sum = density_factors.sum()
        if density_sum == 0: density_sum = 1 # to avoid division by zero
        density_sum = np.full(density_factors.shape, density_sum, np.asarray(density_sum).dtype)
        sampling_weights = (density_factors / density_sum)

        return sampling_weights


    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """

        # determine optimal number of clusters if none is given
        if 'n_clusters' not in self.kmeans_args:
            labels, generate_counts = zip(*self.ratio_.items())
            minority_index, majority_index = np.argmax(generate_counts), np.argmin(generate_counts)
            minority_label, majority_label = labels[minority_index], labels[majority_index]
            minority_count = X[y == minority_label].shape[0]
            majority_count = X[y == majority_label].shape[0]
            if (self.minority_weight < 0) or (self.minority_weight > 1):
                raise ValueError('minority_weight should be between 0 and 1. Got {}'.format(self.minority_weight))
            # n_clusters is set to weighted mean of minority and majority count
            self.kmeans_args['n_clusters'] = math.floor(
                (minority_count * self.minority_weight)
                + (majority_count * (1-self.minority_weight)))

        if self.density_power is None:
            self.density_power = X.shape[1]

        resampled = list()
        for minority_class_label, n_samples in self.ratio_.items():
            if n_samples == 0:
                continue

            cluster_assignment = self._cluster(X)
            sampling_weights = self._filter_clusters(X, y, cluster_assignment, minority_class_label)
            smote_args = self.smote_args.copy()
            if np.count_nonzero(sampling_weights) > 0:
                # perform k-means smote
                for i in np.unique(cluster_assignment):
                    cluster_X = X[cluster_assignment == i]
                    cluster_y = y[cluster_assignment == i]
                    if sampling_weights[i] > 0:
                        # determine ratio for oversampling the current cluster
                        target_ratio = {label: np.count_nonzero(cluster_y == label) for label in self.ratio_}
                        cluster_minority_count = np.count_nonzero(cluster_y == minority_class_label)
                        generate_count = int(round(n_samples * sampling_weights[i]))
                        target_ratio[minority_class_label] = generate_count + cluster_minority_count

                        # make sure that cluster_y has more than 1 class, adding a random point otherwise
                        remove_index = -1
                        if np.unique(cluster_y).size < 2:
                            remove_index = cluster_y.size
                            cluster_X = np.append(cluster_X, np.zeros((1,cluster_X.shape[1])), axis=0)
                            majority_class_label = next( key for key in self.ratio_.keys() if key != minority_class_label )
                            target_ratio[majority_class_label] = 1 + target_ratio[majority_class_label]
                            cluster_y = np.append(cluster_y, np.asarray(majority_class_label).reshape((1,)), axis=0)

                        # modify copy of the user defined smote_args to reflect computed parameters
                        smote_args = self.smote_args.copy()
                        smote_args['ratio'] = target_ratio

                        smote_args = self._validate_smote_args(smote_args, cluster_minority_count)
                        oversampler = SMOTE(**smote_args)

                        # if k_neighbors is 0, perform random oversampling instead of smote
                        if ('k_neighbors' in smote_args) & (smote_args['k_neighbors'] == 0):
                                oversampler = RandomOverSampler(random_state=smote_args['random_state'])

                        # finally, apply smote to cluster
                        with warnings.catch_warnings():
                            # ignore warnings about minority class getting bigger than majority class
                            # since this would only be true within this cluster
                            warnings.filterwarnings(action='ignore', category=UserWarning, message='After over-sampling\, the number of samples \(.*\) in class .* will be larger than the number of samples in the majority class \(class #.* \-\> .*\)')
                            cluster_resampled_X, cluster_resampled_y = oversampler.fit_sample(cluster_X, cluster_y)

                        if remove_index > -1:
                            # since SMOTE's results are ordered the same way as the data passed into it,
                            # the temporarily added point is at the same index position as it was added.
                            cluster_resampled_X = np.delete(cluster_resampled_X, remove_index, 0)
                            cluster_resampled_y = np.delete(cluster_resampled_y, remove_index, 0)

                        resampled.append( (cluster_resampled_X, cluster_resampled_y) )

                    else:
                        # don't oversample this cluster, just add it to resampled results
                        resampled.append((cluster_X, cluster_y))
            else:
                # all weights are zero -> perform regular smote
                minority_count = np.count_nonzero( y == minority_class_label )
                warnings.warn('No minority clusters found for class {}. Performing regular SMOTE. Try changing the number of clusters. Recommended number of clusters: between {} and the number of majority class instances.'.format(minority_class_label, minority_count))

                smote_args = self._validate_smote_args(smote_args, minority_count)
                oversampler = SMOTE(**smote_args)
                return oversampler.fit_sample(X, y)

        resampled = list(zip(*resampled))
        if(len(resampled) > 0):
            X_resampled = np.concatenate(resampled[0], axis=0)
            y_resampled = np.concatenate(resampled[1], axis=0)
        else:
            # no samples were generated because none were requested
            X_resampled = X.copy()
            y_resampled = y.copy()
        return X_resampled, y_resampled


    def _validate_smote_args(self, smote_args, minority_count):
        # determine max number of nearest neighbors considering sample size
        max_k_neighbors =  minority_count - 1
        # check if max_k_neighbors is violated also considering smote's default
        smote = SMOTE(**smote_args)
        if smote.k_neighbors > max_k_neighbors:
            smote_args['k_neighbors'] = max_k_neighbors
            warnings.warn('Adapting smote_args.k_neighbors to {0} because cluster only has {1} minority samples.'.format(max_k_neighbors, minority_count))
            smote = SMOTE(**smote_args)
        return smote_args
