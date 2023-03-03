import numpy as np
from MCCA import MCCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold


class MCCATransformer(BaseEstimator, TransformerMixin):
    """ Implements MCCA transformation for use in sklearn pipelines.

    Parameters:
        n_components_pca (int): Number of PCA components to retain for each subject (default 50)

        n_components_mcca (int): Number of MCCA components to retain (default 10)

        reg (bool): Whether to add regularization. (default False)

        r (int/float): Regularization strength. (default 0)

        nested_cv (bool): If true, use nested cross-validation for new subjects fitted with fit_transform_online.
                          (default True)
    """

    def __init__(self, n_components_pca=50, n_components_mcca=10, reg=False, r=0, nested_cv=True):
        """ Init. """
        self.MCCA = MCCA(n_components_pca, n_components_mcca, reg, r, False)
        self.MCCA_new = MCCA(n_components_pca, n_components_mcca, reg, r, True)
        self.nested_cv = nested_cv
        self.cca_averaged = None

    def fit(self, X, y):
        """ Fit the MCCA weights to the training data X with labels y.
        
        Parameters:
            X (ndarray): The training data (subjects x trials x samples x channels)
                
            y (ndarray): Labels corresponding to trials in X. (subjects x trials)
        """
        data_averaged = [_compute_prototypes(X[i], y[i]) for i in range(len(X))]
        data_averaged = np.stack(data_averaged, axis=0)
        # apply M-CCA to averaged data
        cca_averaged = self.MCCA.obtain_mcca(data_averaged)
        self.cca_averaged = np.mean(cca_averaged, axis=0)
        return self

    def transform(self, X):
        """ 
        Transform single-trial data from sensor dimensions to CCA dimensions,
        concatenate trials across subjects, and flatten time and CCA dimensions
        for the classifier.
        """
        X_ = [self.MCCA.transform_trials(X[i], subject=i) for i in range(len(X))]
        X_ = np.concatenate(X_)
        return X_.reshape((X_.shape[0], -1))

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)

    def fit_online(self, X, y):
        """ 
        Fit MCCA weights to data from a new subject.
        """
        if self.cca_averaged is None:
            raise NotFittedError('MCCA transformer needs to be fitted to ' +
                                 'training data before calling fit_online')
        data_averaged = _compute_prototypes(X, y)
        # Calculate PCA for new subject
        pca_averaged = self.MCCA_new.obtain_mcca(data_averaged[np.newaxis])
        pca_averaged = np.squeeze(pca_averaged)
        # Fit PCA of the new subject to average CCA from training data
        self.MCCA_new.weights_mcca = np.dot(np.linalg.pinv(pca_averaged), self.cca_averaged)[np.newaxis]
        self.MCCA_new.mcca_fitted = True
        return self

    def transform_online(self, X):
        """ 
        Transform data from a new subject from sensor to CCA dimensions, and 
        flatten time and CCA dimensions for the classifier.
        """
        X = self.MCCA_new.transform_trials(X)
        return X.reshape((X.shape[0], -1))

    def fit_transform_online(self, X, y):
        if self.nested_cv:
            # Nested cross-validation: 5-fold stratified shuffle split of the new (/left-out) subject's trials
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            Xs = []
            ys = []
            for train, test in cv.split(X, y):
                Xs.append(self.fit_online(X[train], y[train]).transform_online(X[test]))
                ys.append(y[test])
            return np.concatenate(Xs), np.concatenate(ys)
        else:
            return self.fit_online(X, y).transform_online(X)

    def transform_pca_only(self, X):
        """ 
        Transform single-trial data from sensor dimensions to PCA dimensions,
        concatenate trials across subjects, and flatten time and PCA dimensions
        for the classifier.
        """
        X_ = [self.MCCA.transform_trials_pca(X[i], subject=i) for i in range(len(X))]
        X_ = np.concatenate(X_)
        return X_.reshape((X_.shape[0], -1))

    def transform_online_pca_only(self, X):
        """ 
        Transform data from a new subject from sensor to PCA dimensions, and 
        flatten time and PCA dimensions for the classifier.
        """
        X = self.MCCA_new.transform_trials_pca(X)
        return X.reshape((X.shape[0], -1))


def _compute_prototypes(X, y):
    return np.concatenate([np.mean(X[np.where(y == class_)], axis=0) for class_ in np.unique(y)])
