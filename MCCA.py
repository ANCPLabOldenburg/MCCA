import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from scipy.linalg import eigh, norm


class MCCA:
    """ Performs multiset canonical correlation analysis with an optional
        regularization term based on spatial similarity of weight maps. The
        stronger the regularization, the more similar weight maps are forced to
        be across subjects. Note that the term 'weights' is used interchangeably
        with PCA / MCCA eigenvectors here.

    Parameters:
        n_components_pca (int): Number of PCA components to retain for each subject (default 50)

        n_components_mcca (int): Number of MCCA components to retain (default 10)

        r (int or float): Regularization strength (default 0)

        pca_only (bool): If true, skip MCCA calculation (default False)

    Attributes:
        mu (ndarray): PCA mean (subjects, PCAs)

        sigma (ndarray): PCA standard deviation (subjects, PCAs)

        pca_weights (ndarray): PCA weights that transform sensors to PCAs for each
                               subject (subjects, sensors, PCAs)

        mcca_weights (ndarray): MCCA weights that transform PCAs to MCCAs for each subject.
                                None if pca_only is True. (subjects, PCAs, MCCAs)
    """

    def __init__(self, n_components_pca=50, n_components_mcca=10, r=0, pca_only=False):
        if n_components_mcca > n_components_pca:
            import warnings
            warnings.warn(f"Warning........... number of MCCA components ({n_components_mcca}) cannot be greater than "
                          f"number of PCA components ({n_components_pca}), setting them equal.")
            n_components_mcca = n_components_pca
        self.n_pca = n_components_pca
        self.n_mcca = n_components_mcca
        self.r = r
        self.pca_only = pca_only
        self.mcca_weights, self.pca_weights, self.mu, self.sigma = None, None, None, None

    def obtain_mcca(self, X):
        """ Apply individual-subject PCA and across-subjects MCCA.

        Parameters:
            X (ndarray): Input data in sensor space (subjects, samples, sensors)

        Returns:
            scores (ndarray): Returns scores in PCA space if self.pca_only is true and MCCA scores otherwise.
        """
        n_subjects, n_samples, n_sensors = X.shape
        X_pca = np.zeros((n_subjects, n_samples, self.n_pca))
        self.pca_weights = np.zeros((n_subjects, n_sensors, self.n_pca))
        self.mu = np.zeros((n_subjects, n_sensors))
        self.sigma = np.zeros((n_subjects, self.n_pca))

        # obtain subject-specific PCAs
        for i in range(n_subjects):
            pca = PCA(n_components=self.n_pca, svd_solver='full')
            x_i = np.squeeze(X[i]).copy()  # time x sensors
            score = pca.fit_transform(x_i)
            self.pca_weights[i] = pca.components_.T
            self.mu[i] = pca.mean_
            self.sigma[i] = np.sqrt(pca.explained_variance_)
            score /= self.sigma[i]
            X_pca[i] = score

        if self.pca_only:
            return X_pca
        else:
            return self._mcca(X_pca)

    def _mcca(self, pca_scores):
        """ Performs multiset canonical correlation analysis with an optional
            regularization term based on spatial similarity of weight maps. The
            stronger the regularization, the more similar weight maps are forced to
            be across subjects.

        Parameters:
            pca_scores (ndarray): Input data in PCA space (subjects, samples, PCAs)

        Returns:
            mcca_scores (ndarray): Input data in MCCA space (subjects, samples, MCCAs).
        """
        # R is a block matrix containing all cross-covariances R_kl = X_k^T X_l between subjects k, l
        # S is a block diagonal matrix containing auto-correlations R_kk = X_k^T X_k in its diagonal blocks
        R, S = _compute_cross_covariance(pca_scores)
        # Regularization
        if self.r != 0:
            # R2 and S2 are calculated the same way as R and S above, but using cross-covariance of PCA weights
            # instead of PCA scores
            R2, S2 = _compute_cross_covariance(self.pca_weights)
            # Add regularization term to R and S
            R += self.r * R2
            S += self.r * S2
        # Obtain MCCA solution by solving the generalized eigenvalue problem
        # (R - S) h^i = p^i S h^i
        # where h^i is the concatenation of i-th eigenvectors of all subjects and
        # p^i is the i-th generalized eigenvalue (i-th canonical correlation)
        p, h = eigh((R - S), S, subset_by_index=(R.shape[0] - self.n_mcca, R.shape[0] - 1))
        # eigh returns eigenvalues in ascending order. To pick the k largest from a total of n eigenvalues,
        # we use subset_by_index=(n - k, n - 1).
        # Flip eigenvectors so that they are in descending order
        h = np.flip(h, axis=1)
        # Reshape h from (subjects * PCAs, MCCAs) to (subjects, PCAs, MCCAs)
        h = h.reshape((pca_scores.shape[0], self.n_pca, self.n_mcca))
        # Normalize eigenvectors per subject
        self.mcca_weights = h / norm(h, ord=2, axis=(1, 2), keepdims=True)
        return np.matmul(pca_scores, self.mcca_weights)

    def transform_trials(self, X, subject=0):
        """ Use of MCCA weights (obtained from averaged data) to transform single
            trial data from sensor space to MCCA space.

        Parameters:
            X (ndarray): Single trial data of one subject in sensor space
                         (trials, samples, sensors)
            subject (int): Index of the subject whose data is being transformed

        Returns:
            X_mcca (ndarray): Transformed single trial data in MCCA space
                            (trials, samples, MCCAs)
        """
        if self.mcca_weights is None:
            raise NotFittedError('MCCA needs to be fitted before calling transform_trials')
        X -= self.mu[np.newaxis, np.newaxis, subject]  # centered
        X_pca = np.matmul(X, self.pca_weights[subject])
        X_pca /= self.sigma[np.newaxis, np.newaxis, subject]  # standardized
        X_mcca = np.matmul(X_pca, self.mcca_weights[subject])
        return X_mcca

    def inverse_transform_trials(self, X_mcca, subject=0):
        """ Transform single trial data from MCCA space back to sensor space.

        Parameters:
            X_mcca (ndarray): Single trial data of one subject in MCCA space
                            (trials, samples, MCCAs)
            subject (int): Index of the subject whose data is being transformed

        Returns:
            X (ndarray): Single trial data transformed back into sensor space
                         (trials, samples, sensors)
        """
        if self.mcca_weights is None:
            raise NotFittedError('MCCA needs to be fitted before calling inverse_transform_trials')
        X_pca = np.matmul(X_mcca, np.linalg.pinv(self.mcca_weights[subject]))  # trials x samples x PCAs
        X_pca *= self.sigma[np.newaxis, np.newaxis, subject]  # revert standardization
        X = np.matmul(X_pca, np.linalg.pinv(self.pca_weights[subject]))
        X += self.mu[np.newaxis, np.newaxis, subject]  # revert centering
        return X

    def transform_trials_pca(self, X, subject=0):
        """ Transform single trial data from sensor space to PCA space

        Parameters:
            X (ndarray): Single trial data of one subject in sensor space
                         (trials, samples, sensors)
            subject (int): Index of the subject whose data is being transformed

        Returns:
            X_pca (ndarray): Transformed single trial data in PCA space
                             (trials, samples, PCAs)
        """
        if self.pca_weights is None:
            raise NotFittedError('PCA needs to be fitted before calling transform_trials_pca')
        X -= self.mu[np.newaxis, np.newaxis, subject]  # centered
        X_pca = np.matmul(X, self.pca_weights[subject])
        X_pca /= self.sigma[np.newaxis, np.newaxis, subject]  # standardized
        return X_pca

    def test_mcca(self, data_train, data_test):
        """ Test if the inter-subject correlations/consistency observed from training
            data MCCAs generalize over testing data.

        Parameters:
            data_train (ndarray): Training data in sensor space
                                  (subjects, samples, sensors)
            data_test (ndarray): Test data in sensor space
                                 (subjects, samples, sensors)

        Returns:
            correlations (ndarray): Inter-subject correlations (averaged over
                                    every pair of subjects) over every MCCA component
                                    (first row: training data; second row: testing data)
        """
        if self.mcca_weights is None:
            raise NotFittedError('MCCA needs to be fitted before calling test_mcca')
        t_mu = self.mu[:, np.newaxis]
        t_sigma = self.sigma[:, np.newaxis]
        data_train -= t_mu  # centered
        pca_train = np.matmul(data_train, self.pca_weights)  # PCAs obtained
        pca_train /= t_sigma  # normalized
        mcca_train = np.matmul(pca_train, self.mcca_weights)  # MCCAs obtained
        data_test -= t_mu  # centered
        pca_test = np.matmul(data_test, self.pca_weights)  # PCAs obtained
        pca_test /= t_sigma  # normalized
        mcca_test = np.matmul(pca_test, self.mcca_weights)  # MCCAs obtained

        K = self.mcca_weights.shape[2]
        correlations = np.zeros((K, 2))
        for i in range(K):
            inter_subject_corr_train = np.corrcoef(np.squeeze(mcca_train[:, :, i]))
            inter_subject_corr_test = np.corrcoef(np.squeeze(mcca_test[:, :, i]))
            # averaged inter-subject correlations over training data
            correlations[i, 0] = np.mean(inter_subject_corr_train)
            # averaged inter-subject correlations over testing data
            correlations[i, 1] = np.mean(inter_subject_corr_test)

        return correlations


def _compute_cross_covariance(X):
    """ Computes cross-covariance of PCA scores or components between subjects.

    Parameters:
        X (ndarray): PCA scores (subjects, samples, PCAs) or weights (subjects, sensors, PCAs)

    Returns:
        R (ndarray): Block matrix containing all cross-covariances R_kl = X_k^T X_l between subjects k, l
                     with shape (subjects * PCAs, subjects * PCAs)
        S (ndarray): Block diagonal matrix containing auto-correlations R_kk = X_k^T X_k in its diagonal blocks
                     with shape (subjects * PCAs, subjects * PCAs)
    """
    n_subjects, n_samples, n_pca = X.shape
    R = np.cov(X.swapaxes(1, 2).reshape(n_pca * n_subjects, n_samples))
    S = R * np.kron(np.eye(n_subjects), np.ones((n_pca, n_pca)))
    return R, S
