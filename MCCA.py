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

        reg (bool): Whether to add regularization (default False)

        r (int or float): Regularization strength (default 0)

        pca_only (bool): If true, skip MCCA calculation (default False)

    Attributes:
        mu (ndarray): PCA mean (subjects, PCAs)

        sigma (ndarray): PCA standard deviation (subjects, PCAs)

        weights_pca (ndarray): PCA weights that transform sensors to PCAs for each
                               subject (subjects, sensors, PCAs)

        weights_mcca (ndarray): MCCA weights that transform PCAs to MCCAs for each subject.
                                None if pca_only is True. (subjects, PCAs, MCCAs)
    """

    def __init__(self, n_components_pca=50, n_components_mcca=10, reg=False, r=0, pca_only=False):
        """ Init. """
        if n_components_mcca > n_components_pca:
            import warnings
            warnings.warn(f"Warning........... number of MCCA components ({n_components_mcca}) cannot be greater than "
                          f"number of PCA components ({n_components_pca}), setting them equal.")
            n_components_mcca = n_components_pca
        self.n_components_pca = n_components_pca
        self.n_components_mcca = n_components_mcca
        self.reg = reg
        self.r = r
        self.pca_only = pca_only
        self.weights_mcca, self.weights_pca, self.mu, self.sigma = None, None, None, None
        self.pca_fitted, self.mcca_fitted = False, False

    def obtain_mcca(self, X):
        """ Apply individual-subject PCA and across-subjects MCCA.

        Parameters:
            X (ndarray): Input data in sensor space (subjects, samples, sensors)

        Returns:
            scores (ndarray): Returns scores in PCA space if self.pca_only is true and MCCA scores otherwise.
        """
        n_subjects, n_samples, n_sensors = X.shape
        X_pca = np.zeros((n_subjects, n_samples, self.n_components_pca))
        self.weights_pca = np.zeros((n_subjects, n_sensors, self.n_components_pca))
        self.mu = np.zeros((n_subjects, n_sensors))
        self.sigma = np.zeros((n_subjects, self.n_components_pca))

        # obtain subject-specific PCAs
        for i in range(n_subjects):
            tmpscore = np.squeeze(X[i]).copy()  # time x sensors
            pca = PCA(n_components=self.n_components_pca, svd_solver='full')
            score = pca.fit_transform(tmpscore)
            self.weights_pca[i] = pca.components_.T
            self.mu[i] = pca.mean_
            self.sigma[i] = np.sqrt(pca.explained_variance_)
            score /= self.sigma[i]
            X_pca[i] = score

        self.pca_fitted = True
        if self.pca_only:
            return X_pca
        else:
            return self._mcca(X_pca)

    def _mcca(self, X_pca):
        """ Performs multiset canonical correlation analysis with an optional
            regularization term based on spatial similarity of weight maps. The
            stronger the regularization, the more similar weight maps are forced to
            be across subjects.

        Parameters:
            X_pca (ndarray): Input data in PCA space (subjects, samples, PCAs)

        Returns:
            X_mcca (ndarray): Input data in MCCA space (subjects, samples, MCCAs).
        """
        n_subjects, n_samples, _ = X_pca.shape
        _, n_sensors, _ = self.weights_pca.shape
        KK = self.n_components_pca
        K = self.n_components_mcca

        temp = np.zeros((KK * n_subjects, n_samples))
        for i in range(n_subjects):
            temp[i * KK:(i + 1) * KK, :] = X_pca[i].T

        # R is a block matrix containing all cross-covariances R_kl = X_k^T X_l
        R = np.cov(temp)
        # S is a block diagonal matrix containing auto-correlations R_kk = X_k^T X_k
        # in its diagonal blocks
        S = np.zeros_like(R)
        for i in range(1, KK * n_subjects + 1):
            tmp = np.ceil(i / KK).astype(int)
            S[(tmp - 1) * KK:tmp * KK, i - 1] = R[(tmp - 1) * KK:tmp * KK, i - 1]

        # Regularization
        if self.reg:
            # R2 and S2 are calculated the same way as R and S above, but using
            # cross-covariance of PCA weights instead of PCA scores
            temp = np.zeros((KK * n_subjects, n_sensors))
            for i in range(n_subjects):
                temp[i * KK:(i + 1) * KK, :] = self.weights_pca[i].T
            R2 = np.cov(temp)
            S2 = np.zeros_like(R2)
            for i in range(1, KK * n_subjects + 1):
                tmp = np.ceil(i / KK).astype(int)
                S2[(tmp - 1) * KK:tmp * KK, i - 1] = R2[(tmp - 1) * KK:tmp * KK, i - 1]
            # Add regularization term to R and S
            R = R + self.r * R2
            S = S + self.r * S2

        # Obtain MCCA solution by solving the generalized eigenvalue problem
        # (R - S) h^i = p^i S h^i
        # where h^i is the concatenation of i-th eigenvectors of all subjects and
        # p^i is the i-th generalized eigenvalue (i-th canonical correlation)
        _, tempW = eigh((R - S), S, eigvals=(KK * n_subjects - K, KK * n_subjects - 1))
        # eigh returns smallest eigenvalues first, flip to have largest first
        tempW = np.flip(tempW, axis=1)
        # tempW contains large eigenvectors which are a concatenation of individual
        # subjects' eigenvectors/MCCA transformation weights. Reshape weights into
        # eigenvectors for individual subjects. Also normalize each subject's weights.
        self.weights_mcca = np.zeros((n_subjects, KK, K))
        for i in range(n_subjects):
            W_subject = tempW[i * KK:(i + 1) * KK, :]
            self.weights_mcca[i] = W_subject / norm(W_subject, ord=2, keepdims=True)

        self.mcca_fitted = True
        return np.matmul(X_pca, self.weights_mcca)

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
        if not self.mcca_fitted:
            raise NotFittedError('MCCA needs to be fitted before calling transform_trials')
        X -= self.mu[np.newaxis, np.newaxis, subject]  # centered
        X_pca = np.matmul(X, self.weights_pca[subject])
        X_pca /= self.sigma[np.newaxis, np.newaxis, subject]  # standardized
        X_mcca = np.matmul(X_pca, self.weights_mcca[subject])
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
        if not self.mcca_fitted:
            raise NotFittedError('MCCA needs to be fitted before calling inverse_transform_trials')
        X_pca = np.matmul(X_mcca, np.linalg.pinv(self.weights_mcca[subject]))  # trials x samples x PCAs
        X_pca *= self.sigma[np.newaxis, np.newaxis, subject]  # revert standardization
        X = np.matmul(X_pca, np.linalg.pinv(self.weights_pca[subject]))
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
        if not self.pca_fitted:
            raise NotFittedError('PCA needs to be fitted before calling transform_trials_pca')
        X -= self.mu[np.newaxis, np.newaxis, subject]  # centered
        X_pca = np.matmul(X, self.weights_pca[subject])
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
        if not self.mcca_fitted:
            raise NotFittedError('MCCA needs to be fitted before calling test_mcca')
        t_mu = self.mu[:, np.newaxis]
        t_sigma = self.sigma[:, np.newaxis]
        data_train -= t_mu  # centered
        pca_train = np.matmul(data_train, self.weights_pca)  # PCAs obtained
        pca_train /= t_sigma  # normalized
        mcca_train = np.matmul(pca_train, self.weights_mcca)  # MCCAs obtained
        data_test -= t_mu  # centered
        pca_test = np.matmul(data_test, self.weights_pca)  # PCAs obtained
        pca_test /= t_sigma  # normalized
        mcca_test = np.matmul(pca_test, self.weights_mcca)  # MCCAs obtained

        K = self.weights_mcca.shape[2]
        correlations = np.zeros((K, 2))
        for i in range(K):
            inter_subject_corr_train = np.corrcoef(np.squeeze(mcca_train[:, :, i]))
            inter_subject_corr_test = np.corrcoef(np.squeeze(mcca_test[:, :, i]))
            # averaged inter-subject correlations over training data
            correlations[i, 0] = np.mean(inter_subject_corr_train)
            # averaged inter-subject correlations over testing data
            correlations[i, 1] = np.mean(inter_subject_corr_test)

        return correlations
