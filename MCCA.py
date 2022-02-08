import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import eigh, norm

def obtainCCA(X,KK=50,K=10,reg=False,r=0,pca_only=False,ret_scores=False):
    """ Apply individual-subject PCA and across-subjects CCA. Note that the
        term 'weights' is used interchangeably with PCA/CCA eigenvectors here.
    
    Parameters:
        X (ndarray): Input data in sensor space (subjects, samples, sensors)
        KK (int): Number of PCAs to retain for each subject (default 50)
        K (int): number of CCAs to retain (default 10)
        reg (bool): Whether to add regularization (default False)
        r (int or float): Regularization strength (default 0)
        pca_only (bool): If true, skip CCA and return only PCA transformation 
                         weights (default False)
        ret_scores (bool): If true, return scores along with weights 
                           (default False)
    
    Returns:
        W (ndarray): CCA weights that transform PCAs to CCAs for each subject. 
                     Not returned if pca_only is True. (subjects, PCAs, CCAs)
        mu (ndarray): PCA mean (subjects, PCAs)
        sigma (ndarray): PCA standard deviation (subjects, PCAs)
        weights_pca (ndarray): PCA weights that tranform sensors to PCAs for each 
                               subject (subjects, sensors, PCAs)
        scores (ndarray, optinal): If ret_scores is true, returns PCA weights
                                   if pca_only is true and CCA weights otherwise.
    """
    n_subjects, n_samples, n_sensors = X.shape
    X_pca = np.zeros((n_subjects,n_samples,KK))
    weights_pca = np.zeros((n_subjects,n_sensors,KK))
    mu = np.zeros((n_subjects,n_sensors))
    sigma = np.zeros((n_subjects,KK))
    
    # obtain subject-specific PCAs
    for i in range(n_subjects):
        tmpscore = np.squeeze(X[i]).copy() # time x sensors
        pca = PCA(n_components=KK,svd_solver='full')
        score = pca.fit_transform(tmpscore)
        weights_pca[i] = pca.components_.T
        mu[i] = pca.mean_
        sigma[i] = np.sqrt(pca.explained_variance_)
        score /= sigma[i]
        X_pca[i] = score
    
    if pca_only:
        if ret_scores:
            return mu,sigma,weights_pca,X_pca
        else:
            return mu,sigma,weights_pca
    else:
        # apply M-CCA
        W = mccas(X_pca,K,reg,r,weights_pca)
        if ret_scores:
            return W,mu,sigma,weights_pca,np.matmul(X_pca,W)
        else:
            return W,mu,sigma,weights_pca
    
def mccas(X_pca,K,reg,r,weights_pca):
    """ Performs multiset canonical correlation analysis with an optional 
        regularization term based on spatial similarity of weightmaps. The
        stronger the regularization, the more similar weightmaps are forced to
        be across subjects.
    
    Parameters:
        X_pca (ndarray): Input data in PCA space (subjects, samples, PCAs)
        K (int): number of CCAs to retain
        reg (bool): Whether to add regularization
        r (int or float): Regularization strength
        weights_pca (ndarray): PCA weights that tranform sensors to PCAs for each 
                               subject (subjects, sensors, PCAs)
    
    Returns:
        W (ndarray): CCA weights that transform PCAs to CCAs for each subject. 
                     (subjects, PCAs, CCAs)
    """
    n_subjects, n_samples, KK = X_pca.shape
    _, n_sensors, _ = weights_pca.shape
    if K > KK:
        K = KK
        
    temp = np.zeros((KK*n_subjects,n_samples))
    for i in range(n_subjects):
        temp[i*KK:(i+1)*KK,:] = X_pca[i].T

    # R is a block matrix containing all cross-covariances R_kl = X_k^T X_l
    R = np.cov(temp)
    # S is a block diagonal matrix containing auto-correlations R_kk = X_k^T X_k
    # in its diagonal blocks
    S = np.zeros_like(R)
    for i in range(1,KK*n_subjects+1):
        tmp = np.ceil(i/KK).astype(int)
        S[(tmp-1)*KK:tmp*KK,i-1] = R[(tmp-1)*KK:tmp*KK,i-1]
    
    # Regularization
    if reg:
        # R2 and S2 are calculated the same way as R and S above, but using
        # cross-covariance of PCA weights instead of scores
        temp = np.zeros((KK*n_subjects,n_sensors))
        for i in range(n_subjects):
            temp[i*KK:(i+1)*KK,:] = weights_pca[i].T
        R2 = np.cov(temp)
        S2 = np.zeros_like(R2)
        for i in range(1,KK*n_subjects+1):
            tmp = np.ceil(i/KK).astype(int)
            S2[(tmp-1)*KK:tmp*KK,i-1] = R2[(tmp-1)*KK:tmp*KK,i-1]
        # Add regularization term to R and S
        R = R + r*R2
        S = S + r*S2
    
    # Obtain MCCA solution by solving the generalized eigenvalue problem
    # (R - S) h^i = p^i S h^i
    # where h^i is the concatenation of i-th eigenvectors of all subjects and
    # p^i is the i-th generalized eigenvalue (i-th canonical correlation)
    _, tempW = eigh((R-S),S,eigvals=(KK*n_subjects-K,KK*n_subjects-1))
    # eigh returns smallest eigenvalues first, flip to have largest first
    tempW = np.flip(tempW,axis=1)
    # tempW contains large eigenvectors which are a concatenation of individual
    # subjects' eigenvectors/MCCA transformation weights. Reshape weights into
    # eigenvectors for individual subjects. Also normalize each subject's weights.
    W = np.zeros((n_subjects,KK,K))
    for i in range(n_subjects):
        W_subject = tempW[i*KK:(i+1)*KK,:] 
        W[i] = W_subject / norm(W_subject,ord=2,keepdims=True)
    
    return W

def transformTrials(X,W,mu,sigma,weights_pca,subject=0):
    """ Use of CCA weights (obtained from averaged data) to transform single
        trial data from sensor space to CCA space.
    
    Parameters:
        X (ndarray): Single trial data of one subject in sensor space 
                     (trials, samples, sensors)
        W (ndarray): CCA weights that transform PCAs to CCAs for each subject
                     (subjects, PCAs, CCAs)
        mu (ndarray): PCA mean (subjects, PCAs)
        sigma (ndarray): PCA standard deviation (subjects, PCAs)
        weights_pca (ndarray): PCA weights that tranform sensors to PCAs for each 
                               subject (subjects, sensors, PCAs)
        subject (int): Index of the subject whose data is being transformed 
                              
    Returns:
        CCAs (ndarray): Transformed single trial data in CCA space
                        (trials, samples, CCAs)
    """
    tmu = mu[np.newaxis,np.newaxis,subject]
    tsigma = sigma[np.newaxis,np.newaxis,subject]
    X -= tmu # centered
    PCAs = np.matmul(X,weights_pca[subject])
    PCAs /= tsigma # standardized 
    CCAs = np.matmul(PCAs,W[subject])
    return CCAs

def inverseTransformTrials(CCAs,W,mu,sigma,weights_pca,subject=0):
    """ Transform single trial data from CCA space back to sensor space.
    
    Parameters:
        CCAs (ndarray): Single trial data of one subject in CCA space
                        (trials, samples, CCAs)
        W (ndarray): CCA weights that transform PCAs to CCAs for each subject
                     (subjects, PCAs, CCAs)
        mu (ndarray): PCA mean (subjects, PCAs)
        sigma (ndarray): PCA standard deviation (subjects, PCAs)
        weights_pca (ndarray): PCA weights that tranform sensors to PCAs for each 
                               subject (subjects, sensors, PCAs)
        subject (int): Index of the subject whose data is being transformed 
                              
    Returns:
        X (ndarray): Single trial data transformed back into sensor space 
                     (trials, samples, sensors)
    """
    tmu = mu[np.newaxis,np.newaxis,subject]
    tsigma = sigma[np.newaxis,np.newaxis,subject]
    PCAs = np.matmul(CCAs,np.linalg.pinv(W[subject])) # trials x samples x PCAs
    PCAs *= tsigma # revert standardization
    X = np.matmul(PCAs,np.linalg.pinv(weights_pca[subject]))
    X += tmu # revert centering
    return X

def transformTrialsPCA(X,mu,sigma,weights_pca,subject=0):
    """ Transform single trial data from sensor space to PCA space
    
    Parameters:
        X (ndarray): Single trial data of one subject in sensor space 
                     (trials, samples, sensors)
        mu (ndarray): PCA mean (subjects, PCAs)
        sigma (ndarray): PCA standard deviation (subjects, PCAs)
        weights_pca (ndarray): PCA weights that tranform sensors to PCAs for each 
                               subject (subjects, sensors, PCAs)
        subject (int): Index of the subject whose data is being transformed 
                              
    Returns:
        PCAs (ndarray): Transformed single trial data in PCA space
                        (trials, samples, PCAs)
    """
    tmu = mu[np.newaxis,np.newaxis,subject]
    tsigma = sigma[np.newaxis,np.newaxis,subject]
    X -= tmu # centered
    PCAs = np.matmul(X,weights_pca[subject])
    PCAs /= tsigma # standardized 
    return PCAs

def testCCA(data_train,data_test,W,mu,sigma,weights_pca):
    """ Test if the inter-subject correlations/consistency observed from training
        data CCAs generalize over testing data.
    
    Parameters:
        data_train (ndarray): Training data in sensor space
                              (subjects, samples, sensors)
        data_test (ndarray): Test data in sensor space
                             (subjects, samples, sensors)
        W (ndarray): CCA weights that transform PCAs to CCAs for each subject
                     (subjects, PCAs, CCAs)
        mu (ndarray): PCA mean (subjects, PCAs)
        sigma (ndarray): PCA standard deviation (subjects, PCAs)
        weights_pca (ndarray): PCA weights that tranform sensors to PCAs for
                               each subject (subjects, sensors, PCAs)
    
    Returns:
        correlations (ndarray): Inter-subject correlations (averaged over 
                                every pair of subjects) over every CCA component
                                (first row: training data; second row: testing data)
    """
    tmu = mu[:,np.newaxis]
    tsigma = sigma[:,np.newaxis]
    data_train -= tmu # centered
    pca_train = np.matmul(data_train,weights_pca) # PCAs obtained
    pca_train /= tsigma # normalized
    cca_train = np.matmul(pca_train,W) # CCAs obtained
    data_test -= tmu # centered
    pca_test = np.matmul(data_test,weights_pca) # PCAs obtained
    pca_test /= tsigma # normalized
    cca_test = np.matmul(pca_test,W) # CCAs obtained
    
    K = W.shape[2]
    correlations = np.zeros((K,2))
    for i in range(K):
        intersubject_corr_train = np.corrcoef(np.squeeze(cca_train[:,:,i]))
        intersubject_corr_test = np.corrcoef(np.squeeze(cca_test[:,:,i]))
        # averaged inter-subject correlations over training data
        correlations[i,0] = np.mean(intersubject_corr_train)
        # averaged inter-subject correlations over testing data
        correlations[i,1] = np.mean(intersubject_corr_test)

    return correlations