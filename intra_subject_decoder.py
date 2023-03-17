import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import prepare_data
from MCCA import MCCA


def intra_subject_decoder(n_components_pca=50, n_components_mcca=10, r=0, tsss_realignment=False, n_folds=5,
                          save_fn='intra_subject', mode='MCCA'):
    """
    Splits data from each subject into 80% training and 20% testing. MCCA is applied
    to averaged training data from all subjects, and the weights are used to transform
    all single-trial data into MCCA space. Intra-subject decoders are trained on
    single-trial data in MCCA space for each subject and the results are saved to file.

    Parameters:
        n_components_pca (int): Number of PCA components to retain for each subject (default 50)

        n_components_mcca (int): Number of MCCA components to retain (default 10)

        r (int/float): Regularization strength. (default 0)

        tsss_realignment (bool): Whether to use tSSS realignment

        n_folds (int): Number of cross-validation folds

        save_fn (str): Filename to save results

        mode (str): Feature space to use for decoding. Must be one of ['MCCA', 'PCA', 'sensorspace']

    """
    save_path = prepare_data.results_folder + 'intra_subject/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    X, y, train, test, data_averaged = _load_data(tsss_realignment=tsss_realignment, n_folds=n_folds)
    y_true_all = []
    y_pred_all = []
    BAs = []
    n_subjects = len(y)
    for i in range(n_subjects):
        for j in range(n_folds):
            if mode == 'MCCA':
                mcca = MCCA(n_components_pca, n_components_mcca, r)
                mcca.obtain_mcca(data_averaged[j])
                X_mcca = mcca.transform_trials(X[i], subject=i).reshape((len(y[i]), -1))
            elif mode == 'PCA':
                mcca = MCCA(n_components_pca, n_components_mcca, r, pca_only=True)
                mcca.obtain_mcca(data_averaged[j])
                X_mcca = mcca.transform_trials_pca(X[i], subject=i).reshape((len(y[i]), -1))
            elif mode == 'sensorspace':
                X_mcca = X[i].reshape((len(y[i]), -1))
            else:
                raise Exception('Mode must be one of [MCCA, PCA, sensorspace].')
            X_train = X_mcca[train[j][i]]
            y_train = y[i][train[j][i]]
            X_test = X_mcca[test[j][i]]
            y_test = y[i][test[j][i]]
            clf = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l2', random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_true_all.append(y_test)
            y_pred_all.append(y_pred)
            BAs.append(balanced_accuracy_score(y_test, y_pred))
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    reg_str = '_reg' + str(r) if r else ''
    save = save_path + save_fn + reg_str + ".npz"
    np.savez(save, y_true=y_true_all, y_pred=y_pred_all, scores=BAs)


def _load_data(tsss_realignment=False, n_folds=5):
    """
    Splits data from each subject into 80% training and 20% testing. MCCA is applied
    to averaged training data from all subjects, and the weights are used to transform
    all single-trial data into MCCA space.

    Parameters:
        tsss_realignment (bool): Whether to use tSSS realignment

        n_folds (int): Number of cross-validation folds

    Returns:
        X (ndarray): The transformed data in MCCA space, flattened across MCCA
            and time dimensions for the classifier.
            (subjects x trials x (samples x MCCAs))

        y (ndarray): Labels corresponding to trials in X. (subjects x trials)

        train_indices (list): Indices of trials used in the training set for
            each subject.

        test_indices (list): Indices of trials used in the test set for each
            subject.
    """
    X = []
    y = []
    train_indices = [[] for _ in range(n_folds)]
    test_indices = [[] for _ in range(n_folds)]
    data_averaged = [[] for _ in range(n_folds)]
    X_, y_ = prepare_data.load_single_trial_data(tsss_realignment=tsss_realignment)
    n_subjects = len(y_)
    for i in range(n_subjects):
        data_st, labels = X_[i], y_[i]
        X.append(data_st)
        y.append(labels)
        sss = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        j = 0
        for train, test in sss.split(data_st, labels):
            train_indices[j].append(train)
            test_indices[j].append(test)
            evoked_bottle = np.mean(data_st[train][np.where(labels[train] == 0)], axis=0)
            evoked_pencil = np.mean(data_st[train][np.where(labels[train] == 1)], axis=0)
            evoked_cup = np.mean(data_st[train][np.where(labels[train] == 2)], axis=0)
            data_averaged[j].append(np.concatenate([evoked_bottle, evoked_pencil, evoked_cup], axis=0))
            j += 1
    for j in range(n_folds):
        data_averaged[j] = np.stack(data_averaged[j], axis=0)

    return X, y, train_indices, test_indices, data_averaged
