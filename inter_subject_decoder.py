import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
# import xgboost as xgb

import prepare_data
from MCCA_transformer import MCCATransformer
from config import CONFIG


def inter_subject_decoder():
    """ Inter-subject decoder using leave-one-subject-out cross-validation

    MCCA is computed based on averaged data from all but one subjects. MCCA projection weights
    for the left-out subject are estimated via linear regression from the trial-averaged
    timeseries of the left-out subject in PCA space to the average of other subjects'
    trial-averaged timeseries in MCCA space. All single-trial data is transformed
    into MCCA space with the computed weights.
    Inter-subject classifiers are trained on all but one subject and tested on the
    left-out subject.

    """
    save = CONFIG.results_folder + CONFIG.save_fn + '.npz'
    print(save)

    X, y = prepare_data.load_single_trial_data(tsss_realignment=CONFIG.tsss_realignment)

    n_subjects = y.shape[0]
    y_true_all = []
    y_pred_all = []
    BAs = []

    for i in range(n_subjects):
        X_train, X_test, y_train, y_test = _transform_data(X, y, i)
        clf = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l2', random_state=0)
        # clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', random_state=0)
        # model = xgb.XGBClassifier(n_jobs=1)
        # clf = GridSearchCV(model, {"max_depth": [2, 4], "n_estimators": [50, 100]}, n_jobs=32)
        # clf = xgb.XGBClassifier(n_jobs=32)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        score = balanced_accuracy_score(y_test, y_pred)
        print(i, score)
        BAs.append(score)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    np.savez(save, y_true=y_true_all, y_pred=y_pred_all, scores=BAs)


def _transform_data(X, y, left_out_subject, permute=False, seed=0):
    """
    Apply MCCA to averaged data from all but one subjects. MCCA projection
    weights are estimated via linear regression from the trial-averaged
    timeseries of the left out subject in PCA space and the average of other
    subjects' trial-averaged timeseries in MCCA space. All single-trial data
    is transformed into MCCA space.

    Parameters:
        X (ndarray): The training data (subjects x trials x samples x channels)

        y (ndarray): Labels corresponding to trials in X (subjects x trials)

        left_out_subject (int): Index of the left-out subject

        permute (bool): Whether to shuffle labels for permutation testing

        seed (int): Seed for shuffling labels when permute is set to True

    Returns:
        X_train (ndarray): The transformed data from all but the left out
            subject in MCCA space, flattened across MCCA and time dimensions for
            the classifier. Trials are concatenated across subjects.
            (trials x (samples x MCCAs))

        X_test (ndarray): The transformed data from the left out subject in MCCA
            space, flattened across MCCA and time dimensions for the classifier.
            (trials x (samples x MCCAs))

        y_train (ndarray): Labels corresponding to trials in X_train.

        y_test (ndarray): Labels corresponding to trials in X_test.

    """
    if permute:
        y = _random_permutation(y, seed)
    n_subjects = y.shape[0]
    n_classes = len(np.unique(y))
    leave_one_out = np.setdiff1d(np.arange(n_subjects), left_out_subject)
    transformer = MCCATransformer(CONFIG.n_pcs, CONFIG.n_ccs, CONFIG.r, CONFIG.new_subject_trials == 'nested_cv')
    if CONFIG.mode == 'MCCA':
        X_train = transformer.fit_transform(X[leave_one_out], y[leave_one_out], )
        y_train = np.concatenate(y[leave_one_out], axis=0)
        X_left_out = X[left_out_subject]
        y_left_out = y[left_out_subject]

        if CONFIG.new_subject_trials in ['all', 'nested_cv']:
            X_test, y_test = transformer.fit_transform_online(X_left_out, y_left_out)
        else:
            if CONFIG.new_subject_trials == 'split':
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
                train, test = next(sss.split(X_left_out, y_left_out))
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=CONFIG.new_subject_trials * n_classes,
                                             test_size=None, random_state=0)
                train, test = next(sss.split(X_left_out, y_left_out))
            transformer.fit_online(X_left_out[train], y_left_out[train])
            X_test = transformer.transform_online(X_left_out[test])
            y_test = y_left_out[test]
    elif CONFIG.mode == 'PCA':
        transformer.fit(X[leave_one_out], y[leave_one_out])
        X_train = transformer.transform_pca_only(X[leave_one_out])
        y_train = np.concatenate(y[leave_one_out], axis=0)
        X_left_out = X[left_out_subject]
        y_left_out = y[left_out_subject]
        transformer.fit_online(X_left_out, y_left_out)
        X_test = transformer.transform_online_pca_only(X_left_out)
        y_test = y_left_out
    elif CONFIG.mode == 'sensorspace':
        X_train = np.concatenate([X[i] for i in leave_one_out])
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_train = _standardize(X_train, axes=(1,)).astype(np.float64)
        y_train = np.concatenate(y[leave_one_out], axis=0)
        X_test = X[left_out_subject].reshape((X[left_out_subject].shape[0], -1))
        X_test = _standardize(X_test, axes=(1,)).astype(np.float64)
        y_test = y[left_out_subject]
    else:
        raise Exception('Mode must be one of [MCCA, PCA, sensorspace].')

    return X_train, X_test, y_train, y_test


def _random_permutation(y, seed):
    """ Randomly permute labels for each subject using the provided seed. """
    tmp = []
    for i in range(len(y)):
        tmp.append(np.random.RandomState(seed=seed).permutation(y[i]))
    return np.array(tmp)


def _standardize(x, axes=(2,)):
    """ Standardize x over specified axes. """
    mean_ = np.apply_over_axes(np.mean, x, axes)
    std_ = np.apply_over_axes(np.std, x, axes)
    return (x - mean_) / std_


def permutation_test(n_permutations, start_id=0, n_jobs=-1):
    """ Parallelized permutation test of inter-subject decoder

    Parameters:
        n_permutations: Number of permutations to compute

        start_id: Start seed for shuffling labels (default 0)

        n_jobs (int): Number of jobs to use for parallel execution

    """
    save_folder = CONFIG.results_folder + 'permutation_test/temp/' + CONFIG.save_fn + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    X, y = prepare_data.load_single_trial_data(tsss_realignment=CONFIG.tsss_realignment)
    save_path_observed = save_folder + CONFIG.save_fn + '_observed.npz'
    _leave_one_out_cv(X, y, False, None, save_path_observed)

    from joblib import dump, load
    mmap_fn = CONFIG.results_folder + 'permutation_test/temp/X.mmap'
    if not os.path.exists(mmap_fn):
        dump(X, mmap_fn)
    X = load(mmap_fn, mmap_mode='r')
    import gc
    gc.collect()
    save_path = save_folder + CONFIG.save_fn + '_perm%d.npz'
    Parallel(n_jobs=n_jobs, backend='loky', verbose=100, mmap_mode='r') \
        (delayed(_leave_one_out_cv)(X, y, True, i + start_id, save_path) for i in range(n_permutations))


def _leave_one_out_cv(X, y, perm, seed, save):
    """ Wraps leave-one-subject-out cross-validation loop for parallelization in permutation test. """
    if perm:
        save = save % seed
    if os.path.exists(save):
        print('File already exists: ' + save)
        return
    n_subjects = y.shape[0]
    y_true_all = []
    y_pred_all = []
    BAs = []
    for i in range(n_subjects):
        X_train, X_test, y_train, y_test = _transform_data(X, y, i, permute=perm, seed=seed)
        clf = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l2', random_state=0)
        # clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        BAs.append(balanced_accuracy_score(y_test, y_pred))
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    np.savez(save, y_true=y_true_all, y_pred=y_pred_all, scores=BAs)
