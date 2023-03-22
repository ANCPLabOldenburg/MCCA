import fnmatch
import os

import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG
from scoring import plot_confusion_matrix


def visualize_permutation_test(save_fn='permutation', normalize=False,
                               folder_name='permutation_test'):
    """ Histogram plot for permutation test results. """
    save_folder = CONFIG.results_folder + folder_name + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder + save_fn + '_'
    temp_path = save_folder + 'temp/' + save_fn + '/'
    observed_fn = temp_path + save_fn + '_observed' + '.npz'
    if os.path.exists(observed_fn):
        npzfile = np.load(observed_fn)
        observed_scores = npzfile['scores']
        y_true = npzfile['y_true']
        y_pred = npzfile['y_pred']
        save = save_path + 'observed'
        plot_confusion_matrix(y_true, y_pred, ['bottle', 'pencil', 'cup'], save=save,
                              scores=observed_scores, normalize=normalize)
    else:
        observed_scores = None

    perm_fns = fnmatch.filter(os.listdir(temp_path), save_fn + '_perm*.npz')
    perm_y_true = []
    perm_y_pred = []
    perm_scores = []
    for fn in perm_fns:
        npzfile = np.load(temp_path + fn)
        perm_y_true.append(npzfile['y_true'])
        perm_y_pred.append(npzfile['y_pred'])
        perm_scores.append(npzfile['scores'])

    n_permutations = len(perm_scores)
    #    for i in range(n_permutations):
    #        save = save_path+'perm'+str(i)
    #        plot_confusion_matrix(perm_y_true[i],perm_y_pred[i],['bottle','pencil','cup'],
    #                              save=save,scores=perm_scores[i],normalize=normalize)

    perm_scores = np.mean(np.array(perm_scores), axis=1)
    if observed_scores is not None:
        observed_scores_mean = np.mean(observed_scores)
        C = np.count_nonzero(perm_scores >= observed_scores_mean, axis=0)
        pvalue = (C + 1) / (n_permutations + 1)
        print("p-value: %0.4f" % pvalue)

    perm_y_true = np.concatenate(perm_y_true)
    perm_y_pred = np.concatenate(perm_y_pred)
    #    save = save_path + 'all_perm' + regstr
    #    plot_confusion_matrix(perm_y_true,perm_y_pred,['bottle','pencil','cup'],
    #                          save=save,scores=perm_scores,normalize=normalize)

    fig, ax = plt.subplots(1)
    ax.hist(perm_scores, bins=16)
    if observed_scores is not None:
        #        ax2 = ax.twinx()
        #        ax2.hist(observed_scores,bins=60, color='r')
        #        ax2.set_ylabel('Count (observed)',color='r')
        #        ax2.tick_params(axis='y', labelcolor='r')
        #        ax2.set_yticks([0,1,2])
        ax.axvline(observed_scores_mean, color='r', linestyle='-', label='Observed mean')
    ax.set_xlabel('Balanced accuracy')
    ax.set_ylabel('Count (permutations)')
    #    ax.tick_params(axis='y', labelcolor='b')
    fig.tight_layout()
    plt.show()
    fig.savefig(save_path + 'hist', format='svg')


def visualize_online():
    """ Balanced accuracy vs number of trials plot for simulated online decoding results. """
    save_folder = CONFIG.results_folder
    result_folder = save_folder + 'online/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    fns = fnmatch.filter(os.listdir(result_folder), 'online_*trials.npz')
    import re
    n_trials = []
    BAs = []
    err = []
    for fn in fns:
        n_trial = int(re.sub('[^0-9]', '', fn))
        n_trials.append(n_trial)
        npzfile = np.load(result_folder + fn)
        scores = npzfile['scores']
        BAs.append(np.mean(scores))
        err.append(np.std(scores))

    n_trials, BAs, err = zip(*sorted(zip(n_trials, BAs, err)))
    BAs = np.array(BAs)
    err = np.array(err)

    fig, ax = plt.subplots(1)
    ax.plot(n_trials, BAs)
    plt.xticks(np.arange(0, 101, 5))
    ax.fill_between(n_trials, BAs + err, BAs - err, alpha=0.3)
    ax.set_xlabel('Number of trials per class')
    ax.set_ylabel('Balanced accuracy')
    fig.tight_layout()
    plt.show()
    fig.savefig(save_folder + CONFIG.save_fn, format='svg')


def visualize_decoder():
    """ Plot confusion matrix for decoding results. """
    save_folder = CONFIG.results_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    temp_file = save_folder + CONFIG.save_fn + '.npz'
    if os.path.exists(temp_file):
        npzfile = np.load(temp_file)
        y_true = npzfile['y_true']
        y_pred = npzfile['y_pred']
        BAs = npzfile['scores']
        save = save_folder + CONFIG.save_fn + '_cm'
        plot_confusion_matrix(y_true, y_pred, ['bottle', 'pencil', 'cup'], save=save, scores=BAs)
