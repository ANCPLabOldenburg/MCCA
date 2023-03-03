import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=None,
                          scores=None):
    """ This function plots the confusion matrix.
    
    Normalization can be applied by setting `normalize=True`.
    """
        
    if scores is not None:
        score = np.mean(scores)
        err = np.std(scores)
    else:
        score = balanced_accuracy_score(y_true, y_pred)
        err = None
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
        
    fig, ax = plt.subplots(1)
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #ax.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = (cm.max()+cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if err is not None:
        plt.text(2.8, 1, "BA: %0.2f (+/- %0.2f)" % (score,err),fontsize=14)
    else:
        plt.text(2.8, 1, "BA: %0.2f" % score,fontsize=14)
    plt.tight_layout()
    # plt.show()
    if save:
        fig.savefig(save,format='svg')