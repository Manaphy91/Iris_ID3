import numpy as np
import id3lib as id3
import performances as perf
import reduced_error_pruning as rep

def get_accuracy(tr_mat, tr_res, results, attributes, prune=False, folds=10):

    dataset_len = len(tr_res)
    fl = round(dataset_len / folds)

    accuracy = 0
    for i in range(folds):

        if i == 0:
            train_mat = np.copy(tr_mat[fl:,:])
        else:
            train_mat1 = np.copy(tr_mat[:i*fl,:])
            train_mat2 = np.copy(tr_mat[(i+1)*fl:,:])
            train_mat = np.vstack((train_mat1, train_mat2))

        train_res = tr_res[:i*fl] + tr_res[(i+1)*fl:]

        test_mat = np.copy(tr_mat[i*fl:(i+1)*fl,:])
        test_res = tr_res[i*fl:(i+1)*fl]

        # generate tree from training data
        tree = id3.id3(train_mat, train_res, attributes)

        if prune:
            rep.prune(tree, test_mat, test_res)

        # calculate confusion matrix for current tree and testing data provided
        conf_mat = perf.calculate_confusion_matrix(tree, test_mat, test_res, results)
        # sum out the accuracy obtained from the conf mat calculated
        accuracy += perf.get_accuracy(conf_mat)

    # return average accuracy 
    return round(accuracy / folds, 3)
