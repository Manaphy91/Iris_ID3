#!/usr/bin/python3 -W ignore

import id3lib as id3
import numpy as np
import graph as gv
import performances as perf
import reduced_error_pruning as rep
import cross_validation as kfold
from sklearn.metrics import classification_report
from tqdm import tqdm


ATTRIBUTES = tuple(["fixed acidity", "volatile acidity", "citric acid", "residual sugar", \
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", \
    "sulphates", "alcohol"])

if __name__ == "__main__":
    dataset = id3.read_dataset_from_csv("winequality-red.csv", delimiter=';')

    RESULTS = tuple(set(id3.get_results(dataset)))
    
    training, test = id3.shuffle_and_split_dataset(dataset)

    tr_mat, tr_res = id3.discard_results(training), id3.get_results(training)

    tr_mat = np.array(id3.strings_to_numbers(tr_mat))

    te_mat, te_res = id3.discard_results(test), id3.get_results(test)

    te_mat = np.array(id3.strings_to_numbers(te_mat))

    ################# Original Tree Making Of #################

    tree = id3.id3(tr_mat, tr_res, ATTRIBUTES)

    kaccuracy = kfold.get_accuracy(tr_mat, tr_res, RESULTS, ATTRIBUTES)

    predicted = []
    pbar = tqdm(total=len(te_mat), desc='', leave=False)
    for i in range(te_mat.shape[0]):
        res = id3.get_class(tree, te_mat[i,:])
        predicted.append(res)
        # print("Inferred class: {} expected class: {}".format(res, te_res[i]))
        pbar.update(1)
    pbar.close()

    gv.create_graph(tree, 'ID3 Wine quality Decision Tree')

    conf_mat_tr = perf.calculate_confusion_matrix(tree, tr_mat, tr_res, RESULTS)
    conf_mat_te = perf.calculate_confusion_matrix(tree, te_mat, te_res, RESULTS)

    print("------------Original tree------------")
    print("KFold Cross Validation Accuracy on training set: {}".format(kaccuracy))
    print("Accuracy on training set: {}".format(perf.get_accuracy(conf_mat_tr)))
    print("Accuracy on testing set: {}".format(perf.get_accuracy(conf_mat_te)))
    print("Classification report:\n")
    print(classification_report(te_res, predicted))

    ################# Pruned Tree Making Of #################

    rep.prune(tree, te_mat, te_res)

    kaccuracy = kfold.get_accuracy(tr_mat, tr_res, RESULTS, ATTRIBUTES, prune=True)

    predicted = []
    pbar = tqdm(total=len(te_mat), desc='', leave=False)
    for i in range(te_mat.shape[0]):
        res = id3.get_class(tree, te_mat[i,:])
        predicted.append(res)
        # print("Inferred class: {} expected class: {}".format(res, te_res[i]))
        pbar.update(1)
    pbar.close()

    gv.create_graph(tree, 'ID3 Wine quality Decision Tree - Pruned')

    conf_mat_tr = perf.calculate_confusion_matrix(tree, tr_mat, tr_res, RESULTS)
    conf_mat_te = perf.calculate_confusion_matrix(tree, te_mat, te_res, RESULTS)

    print("------------Pruned tree------------")
    print("KFold Cross Validation Accuracy on training set: {}".format(kaccuracy))
    print("Accuracy on training set: {}".format(perf.get_accuracy(conf_mat_tr)))
    print("Accuracy on testing set: {}".format(perf.get_accuracy(conf_mat_te)))
    print("Classification report:\n")
    print(classification_report(te_res, predicted))
