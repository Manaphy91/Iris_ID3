#!/bin/python3

import id3lib as id3
import numpy as np
import graph as gv
import performances as perf
import reduced_error_pruning as rep

ATTRIBUTES = ('Sepal length', 'Sepal width', 'Petal length', 'Petal width')

dataset = id3.read_dataset_from_csv("iris.data")

RESULTS = tuple(set(id3.get_results(dataset)))

training, test = id3.shuffle_and_split_dataset(dataset)

tr_mat, tr_res = id3.discard_results(training), id3.get_results(training)

tr_mat = np.array(id3.strings_to_numbers(tr_mat))

tree = id3.id3(tr_mat, tr_res, ATTRIBUTES)

te_mat, te_res = id3.discard_results(test), id3.get_results(test)

te_mat = np.array(id3.strings_to_numbers(te_mat))

for i in range(te_mat.shape[0]):
    res = id3.get_class(tree, te_mat[i,:])
    print("Inferred class: {} expected class: {}".format(res, te_res[i]))

gv.create_graph(tree, 'ID3 Decision Tree')

conf_mat_1 = perf.calculate_confusion_matrix(tree, te_mat, te_res, RESULTS)

print("original accuracy: {}".format(perf.get_accuracy(conf_mat_1)))

rep.prune(tree, te_mat, te_res)

gv.create_graph(tree, 'ID3 Decision Tree - Pruned')

conf_mat_2 = perf.calculate_confusion_matrix(tree, te_mat, te_res, RESULTS)

print("pruned tree accuracy: {}".format(perf.get_accuracy(conf_mat_2)))
