#!/usr/bin/python3 -W ignore

import id3lib as id3
import numpy as np
import graph as gv
import performances as perf
import reduced_error_pruning as rep
from sklearn.metrics import classification_report

ATTRIBUTES = tuple(map(lambda x: "point#{}".format(x), range(0, 256))) 

dataset = id3.read_dataset_from_csv("semeion.data")

RESULTS = tuple(set(id3.get_results(dataset)))
    
training, test = id3.shuffle_and_split_dataset(dataset)

tr_mat, tr_res = id3.discard_results(training), id3.get_results(training)

tr_mat = np.array(id3.strings_to_numbers(tr_mat))

tree = id3.id3(tr_mat, tr_res, ATTRIBUTES)

te_mat, te_res = id3.discard_results(test), id3.get_results(test)

te_mat = np.array(id3.strings_to_numbers(te_mat))

print("------------Original tree------------")

predicted = []
pbar = tqdm(total=len(combs), desc='', leave=False)
for i in range(te_mat.shape[0]):
    res = id3.get_class(tree, te_mat[i,:])
    predicted.append(res)
    # print("Inferred class: {} expected class: {}".format(res, te_res[i]))
    pbar.update(1)
pbar.close()

gv.create_graph(tree, 'ID3 Decision Tree')

conf_mat_1 = perf.calculate_confusion_matrix(tree, te_mat, te_res, RESULTS)

print("Aaccuracy: {}".format(perf.get_accuracy(conf_mat_1)))
print("Classification report:\n")
print(classification_report(te_res, predicted))

rep.prune(tree, te_mat, te_res)

print("------------Pruned tree------------")
predicted = []
pbar = tqdm(total=len(combs), desc='', leave=False)
for i in range(te_mat.shape[0]):
    res = id3.get_class(tree, te_mat[i,:])
    predicted.append(res)
    # print("Inferred class: {} expected class: {}".format(res, te_res[i]))
    pbar.update(1)
pbar.close()

gv.create_graph(tree, 'ID3 Decision Tree - Pruned')

conf_mat_2 = perf.calculate_confusion_matrix(tree, te_mat, te_res, RESULTS)

print("Accuracy: {}".format(perf.get_accuracy(conf_mat_2)))
print("Classification report:\n")
print(classification_report(te_res, predicted))

