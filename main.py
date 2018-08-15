#!/bin/python3

import id3lib as id3
import numpy as np

dataset = id3.read_dataset_from_csv("iris.data")

training, test = id3.shuffle_and_split_dataset(dataset)

tr_mat, tr_res = id3.discard_results(training), id3.get_results(training)

tr_mat = np.array(id3.strings_to_numbers(tr_mat))

tree = id3.id3(tr_mat, tr_res)

te_mat, te_res = id3.discard_results(test), id3.get_results(test)

te_mat = np.array(id3.strings_to_numbers(te_mat))

for i in range(te_mat.shape[0]):
    res = id3.get_class(tree, te_mat[i,:])
    print("Inferred class: {} expected class: {}".format(res, te_res[i]))
