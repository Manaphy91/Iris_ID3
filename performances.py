import numpy as np
import id3lib as id3

def calculate_confusion_matrix(tree, test_mat, test_res, res_lst):
    cols = len(res_lst)
    conf_mat = np.zeros((cols, cols))
    
    index_dct = dict(zip(res_lst, range(cols)))

    for i in range(test_mat.shape[0]):
        res = id3.get_class(tree, test_mat[i,:])

        conf_mat[index_dct[test_res[i]],index_dct[res]] += 1

    return conf_mat

def get_accuracy(conf_mat):
    ''' Returns the accuracy starting from a confusion matrix.
    '''
    cols = conf_mat.shape[1]

    acc = np.sum(conf_mat[np.arange(cols),np.arange(cols)]) / \
        np.sum(np.sum(conf_mat, axis=0))

    return np.around(acc.astype(np.double), decimals=3)
