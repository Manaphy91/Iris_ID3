import csv
import operator
import numpy
import functools
import random
import tree

def read_dataset_from_csv(csv_name, delimiter=","):
    lst = []
    try:
        with open(csv_name, "r") as src:
            reader = csv.reader(src, delimiter=delimiter)
            for line in reader:
                lst.append(line)
    except OSError:
        print("Error: the file named " + csv_name + " can't be opened!")

    return lst

def discard_results(dataset):
    lst = []
    for r in dataset:
        lst.append(r[:-1])
    return lst

def get_results(dataset):
    lst = []
    for r in dataset:
        lst.append(r[-1])
    return lst

def strings_to_numbers(dataset):
    if len(dataset) ==  0:
        return []
    for column in range(len(dataset[0])):
        for row in range(len(dataset)):
            if len(dataset[row][column]) > 0:
                try:
                    new_val = float(dataset[row][column])
                    dataset[row][column] = new_val
                except ValueError:
                    break

    return dataset

def get_entropy(vec):
    ''' Get the entropy from one vector of values
    '''

    def partial(elem):
        if elem == 0.0:
            return 0.0
        else:
            _res = - elem * numpy.log2(elem)
            return numpy.around(_res.astype(numpy.double), decimals=3)

    if numpy.sum(vec) != 1.0:
       vec = vec / numpy.sum(vec)

    func = numpy.frompyfunc(partial, 1, 1)

    return numpy.sum(func(vec))

def get_information_gain(entropy, len, lst):
    ''' Take entropy of the entire dataset, the length of the attribute value
    list and the list of values and produces the information gain.
    '''
    res = 0
    for el in lst:
        res += (sum([v for _, v in el.items()]) / len) * \
            get_entropy([v for _, v in el.items()])
    return entropy - res

def shuffle_and_split_dataset(dataset):
    ''' Peforms shuffling and splitting of dataset using the Pareto law: 80%
    usable for training and the 20% remaining for testing.
    Returns two numpy arrays containing 80% and 20% of original dataset
    provided as argument.
    '''
    split_ok = False

    # fix a seed and use it for any splitting operation in order to obatin
    # after a specific number of iterations, always the corrispondent set
    SEED = 53
    random.seed(SEED)

    while not split_ok:

        random.shuffle(dataset)

        items = len(dataset)
        training = round(items * 0.80)
        test = items - training

        tr_set = dataset[:training]
        te_set = dataset[training:]

        result_column = set(get_results(dataset))

        # balance the dataset split: all the known results should belong to
        # both the set results of the split
        if result_column == set(get_results(tr_set)) and \
            result_column == set(get_results(te_set)):
            split_ok = True
        print("Splitting performed not balanced: some outcomes are missing " + \
            "from one of the sets! Perform a new splitting!")

    return tr_set, te_set

def take_count(dict, elem):
    ''' Use a dict as counter for element into elem variable.
    '''
    if elem in dict:
        dict[elem] = dict[elem] + 1
    else:
        dict[elem] = 1
    return dict

def accumulate(dict, couple):
    ''' Memorize in a dict the sum of the values at position 1 of couples
    provided to the method in object and having the same value at postion 0.
    '''
    if couple[0] in dict:
        dict[couple[0]] += couple[1]
    else:
        dict[couple[0]] = couple[1]
    return dict

def get_greater(dict):
    ''' Take as argument a dict having comparable values and returns
    the key having the maximum value.
    '''
    max_key = None
    max_val = -1
    for k, v in dict.items():
        if max_val < v:
            max_key = k
            max_val = v
    return max_key

def get_best_mean_point(entropy, att_res_lst):
    lst = sorted(att_res_lst, key=operator.itemgetter(0))
    dst = []
    prev_res, prev_val = None, None
    count = 0
    for val, res in lst:

        if prev_res != None and prev_res != res:
            dst.append([(val + prev_val) / 2, prev_res, count])
            count = 1
        else:
            count += 1
    
        prev_val, prev_res = val, res

    dst.append([prev_val, prev_res, count])

    attr_gain_lst = []
    for i in range(len(dst)):
        if i == len(dst) - 1:
            pass
        else:
            lst = []
            res_lst = list(map(lambda x: x[1:], dst[:i + 1]))
            lst.append(functools.reduce(accumulate, res_lst, {}))
            res_lst = list(map(lambda x: x[1:], dst[i + 1:]))
            lst.append(functools.reduce(accumulate, res_lst, {}))
            # calculate the information gain for the two parts of the interval
            # contained in the list lst
            info_gain = get_information_gain(entropy, len(att_res_lst), lst)
            # add to attr_gain_lst a couple made from the mean point and the
            # information gain just calculated
            attr_gain_lst.append([round(dst[i][0], 3),
                numpy.around(info_gain.astype(numpy.double), decimals=3)])
    
    # return the couple having the max value at position 1
    return max(attr_gain_lst, key=operator.itemgetter(1))

def id3(matrix, result, attr_lst):
    # check if all outcomes are equal, if this is the case set this as Node
    # value and return
    if len(set(result)) == 1:
        value = set(result).pop()
        return tree.Node(value, [], prob=100.0, approx_value=value)
    # check if at least one column is present in the matrix, if this is not
    # the case it means that the value for the node to return have to be set
    # to the most likely one
    else:
        # obtain a dictionary containing for each result the number of
        # instances having this expected result
        results = functools.reduce(take_count, result, {})
        # calculate the entropy from the dictionary obtained above 
        entropy = get_entropy([v for _, v in results.items()])
        attr = []
        for j in range(matrix.shape[1]):
            # obtain best mean point of column j

            best_point = get_best_mean_point(entropy, list(map(lambda x, y: [x, y], \
                matrix[:,j], result)))

            attr.append([j] + best_point)

        split_ok = False
        while not split_ok:
            # get the feature having the maximum information gain between the
            # ones collected before
            best_point = max(attr, key=operator.itemgetter(2))

            # construct the node
            root = tree.Node(lambda x: x[best_point[0]] < best_point[1], [])
            root.set_treshold_name(attr_lst[best_point[0]])
            root.set_treshold_value(best_point[1])
            root.set_approx_value(get_greater(results))
            
            j = best_point[0]
            left_mat, left_res = [], []
            right_mat, right_res = [], []
            # perform the following dicotomy:
            # - instances having feature j less than the selected one(best
            # point), to be put into the left_mat matrix
            # - othe instances to be put into the right_mat matrix
            for i in range(matrix.shape[0]):
                if matrix[i,j] < best_point[1]:
                    left_mat.append(i)
                    left_res.append(result[i])
                else:
                    right_mat.append(i)
                    right_res.append(result[i])

            if len(left_mat) == 0 or len(right_mat) == 0:
                # it seems all the instances adhere to the same test condition,
                # this brought to obtain one of the two splittings equal to
                # the empty set
                # SOLUTION: remove the attribute from the attr array and
                # perform a new splitting based on a different feature
                split_ok = False
                del attr[attr.index(best_point)]
            else:
                split_ok = True

        left = id3(matrix[left_mat], left_res, attr_lst)
        left.set_parent(root)

        right = id3(matrix[right_mat], right_res, attr_lst)
        right.set_parent(root)

        root.set_sons([left, right])

        return root

def get_class(tree, attr_lst):
    if not callable(tree.get_value()):
        return tree.get_value()
    else:
        func = tree.get_value()
        if func(attr_lst):
            return get_class(tree.get_sons()[0], attr_lst)
        else:
            return get_class(tree.get_sons()[1], attr_lst)
