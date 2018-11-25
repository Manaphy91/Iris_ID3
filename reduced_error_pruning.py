import tree
import id3lib as id3

def is_leaf(node):
    ''' A node without any child is a leaf.
    Returns True if node is a leaf, False otherwise.
    '''
    return len(node.get_sons()) == 0

def make_stats(root, attr_lst, res):
    ''' Add error statistics to a tree.
    '''
    approx_value = root.get_approx_value()
    # if approximated value is different by res, increment errors of the node
    # by one
    if approx_value != res:
        root.add_errors()

    func = root.get_value()
    # if the node value is a function, the node is not a leaf and the function
    # has to recur on node's sons
    if callable(func):
        # if func returns True, the attribute selected is lesser than the
        # threshold, make_stats recurs on the left son
        if func(attr_lst):
            make_stats(root.get_sons()[0], attr_lst, res)
        else:
            # otherwise, it recurs on the right son
            make_stats(root.get_sons()[1], attr_lst, res)

def prune_tree(root):
    # if root is a leaf node, recur
    if is_leaf(root):
        return

    sons = root.get_sons()
    for son in sons:
        # run prune_tree on each child of the current node
        prune_tree(son)

    # if the current node is the father of two leafs
    if is_leaf(sons[0]) and is_leaf(sons[1]):
        # if the errors made on the two leafs are greater than the ones made on
        # the current node
        if (sons[0].get_errors() + sons[1].get_errors()) >= root.get_errors():
            # remove the sons and substitute the threshold function with and
            # approximate value
            root.set_sons([])
            root.set_value(root.get_approx_value())

def prune(tree, val_set, val_res):
    # call make_stats subroutine on all raws pf the provided matrix
    for i in range(len(val_res)):
        make_stats(tree, val_set[i,:], val_res[i])

    # then, prune the three
    prune_tree(tree)
